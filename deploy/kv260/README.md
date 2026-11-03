# TARS-mini on KV260 — Qwen3.5-0.8B 上板指南

本目录是完整的上板包：比特流 + 驱动 + 模型数据 + 自检。核心配置为 R=8, C=2, VG=1，100 MHz，
通过 S_AXI_HP0_FPD 访问 PS 侧 DDR。Qwen3.5-0.8B（真实权重，W8A8）全部 24 层、final norm 和
248320 词表的 head 都在 TARS-mini 上执行；A53 只做分词、embedding 查表和 argmax。

## 验证状态（PC 上完成）

- 比特流（v3.2：去掉脉动阵列，GEMM 全部走 GEMV 单元（不再支持 conv 和 K > 4096 的 GEMM，Qwen 用不到）；
  四个读口——核心每拍读 128 字节，按 1 KB 地址交错分到 HP0 / HP3 / HP2 / HP1（HP1 与写口共用 SmartConnect），
  各 128 位 @ 250 MHz，经核心顶层内的异步 FIFO 跨时钟；写口 256 位；GEMV 单元一次至多 4 行、每拍 128 字节权重；
  权重 128 字节对齐）：Vivado 2025.2，XCK26，100 / 250 MHz 时序收敛（WNS +0.176 ns，WHS +0.010 ns）；
  LUT 100,743 (86.0%)、DSP 879、BRAM 118。
- **实板（v3.2，2026-11-03）：`run.sh hw` 24/24、`run.sh selftest` ALL PASS，每一步 logits 与 golden model 逐比特一致；
  decode 约 15.7 M 周期/token ≈ 0.16 s/token（v3.0：0.21）；20 token 的对话提示分块 prefill 2.29 s ≈ 0.115 s/token；
  对话示例全程 4.5 s（v3.0：4.9 s）。** v3.0 的固件和数据在 `firmware_v30/`、`data_v30/`（换回：把两个目录改名回 `firmware`、`data`，
  `SHA256SUMS.data.v30` 改回 `SHA256SUMS.data`，再 `sudo ./setup.sh`）。
- v3.0（实板验证过）：两个读口，每拍 64 字节；`run.sh selftest` ALL PASS，每一步 logits 与 golden model 逐比特一致；
  decode 约 21 M 周期/token ≈ 0.21 s/token；提示词按 4 个 token 分块 prefill ≈ 0.12 s/token
  （v1.0：2.6 s/token；v1.1：0.90；v1.3：0.73；v1.4：0.60；v2.0：0.36）。
- RTL 回归（FPGA 版，经 KV260 顶层：AXI-Lite + 四读口突发桥 + 页表打乱 + 随机背压/延迟）386 通过 / 22 跳过（conv 与阵列专用用例）、核心 testbench 425/425 逐字节一致。
- **完整 Qwen3.5-0.8B（真实权重）经本目录的板上软件（`tars_board.Qwen`，后端换成 RTL 仿真）与
  KV260 顶层 RTL 运行，logits 与 golden model 逐比特一致**；单 token 84.8 M 周期 ≈ **0.85 s/token**
  （100 MHz，仿真内存模型；实际 DDR 时延下可能略慢）。
- 可综合的结构化浮点单元（比特流里实际的运算）跑 2 层真实维度 Qwen：逐比特一致。

## 目录

| 路径 | 内容 |
|---|---|
| `firmware/tars_kv260/` | `tars_kv260.bit.bin`、`tars_kv260.dtbo`、`shell.json`（xmutil 格式） |
| `firmware/tars_kv260.xsa` | 硬件平台（Vitis/PetaLinux 用，可选） |
| `sw/libtmdrv.so` (+ `tmdrv.c`) | 用户态驱动：/dev/mem 寄存器、hugepage 缓冲区与物理页号、cache 维护 |
| `sw/tars_board.py` | 设备驱动 + Qwen 运行时 |
| `sw/qwen_chat.py` | 串口对话 demo |
| `sw/selftest.py` | 上板自检（硬件通路 + 模型逐比特比对） |
| `data/` | `image.bin`（887 MiB：权重+状态）、`embed_bf16.bin`、`queues.bin`、`manifest.json`、`tokenizer.json`、`expected.json`、`selftest/` |
| `wheels/` | aarch64 的 numpy / tokenizers 离线安装包（板上不需要联网） |
| `setup.sh` / `run.sh` | 一次性安装 / 启动 |

## 前提

- KV260 已烧好官方 **Kria Ubuntu 22.04** 镜像并能启动（默认用户 `ubuntu`）。
- Windows 通过 micro-USB 连接板子：设备管理器里会出现 4 个 COM 口，Linux 控制台是**第二个**
  （编号次小的那个之后一个，常见为 COMx+1），PuTTY / MobaXterm 设置 **115200 8N1**。
- 把本目录拷到板上（网线 `scp -r kv260 ubuntu@<板子IP>:~/tars`，或 U 盘），约 1.5 GB。

## 步骤

```bash
cd ~/tars                 # 本目录
sudo ./setup.sh           # 一次即可：解压 Python 包到 pylib/、安装固件到 /lib/firmware/xilinx/tars_kv260、
                          #          申请 hugepage、校验数据文件
sudo ./run.sh selftest    # 自检（约 2-3 分钟，见下）
sudo ./run.sh             # 串口对话
```

`run.sh` 每次都会 `xmutil unloadapp && xmutil loadapp tars_kv260` 重新加载比特流（失败时退回
`fpgautil`）。**不要在没加载比特流时访问 0xA0000000**，否则总线会挂死。

### 自检 `sudo ./run.sh selftest`

1. **硬件通路**：24 个随机程序（RTL 回归用的同一个生成器，其中 10 个带注入的描述符错误），
   把核心写出的每个字节和结束状态与 golden model 比对。
2. **模型**：Qwen3.5-0.8B 按对话程序的方式运行两个提示词（"The capital of France is" 和一个对话：
   提示词分块 prefill，之后逐 token），**每次输出的 logits（248320 个 fp32）的 SHA-256 必须与 PC 上
   golden model 完全一致**；`sudo ./run.sh selftest --per-token` 则每个位置都逐 token 运行并比对。
   期望输出：`" Paris.\nThe capital of France is"`。

最后打印 `ALL PASS`。`sudo ./run.sh quick` 只跑每个提示词的前 6 步；`sudo ./run.sh hw` 只跑第 1 部分。

### 对话 `sudo ./run.sh`

```
you> 介绍一下长城
tars> ……（逐 token 输出）
  [N prompt tokens .. s, M new tokens .. s (.. tok/s), pos/256 context]
```

`/reset` 开始新对话，`/quit` 退出；上下文 256 token，满了自动开新对话。`sudo ./run.sh chat --raw`
不套聊天模板（纯续写）；`--max-new N` 限制每轮生成长度。

## 原理（板上软件做什么）

- 核心的地址空间是 1 GB 的平坦空间，被切成 512 个 2 MB 页；软件申请 512 个 Linux hugepage，
  从 `/proc/self/pagemap` 取得各自的物理地址写入核心里的页表（寄存器 0x2000+4i），所以不需要
  物理连续内存，也不需要内核模块。
- HP0 端口不与 CPU cache 一致：写给核心的数据（权重、输入、描述符队列）先 `DC CVAC` 写回，
  读核心结果（logits）前 `DC CIVAC` 失效。
- 每个 token：写入该 token 的 embedding（bf16→fp32，精确）和该位置的描述符队列（离线生成，
  约 2730 条/步），复位并启动核心，轮询 STATUS（halted 且 AXI 写全部应答），读 logits，argmax。

## 寄存器（0xA000_0000）

| 偏移 | 名称 | 说明 |
|---|---|---|
| 0x000 | ID | 0x54415253 ("TARS") |
| 0x008 | CTRL | bit0 start（写 1 产生脉冲），bit1 核心复位（电平） |
| 0x00C | STATUS | bit0 halted, bit1 done, [7:4] err, bit8 总线空闲, bit9/10 AXI 读/写错误 |
| 0x010 | OP_IDX | 停止时的描述符序号 |
| 0x014/018/01C | Q_ADDR_LO/HI, Q_LEN | 队列地址与长度 |
| 0x020/024 | CYCLES | 核心运行周期（先读 LO） |
| 0x028/02C | RD/WR_BEATS | AXI 读写拍数 |
| 0x034 | PARAMS | R, C, VG, 页大小 |
| 0x2000+4i | 页表 | 第 i 页（2 MB）的物理页号 = 物理地址 >> 21 |

## 排错

| 现象 | 处理 |
|---|---|
| `only N hugepages of 528` | 内存碎片化：重启后先执行 `sudo ./run.sh`（会先申请 hugepage） |
| `no TARS-mini at 0xA0000000` | 比特流未加载：`sudo xmutil listapps` 看 tars_kv260 是否 active |
| `xmutil loadapp` 失败 | 先 `sudo xmutil unloadapp`；确认 `/lib/firmware/xilinx/tars_kv260/` 有 3 个文件 |
| 自检第 1 部分失败 | 硬件/总线问题，记下首个出错字节地址 |
| 自检第 2 部分 MISMATCH | 记下第一个不一致的 pos，以及 `expected argmax` |

## 重新生成（PC 上）

```bash
cd tars_mini_fpga && source env.sh
vivado -mode batch -source fpga/kv260/build.tcl    # 比特流 (build/kv260/)
fpga/kv260/package.sh                              # -> deploy/kv260/firmware/
python3 tools/export_kv260.py --ckpt ~/models/Qwen3.5-0.8B   # -> deploy/kv260/data/
```
