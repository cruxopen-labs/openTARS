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
- **FDOT 优化（2026-11-07，比特流沿用 VERSION 0x00030100）：** 折叠与下一个输出重叠、Q 留在片上、NN 每 4 个 k 读一次 A；
  结果逐比特不变。WNS +0.187 ns，LUT 100,316 (85.65%)、DSP 879、BRAM 120。实测 decode：Qwen3-0.6B t=250 快 1.40×，
  Qwen3.5-0.8B 快 1.05–1.10×。上一版比特流在板上 `firmware_v32/`。
- **实板（v3.2，2026-11-03）：`run.sh hw` 24/24、`run.sh selftest` ALL PASS，每一步 logits 与 golden model 逐比特一致；
  decode 约 15.7 M 周期/token ≈ 0.16 s/token（v3.0：0.21）；20 token 的对话提示分块 prefill 2.29 s ≈ 0.115 s/token；
  对话示例全程 4.5 s（v3.0：4.9 s）。** v3.0 的固件和数据在 `firmware_v30/`、`data_v30/`（换回：把两个目录改名回 `firmware`、`data`，
  `SHA256SUMS.data.v30` 改回 `SHA256SUMS.data`，再 `sudo ./setup.sh`）。
- v3.0（实板验证过）：两个读口，每拍 64 字节；`run.sh selftest` ALL PASS，每一步 logits 与 golden model 逐比特一致；
  decode 约 21 M 周期/token ≈ 0.21 s/token；提示词按 4 个 token 分块 prefill ≈ 0.12 s/token
  （v1.0：2.6 s/token；v1.1：0.90；v1.3：0.73；v1.4：0.60；v2.0：0.36）。
- **v3.3（2026-11-07，VERSION 0x00030300，当前固件）：** FDOT 每周期 16 次乘加（4 组 lane 轮流累加，NT 4 个 key 一组、NN 64 列一组，
  128 字节读），结果逐比特不变。WNS +0.013 ns，LUT 104,879 (89.55%)、DSP 903、BRAM 118。实测 decode：Qwen3-0.6B t=16 6.71 M、
  t=250 9.51 M 周期；Qwen3.5-0.8B 13.2–13.6 M 周期。上一版在板上 `firmware_v32b/`。
- **MTP 投机解码（可选，默认关闭）：** 用 `tools/export_kv260.py --mtp --out deploy/kv260/data_mtp` 导出后，
  `sudo ./run.sh chat --data data_mtp --spec`；输出与普通解码逐比特相同。实测平均 0.97×（命中率约 77% 才不亏，
  最好的 JSON 任务 1.13×），`sudo ./run.sh bench --data data_mtp` 可复现对比。
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
| `data_mtp/`、`data_qwen3/`、`data_gemma3/` | 其他模型的导出，结构同 `data/`（用 `run.sh <模型>` 选择） |
| `data_fp16/`、`data_fp32_v34/`、`firmware_v34/` | FP16 状态实验（需要 v3.4 固件，`run.sh` 不加载，仅存档） |
| `firmware_v33_profile/` | 带分算子计数器的 v3.3 实验比特流（仅存档） |
| `results/` | 板上测量记录：`gemma3/`、`fp16_v34/`、`profile_v33/` |
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

### 选择模型

`sudo ./run.sh [模型] [命令] [参数]`，模型省略时为 Qwen3.5-0.8B，命令省略时为 `chat`：

| 模型参数 | 数据目录 | 说明 |
|---|---|---|
| `qwen3.5`（默认） | `data/` | Qwen3.5-0.8B |
| `qwen3.5-mtp` | `data_mtp/` | Qwen3.5-0.8B + MTP 头（`chat --spec`、`bench`） |
| `qwen3` | `data_qwen3/` | Qwen3-0.6B |
| `gemma3` | `data_gemma3/` | Gemma 3 1B（gemma-3-1b-it） |

例如 `sudo ./run.sh gemma3`（Gemma 对话）、`sudo ./run.sh qwen3 selftest`、`sudo ./run.sh gemma3 profile --out p.json`。
四个模型共用同一个固件（`firmware/`，v3.3）。`sudo ./run.sh --help` 列出全部参数。

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

## Gemma 3 1B（可选，Qwen 仍为默认）

固件不变（v3.3）。导出到 `data_gemma3/`，和其他模型放在同一个上板目录里，`run.sh` 第一个参数选模型：

```bash
python3 tools/export_kv260.py --ckpt ~/models/gemma-3-1b-it --T 256 --out deploy/kv260/data_gemma3
sudo ./run.sh gemma3 selftest
sudo ./run.sh gemma3 chat
```

实板：自检 ALL PASS（逐比特一致），decode 约 94–103 ms/token，对话约 9.5–10 token/s；上下文 256 token
（1 GB 内存余约 16 MiB）。映射方式、精度和限制见 [`Gemma 3 1B 报告`](../../docs/gemma3_1b/README.md)。

## 分算子性能采集

### FP16 状态（v3.4）

可选（默认关闭）：`--state-dtype fp16` 将 Qwen3.5 的 DeltaNet 状态 `S` 以 packed IEEE FP16
存储，衰减和更新写回时舍入；读回后乘法、累加、归一化及 SFU 均保持 FP32。权重、KV cache、
卷积历史和临时向量不变。导出工具和 `QwenLowering` 的默认值都是 `fp32`：
在 KV260 上实测 FP16 状态比 FP32 慢约 3%，且约 10% 位置的 top-1 改变。

```bash
python3 tools/export_kv260.py --ckpt ~/models/Qwen3.5-0.8B \
    --state-dtype fp16 --out build/data_fp16
# 同 checkpoint 的旧精度基线，独立目录，不覆盖原数据：
python3 tools/export_kv260.py --ckpt ~/models/Qwen3.5-0.8B \
    --state-dtype fp32 --out build/data_fp32
python3 tools/check_state_precision.py --fp32 build/data_fp32 --fp16 build/data_fp16 \
    --tokens 128 --out build/state_precision.json
```

CPU golden 比较较慢。两份数据上传后，也可在板上运行：

```bash
sudo ./run.sh precision --fp32 data_fp32 --fp16 data_fp16 \
    --tokens 256 --out precision.json
```

脚本先完成全部 FP32 基线采样，再加载 FP16，
不会把同一 FPGA 缓冲误当成两台独立设备；临时 logits 文件会自动清理。

必须重新构建并安装 v3.4 固件及重新导出数据、golden logits；仅替换 Python 无效。
manifest 标记最低版本，旧固件会被运行时提前拒绝。Qwen3 没有 DeltaNet 状态，
不受此精度选项影响。精度比较是对当前 W8A8/FP32 状态基线的增量检查，
不是通用模型质量保证；容量减半也不等于执行时间减半。
已观察到部分 greedy 输出改变，不能视为无损转换。实现范围、验证进度和实测报告见
[`FP16 状态报告`](../../docs/fp16_state_v34/README.md)。

### 采集命令

`sudo ./run.sh profile --data data --positions 16,64,128,192,250 --out profile.json`
保持完整队列运行，输出总读写流量和等待计数；旧 v3.3 固件只支持这些总计数。
重新构建当前 RTL 后，`PROFILE_CAP`（0x05c）为 `0x50524601`，可额外采集
GEMV、ELT、FDOT NT/NN、RoPE、QUANT、Norm、Softmax、ACT 及取指/校验开销。
加 `--require-operators` 可拒绝旧固件。`--mode prefill --check chat` 采集默认分块 prefill。
等待计数相互重叠，不可相加；分类周期之和必须等于核心周期。
完整的构建、计数口径、验证限制和 FP16 收益判断见
[`docs/kv260_profiling.md`](../../docs/kv260_profiling.md)。

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
