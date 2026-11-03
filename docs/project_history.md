# TARS-mini：从 ASIC 参考设计到 KV260 上跑 Qwen3.5-0.8B 的全过程

时间：2026-10-31 至 2026-11-02。结论先行：

- **结果**：TARS-mini 在 KV260（XCK26）上以 100 MHz 运行 Qwen3.5-0.8B 真实权重，每一步 logits 与 golden 模型逐比特一致。decode 从最初的 2.6 s/token 优化到 **0.36 s/token**（快 7.2 倍）；20 个 token 提示词的 prefill 从 2.6 s/token 降到 **0.15 s/token**（快 18 倍）；31 个位置的示例对话从约 80 s 降到 **7.1 s**。
- **方法**：契约和 golden 模型是唯一真值。所有改动只允许改变周期数，不允许改变任何一个结果比特；每一版都先在仿真里逐比特对比，再上板复验。
- **当前瓶颈**：LUT 已用 89.4%，后续加功能都要先腾出面积。

---

## 1. 起点：TARS-mini（ASIC 参考设计）

TARS-mini 是 TARS NPU 的第一个端到端切片：单核，接收 512 位描述符（V1.3.1 格式）。它在本项目开始前已经完成到 v0.4，本项目没有改动它的 ASIC 行为。

| 部分 | 内容 |
|---|---|
| 契约 | v0.4。`contract.py` 和 `golden.py` 是它的可执行形式，golden 模型里没有任何微架构参数（R、C、NB、OT…） |
| RTL | R×C INT8 输出驻留脉动阵列（双累加 bank）、epilogue、行引擎（vpu_lane + fold）、扩展单元 tm_xpu（ELT/ROPE/FDOT/QUANT/GEMV）、fp32 流水单元 |
| 算子 | INT8 GEMM/Conv（tensor、channel、W8A8 ROWCOL 量化）；ReLU/GELU/SiLU/sigmoid/softplus/exp；Softmax（含因果）、LayerNorm、RMSNorm、L2NORM、QUANT；ELT、ROPE、FDOT |
| 验证体系 | 748 个测试；随机队列 fuzz（5 种阵列形状、随机反压和延迟、覆盖断言）；141 个变异体的变异测试；fp 单元约 60 万向量 |
| 时序 | ASAP7 + Yosys/ABC + 自写 `sta.py`，每级 ≤650 ps（1 GHz），最差 644 ps |
| 访存模型 | DRAMsim3 模拟 LPDDR4X-4266 |
| Qwen | 全模型 24 层 + head，一个 decode 步 2,730 个描述符；理想端口 57.3 M 周期，LPDDR4X 下 76.5 M 周期（1 GHz 时约 76 ms/token） |

### 1.1 本项目中的复现与验证

1. **环境**：AlmaLinux 8，没有 sudo。装了 Miniforge 环境 `tars`（Verilator 5.052、Yosys 0.69、torch CPU、transformers）、sv2v（通过 conda 的 glibc 2.34 loader 运行）、ASAP7 库和 DRAMsim3。
   - 踩过的坑：torch 加载了系统的旧 libstdc++，用 conda 的 activate.d 钩子修好；CMake 4 不接受 DRAMsim3，加 `CMAKE_POLICY_VERSION_MINIMUM=3.5`；sv2v 需要 glibc ≥ 2.34。
2. **复现结果**：
   - 333 个非 RTL 测试和 RTL fp 测试通过。
   - golden 全模型生成的 token `[14184,156431,220662,75312]` 与原始记录一致。
   - 与 transformers 参考逐层对比的结果与原记录一致。
   - 时序校准一致：32 位加法器 170 ps，乘法器 580 ps（文档为 167/578）。
   - **没有完成**：两个 RTL 全模型 decode 仿真（理想端口和 DRAM）各跑到 10 M 周期时，按用户要求停掉。单线程约 66 分钟跑 1000 万周期，跑完要 6–8 小时。
3. **真实权重（原 README 列出的下一步）**：下载 Qwen3.5-0.8B（bf16），`qwen_ref.load_weights` 按名字和形状逐个核对后载入。
   - golden 输出："The capital of France is" → " Paris.\nThe capital of France is Paris..."
   - 与 fp32 transformers 对比：argmax 17/17 一致，logits cos 最小 0.9932，平均 0.9972。与同样做 W8A8 的 transformers 对比，cos 最小 0.9945。
   - 逐层相对误差中位数 1.8e-7。LA 层 11、17、20、22 约为 2e-3，原因推测是 INT8 量化取整翻了一档，不影响 argmax。
4. **Verilator 多线程**：`TM_THREADS`。单独运行时默认 min(8, 核数/2) 个线程，在 pytest-xdist 下每个 worker 1 个线程。

---

## 2. FPGA 移植：把 604k LUT 压进 117k

目标器件 XCK26：117,120 LUT、234,240 FF、1,248 DSP、144 BRAM36、64 URAM。全部 FPGA 改动都放在 `` `ifdef TM_FPGA `` 下，ASIC 构建的比特和周期都不变。移植在副本 `tars_mini_fpga` 里进行，原 `tars_mini` 只加入了 SFU 合并和共享 SFU（同样在 ifdef 下）。

### 2.1 第一次全核综合：放不下

最小可用配置是 R=8、C=2、VG=1、NB=4、OT=64、HB=256、OQ=32、LG=4、VQMIN=32，乘法器改用 DSP。综合结果是 **603,792 LUT（KV260 的 5.2 倍）**、270k FF、1,452 DSP。

根本原因是：为 1 GHz 设计的向量通路每周期处理 4 个元素，**每个元素槽都带一套完整的 fp32 流水线**。全核共 10 个完整激活单元（epilogue ×C、行引擎 ×4、xpu ×4），约 115 个 fadd；hbuf、abuf、wbuf 因为要多端口异步按字节读，被综合成了触发器。

结论是只换成 BRAM 和 DSP 远远不够，必须改结构。

### 2.2 面积压缩各步

| 步骤 | 做法 | LUT |
|---|---|---|
| 原始（乘法器改 DSP） | 尾数乘法 `mul24_cs` 在 FPGA 下改为 `ma*mb`；结果比特相同，因为下一级只用 ps+pc | 603,792 |
| SFU 内部合并 | GELU/SiLU/sigmoid/exp/softplus 共用一个 exp、一个 recip、一组 fmul/fadd，每个元素沿流水线携带自己的函数码。单个 fpu_act 从 62k 降到 18.5k LUT | — |
| 全核共享 SFU | 10 个激活单元合成 1 个。`tm_sfu_port` 把各 lane 错开一拍送进 SFU，代价是长激活每 W 个周期才能发一步，延迟增加 W−1 | 302,363 |
| 去掉重复单元 + fadd 对齐移位放进 DSP | FPGA 下关闭行引擎的行重叠级（它自带一套累加器和 fold）；softmax 的 exp 改走 SFU；对齐移位改成乘以 2^(26−d) | 263,055 |
| **hbuf → LUTRAM**（决定性一步） | hbuf 按行拆成多个 bank，用单写口 LUTRAM（`tm_lutram`）；`vq_d`、`oq_*`、xpu 队列也改用 LUTRAM | 119,371 |
| 跨单元共享 | xpu 借用行引擎空闲的 16 个累加器、fold 加法树和 recip；`u_inv`（1/N）复用 fold 的 recip | ≈98k（核心） |
| 加上 AXI 桥和顶层，实现 | — | **100,434（85.8%）** |

### 2.3 v1.0：第一版比特流

- **板级结构**：Zynq MPSoC（kv260_som preset），pl_clk0 = 100 MHz。HPM0_FPD 经 SmartConnect 接 AXI-Lite 寄存器；核心 m_axi 经 SmartConnect 接 S_AXI_HP0_FPD（128 位）。
- **AXI 桥 `tm_kv260_top`**：
  - 每次传输单拍 16 B；有未确认的写时所有读暂停，保证读写顺序。
  - 2 MB 页表（512 项）把核心的 1 GB 地址空间映射到 Linux hugepage。
  - 寄存器：ID、CTRL、STATUS、OP_IDX、队列地址和长度、周期和拍数计数、PARAMS、MEM_MB，以及 0x2000 起的页表。
- **实现结果**：WNS +0.186 ns，LUT 100,434，DSP 380，BRAM 2。
- **仿真验证**：
  - 经过 KV260 顶层的回归 399/399：页面打乱、AXI 随机停顿和随机延迟。
  - 全模型真实权重：把板上软件原样接到 RTL 后端运行，连续 3 个 token 的全部 248,320 个 logits 逐比特一致。仿真里每 token 84.8 M 周期。
  - 用实际进入比特流的结构化 fp 单元跑 2 层真实维度模型，逐比特一致。

### 2.4 移植阶段遇到的问题和解决

| 问题 | 原因 | 解决 |
|---|---|---|
| 回归里出现 3 个失败 | 6 个 worker 同时用 `-j 0` 编译 Verilator，内存耗尽，构建被 OOM 杀掉 | 每个 worker 的编译并发改为核数/worker 数；`-j` 不再计入构建 hash |
| 关闭行重叠后 softmax 死锁 | `r_gb0` 和 softmax 的 `S_W_RUN` 分派仍按有重叠的方式走 | 两处都加上 `SFU==0` 门控 |
| 不用 SFU 的行引擎步骤冲掉了正在送数的 lane | 端口在每一步都会重新装载 | `v_sv <= v_con && v_sneed` |
| 行引擎死锁 | `` `ifndef `` 吞掉了一个 `if` 的语句体 | 把整个 `if` 包进去 |
| FIFO 的 head+1、head+2 读地址不回绕 | 先转成 32 位再相加，加法被扩宽 | 先截成 LQ 位再扩宽 |
| conv 补零和读返回在同一周期写 LUTRAM | LUTRAM 只有一个写口 | FPGA 下补零推迟一拍，只改变周期数 |
| 新测试未必真的走到 SFU 新路径 | — | 在三个端口各植入一个 bug（晚一拍、去掉寄存器、结果交换），分别有 4、2、23 个测试失败。SFU 内部也植入过一个错，混合函数测试抓到 6,133/50,029 |
| testbench 问题 | `$fread` 不支持动态数组；AXI-Lite 任务和时钟沿竞争；64 MB 镜像时页表溢出 | 改用静态数组；在下降沿驱动和采样；页大小选择让页表 ≤2048 项 |

---

## 3. Qwen 部署方案（软硬件分工）

```
PC（离线一次）                          KV260（全部在板上运行）
──────────────                          ──────────────────────────────
safetensors → W8A8 量化 → 降级成描述符     setup.sh：装固件、装离线 wheels
 image.bin（887 MiB，权重+状态+scratch）   run.sh：检查无人占用 → 释放 hugepage → xmutil 加载
 queues.bin（每个位置一个预编译队列）          → 确认 fpga0 "operating" 且 tars_mini 设备存在
 queues_c8/c4.bin（分块 prefill 队列）     tars_board.py：tokenizer → bf16 embedding（host 侧）
 embed_bf16.bin、tokenizer、expected.json     → 写入 hugepage → 启动队列 → 读 logits → argmax
 selftest/（24 个随机程序）               libtmdrv.so：/dev/mem 访问寄存器，pagemap 查物理页，
                                            DC CVAC/CIVAC 维护 cache（HP0 不一致）
```

- **硬件**：TARS-mini 核心执行全部 2,730 个描述符/token，包括 24 层的 LA/FA、norm 和 tied head。host 只做 embedding 查表和 argmax。
- **软件**：没有内核驱动。1 GB 模型放在 512 个 2 MB hugepage 中，物理页号写进核心页表。每个位置的描述符队列在 PC 上预编译好，板上只需改写 embedding 再启动队列。
- **入口**：
  - `qwen_chat.py`：在串口或 SSH 上对话，默认分块 prefill，`--per-token` 切回逐 token。
  - `selftest.py`：第一部分跑 24 个硬件随机程序，其中 10 个故意注入错误码；第二部分对比每一步 logits 的哈希。
  - `TARS_BACKEND=golden`：在 PC 上用同一份软件跑 golden 后端，用于对照。
- **数据量**：每 token 读约 912 MB，其中权重为主，输出层（词表 248,320）约占 28%。

### 3.1 板卡 bring-up 的问题

| 问题 | 解决 |
|---|---|
| QSPI 里的旧 U-Boot 2020.01 无法启动 Ubuntu 22.04 | 用 recovery 工具升级到 K26 BootFW 1.02（2023.2）；第一次 CRC 校验失败，重试成功 |
| Etcher 写 SD 卡总是失败（"write ending unexpectedly"） | 廉价 USB 读卡器会损坏写入的数据。改为在 WSL 里用 usbipd 挂载读卡器，dd 写入后 cmp 校验 |
| 首次 `run.sh hw` 时板子挂死（cma alloc failed，然后 mmc 超时、断网） | 先分配 hugepage 占掉了 CMA，比特流加载静默失败，读 0xA0000000 把总线挂死。改为先加载比特流再分配 hugepage，并在访问 PL 前检查加载状态 |
| 我在用户对话进行中重载了 PL | run.sh 发现已有对话或自检进程时拒绝运行。每次上板操作前先执行 `pgrep -af "sw/(qwen_chat\|selftest)\.py"` |

**v1.0 上板结果**：hw 24/24，selftest ALL PASS，两个提示词共 43 步全部逐比特一致。速度 **2.6 s/token（261 M 周期）**，是仿真的 3 倍：真实 DDR 延迟高，而桥只发单拍 16 B 请求。

---

## 4. 提速迭代（全部上板，ALL PASS，逐比特一致）

| 版本 | 主要改动 | decode s/token | prefill s/token（20 个 token） | 示例对话 | 周期/token | LUT |
|---|---|---|---|---|---|---|
| v1.0 | 单拍 16 B 桥 | 2.6 | 2.6 | ~80 s | 261 M | 85.8% |
| v1.1 | 读突发；读只被重叠地址的写挡住；性能计数器 | 0.90 | 0.90（逐 token） | 28.5 s | 90 M | 86.1% |
| v1.2 | 突发延迟关闭（仅仿真，**放弃**） | — | — | — | — | — |
| v1.3 | 多流交错收集 + ROB；在途读 128；ELT 标量操作数不重复读；**批量 GEMV（MB=8）** | 0.73 | 0.32（分块 8/4） | 14.8 s | 73 M | 87.5% |
| v1.4 | 写突发合并 + 同拍按字节合并，16 个写突发在途 | 0.60 | 0.19 | 10.7 s | 60 M | — |
| **v2.0** | **32 B 读口**：256 位 AXI → SmartConnect → HP0 128 位 @200 MHz；GEMV 每拍 32 B；PE 改 DSP | **0.36** | **0.15** | **7.1 s** | 36 M | 89.4% |

### v1.1：读突发
- **改动**：读请求先进输入寄存器（消除 `mr_addr→mr_ready` 组合环），连续地址攒成最多 16 拍的突发，突发不跨 4 KB。8 项写队列加年龄掩码 `in_old`，只有真正和未完成写地址重叠的读才等待。新增 7 个计数器：突发数、冲突、AR 等待、写等待、读占用（可算出平均延迟）。
- **效果**：AXI 读事务数从 57 M 降到 18 M，平均每个突发 3.1 拍，平均读延迟 63 周期。
- **问题**：第一版有组合环（Verilator UNOPTFLAT），用输入寄存器解决。仿真的内存模型低估了真实 DDR 处理零碎请求的代价：仿真只快 18%，板上快了 2.9 倍。
- **B 方案，分块 prefill**：结果逐比特正确，但板上**更慢**（1.6–1.8 s/token）。M≥2 走 8×2 阵列 GEMM，受算力限制，激活值要为每 2 列输出重读一遍。当时默认关闭。

### v1.2 → v1.3：找准瓶颈再动手
- **v1.2**：只做"更长突发 + 更多在途"，仿真里几乎没有收益，放弃。
- **定位**：建立贴近板子的仿真内存模型（`+nout=16 +blat=50`，1.55 周期/拍，板上实测 1.58）。发现约 1/3 的时间花在 LA 状态更新：3 路数据交错读，每个突发只有 1 拍，被 HP 口在途突发数限制在 0.33 拍/周期。
- **桥**：NS=4 路交错流各自收集成突发，平均约 25 拍；BRAM ROB 按原顺序把数据还给核心。核心在途读 64 → 128（Little 定律：约 100 周期延迟 × 1 拍/周期）。ELT 的 bmode 2/3（每行或全局一个标量）不再每拍重读。
- **大阵列行不通**：PE 改 DSP 后，16×8 阵列仍要 133k LUT（114%），16×16 要 175k，因为 epilogue、行读出 mux 和 loader 表都随 R/C 增长。改做**批量 GEMV**：权重只流过一次，每拍 16 个权重同时乘 M≤8 个 token，乘法放 DSP。8 个 token 一块，每 token 比逐 token decode 快 3.3 倍。
- **问题**：块大小 16 超过 MB=8，会退回慢阵列，所以导出改为 8/4。`test_loader_reads_in_flight` 的 M 从 8 改成 16，因为 M=8 现在走 GEMV，测不到阵列 loader。
- **上板**：decode 只快 19%，仿真预期是 29%。计数器显示写通路成了新瓶颈：每 token 16 M 周期在等写，因为写是单拍、最多 8 个在途。

### v1.4：写突发
- **改动**：连续的写攒成最多 16 拍的突发；GEMV 逐元素写回的 4 B 写按字节使能合并成整拍；最多 16 个写突发在途。读写冒险按 16 B 精度检查。
- **效果**：写等待从 16 M 降到 0.15 M 周期，decode 快 18%，和事先预测的 15–20% 吻合。

### v2.0：32 B 读口
- **改动**：
  - 桥改成 32 B 拍，ROB 每个槽放一个 32 B 拍，两个 16 B 半拍共享一个槽。
  - 新增 pl_clk1 = 200 MHz 驱动 HP0；SmartConnect 用两个时钟做 256→128 位宽和时钟转换；设备树覆盖里加 `clocking1`。
  - GEMV 增加 WIDE 模式（K%32==0 且基址按 32 B 对齐），每拍 32 B，窄模式时把 16 B 放进对应半拍。
- **问题**：
  - 第一版反而更慢：MAXB=64 个 32 B 拍，正好用光 128 个在途信用。改成 MAXB=16 解决。
  - 第一次实现布局失败，LUT 94.2%，卡在 CLB packing。用了四招：阵列 PE 改 DSP；阵列 loader 在途读降到 32（GEMV 和行引擎的 VQMIN 保持 128）；NQ=256；GEMV 窄模式放置 mux 从 8 路合成 1 路；再加上 AreaOptimized_high 综合策略。最终 89.4%，WNS +0.086 ns，DSP 668，BRAM 40.5。
- **效果**：decode 快 40%，36.4 M 周期/token，平均读延迟约 108 周期。

### 每一版的验证流程（固定不变）
1. 修改 RTL，lint；
2. 跑两套 RTL 回归，各 416 项（核心 testbench，以及经 KV260 顶层、页面打乱、AXI 随机延迟）；
3. 用板级内存模型跑 2 层真实维度模型，确认逐比特一致并量周期；
4. Vivado 实现，打包 bit.bin、dtbo；
5. 上板：`run.sh hw` → `selftest --per-token` → `selftest`（分块），全部 ALL PASS 才替换默认固件。旧固件保留：`firmware_v1`、`_v11`、`_v14`。

---

## 5. 给未来迭代的启示

**方法**
1. **契约和 golden 是一切优化的底气。** golden 里没有微架构参数，所以共享 SFU、关闭行重叠、改桥、改读口宽度都只改变周期，可以大胆做，再用逐比特对比兜底。
2. **验证要证明自己确实测到了改动。** 每加一条新路径，就植入一个 bug 看测试能否抓到。SFU 内部、三个端口、loader 测试改 M 都靠这一步发现问题。只看"测试通过"不够。
3. **先测量瓶颈，再选方案。** v1.2 盲目加长突发没有收益；有了计数器和校准过的内存模型，v1.3 的多流收集、v1.4 的写突发都事先算准了收益（v1.4 预测 15–20%，实测 18%）。性能计数器应该从第一版比特流就带上。
4. **仿真内存模型必须对齐真机。** 理想端口给出 0.85 s/token，真机是 2.6 s。按板上 1.58 周期/拍校准之后，仿真预测才可信。下一个平台第一件事就是测真实的延迟和带宽，再校准模型。

**FPGA 架构**
5. **ASIC 的"每 lane 一套完整 fp 流水线"在 FPGA 上不可行。** 1 GHz 的设计用面积换吞吐，到 100 MHz FPGA 上就要反过来：共享 SFU，用 LUTRAM 和 BRAM 存储，跨单元借用空闲的加法器和 recip。
6. **存储结构决定面积。** hbuf 一项占了 125k LUT：多端口异步按字节读会被综合成触发器。新设计从一开始就按 FPGA 的单写口 LUTRAM 或 BRAM 加寄存读来写存储。
7. **LUT 才是墙，DSP 很宽裕（668/1248）。** 想提算力，优先考虑"权重只流一次、乘法放 DSP"的形状，例如批量 GEMV，而不是放大脉动阵列。下一步可用的 LUT 余量：合并 fold 和 SFU 里的两个 recip（约 3k）、xpu 收窄到 2 lane（约 5k）、loader 表瘦身。
8. **decode 受带宽限制。** 突发长度、在途数量、读写分离、读口宽度依次是收益最大的旋钮。接下来的方向：GEMV 也用 HP1 做第二个读口（双 HP 口）；权重降到 INT4，读的数据量减半（这需要新契约和新 golden，结果不再与当前逐比特一致）；把 LA 递推状态放到片上，省掉状态读写。

**工程和板上操作**
9. **机器资源要算账。** 31 GB 内存同时跑 Vivado、多个 Verilator 构建和长仿真，会被 OOM 或系统回收进程打断：每个 worker 的编译并发设为核数/worker 数；Vivado 和大回归错开跑。
10. **不要用会匹配到自己的 `pgrep -f`/`pkill -f` 模式**，改用 `ps aux | grep "[p]attern"` 或按 PID 操作。
11. **构建或回归运行期间不要改 RTL**，否则不同构建会读到混合版本的源码。
12. **上板前必须确认没有人正在用 PL。** 重载比特流会清空页表，正在运行的进程会访问错误的物理地址。run.sh 已经把这一点做成了硬检查。
13. **Linux 侧的顺序依赖**：先加载比特流，再分配 hugepage；访问 PL 前确认 fpga0 处于 operating 状态。这类问题仿真里覆盖不到，要写成脚本里的检查。

## 6. 未完成或未验证的项目
- 移植副本 `tars_mini_fpga` 的 **ASIC 构建回归没有跑完**：内存不足时被系统中止于 35%，2 个失败没有诊断。原 `tars_mini` 的 ASIC 构建不受影响。
- 本项目没有重新跑完 ASIC 构建的全模型 RTL decode 仿真（按要求停在 10 M 周期）。FPGA 构建的全模型已在 RTL 上做过逐比特验证。
- 移植后的 FPGA 改动没有重跑 `mutate_rtl.py` 全量变异测试，只对新路径做了定向植入 bug 检查。
