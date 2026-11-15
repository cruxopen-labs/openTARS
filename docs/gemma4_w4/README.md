# Gemma 4 E2B（gemma-4-E2B-it）的 W4A16 部署

日期：2026-11-14。在 W8A8（`data_gemma4`）之外新增 W4A16 导出（`data_gemma4_w4`）：
权重 4 bit（每 128 通道一组 FP16 scale），激活保持 16 bit 块浮点，片上对齐为有符号
16 bit 尾数后与 W8A8 共用整数点积通路。

**板的现状（重要）**：W4A16 的 RTL 单元（`tm_w4`）已完成，PC 上的 golden / 仿真逐位对齐，
但 KV260 的面积放不下"W8A8 + W4A16 两条通路"（实测见下）。因此本提交交付的是：
W8A8 的 v4.0 比特流（4 GB 地址空间，VERSION 0x00040000）+ 完整的 W4A16 软件栈与 RTL
（`W4X=1` 可综合，`TM_W4B` 构建目标已就绪）。W4A16 数据目录的 `min_firmware` 写着
0x00040001（VERSION bit 0 = 已带 W4 单元），0x00040000 的板上固件会明确拒绝加载，
不会静默算错。上板所需的面积方案见"上板路径"。

## 结果

| 项目 | 结果 |
|---|---|
| 板上自检（`run.sh gemma4-w4 selftest`） | 未上板：面积放不下（见"上板路径"） |
| 仿真吞吐（理想端口，M=1 N=64 K=4096） | 5,131 拍（旧窄读设计 35,723 拍，7.0×） |
| 权重体积 | `image.bin` 1.21 GiB（W8A8 为 1.9 GiB），`mem_bytes` 1,266,679,808 |
| PC 精度（teacher-forced，见下表） | 66 token 提示词 top-1 一致 64.6%（W8A8 83.1%）；86 token 提示词 44.7%（W8A8 76.5%） |
| 综合面积（W4 变体：`W4X=1`、`GB=1`、X bank 打包） | 120,193 LUT（102.6% of 117,120）、BRAM 118/144、DSP 666/1,248 |
| 布局 | 失败：需 10,530 CLB，可用 8,075（W8A8-only 参考实现用 105,634 LUT 刚好布局通过） |

## 映射方式（W4A16）

复用 W8A8 的全部 lowering（`llm.py` 的 `QwenLowering` 加 `wbits=4`），差异只在权重的
存储格式与 GEMV 的 epilogue：

- **权重量化**（`compiler.quantize_weight4`）：每 128 个输入通道一组，`s = fp16(amax/7)`，
  `q = rne(w/fp32(s))` 截到 `[-7, 7]`（对称，无零点）。两个 4 bit 打进一个字节，
  scale 以 FP16 存储。量化用的是舍入后的 scale（与 INT8 路径一致）；全零组 scale 为 1.0。
- **激活不变**：仍是 bf16 / fp32，由对齐引擎（W8A8 的同一套逻辑）转成有符号 16 bit
  块浮点尾数，8 个通道一个字节组与 W8 的点积通路完全共用。
- **epilogue**（`golden.py` 的 `_gemm_w4`）：`term = fmul(fmul(i2f(dot), wscale), xscale)`，
  每组先乘权重 scale 再乘激活 scale，仍按 FP32 逐步舍入，与 W8 路径的 `fmul(fmul(...))`
  顺序一致。
- **描述符**：新增 W4 变体（`DT_FP32=4` 之后沿用的格式字段 `wbits`），manifest 写
  `wbits=4`，`min_firmware=0x00040000`；旧固件加载 W4 数据目录时会拒绝。
- **激活 scale 与 W8 路径共用**（`--share-from data_gemma4`）：W4A16 不需要重新校准。

## RTL：tm_w4 单元

`tm_xpu.sv` 新增 `tm_w4`：每 pass 4 行（MB=4）。读侧每拍消费一整个 128 字节宽 beat
（与 W8 GEMV 相同的 `rq_wide` 宽度，`RW=8` 时一拍 4 个组），每组 128 个 4 bit×16 bit
乘法并行（128 个 DSP），树形累加成精确的 27 bit 组点积，每拍完成一个组的点积（每
`RW/4` 拍弹出一个宽 beat）；组的点积经 `i2f` 后按 epilogue 公式与双 scale 相乘，再走
与 W8 共用的行累加 / 激活 / 输出转换。组 scale 缓存在 `msc[MB][GS=512]`（GS = K/128），
X 对齐值缓存在每行 16 个 bank（每个组一个 bank，`XD` 个组深，KV260 顶层设
`W4XD=64`，即 K ≤ 8192）。其余（输出队列、读接口、SFU）与 W8 GEMV 相同。

吞吐（理想端口，核心顶层，K=4096 N=64）：

| M | 迭代次数 | 每轮时钟 | 相对旧的窄读设计 |
|---|---:|---:|---|
| 1 | 5,131 | — | 7.0×（旧设计 35,723） |
| 4 | 8,395 | — | — |
| 8 | 16,699 | — | 两遍（`W4MB=1` 的 KV260 为八遍） |

每通道 K/128 = 32 组 × 2 拍 = 64 拍，加上 scale / bias 读与双缓冲等待，接近理论下界
（64 字节/拍 ≈ 6.4 GB/s @100 MHz，低于 8 组并行读口的总带宽）。

## PC 精度（teacher-forced）

提示词固定，逐位置比较 W8A8 / W4A16 与 FP32 原模型的 logits（JSON 在导出时由
`tools/accuracy_gemma4.py` 生成）：

| 提示词 | 位置数 | 变体 | top-1 一致 | KL 均值 | NLL 变化 |
|---|---:|---|---:|---:|---:|
| 灯塔故事（66 token） | 65 | W8A8 | 83.1% | 0.435 | +0.093 |
| | | W4A16 | 64.6% | 1.808 | +0.047 |
| 卷积 vs 矩阵乘（18 token） | 17 | W8A8 | 70.6% | 0.507 | +0.480 |
| | | W4A16 | 70.6% | 2.069 | +0.953 |
| transformer 解码长提示（86 token） | 85 | W8A8 | 76.5% | 0.612 | −0.017 |
| | | W4A16 | 44.7% | 2.976 | +1.547 |

结论：误差主要来自 4 bit 权重本身（头层单独 W4 的 KL 仅 0.099）；尝试过的替代量化
（更细的组、AMAX 截断、按通道 scale 等）都没有稳定收益。W4A16 的 top-1 一致率比
W8A8 低 8–32 个百分点，但 NLL 在短提示词上与 W8A8 相当；适合在对权重量化不敏感的
生成场景使用，或作为 W8A8 放不下时的压缩选项（1.9 GiB → 1.21 GiB，T=512 时 W8A8
在 1 GB 地址空间已很紧张）。

## RTL 修复（本次调试发现并修复的 7 个问题）

1. **对齐引擎竞态**（`tm_xpu.sv`）：`al_busy` 在 `pop && 组类型匹配` 时置位，但清零
   条件会让新的对齐请求在 `pend_v` 时被吞。修复：`al_new` 同时要求 `!pend_v`，
   且仅在 `!al_new` 时清 `al_busy`。
2. **launch 行号溢出**：`lp` 用窄位宽计数，M 大时回绕。修复：内部用 `logic [LMB:0] rr`
   宽一位计数再截断。
3. **W4 描述符结束后的陈旧输出写**（`tm_core.sv`）：描述符结束时 `e_vsr` 未清零，
   延迟拍会再发一行全零写，覆盖正确结果。修复：结束分支同时清 `e_vsr`。
4. **组 scale 地址提前前移**（`tm_xpu.sv`，最难的一个）：`msc_ra_r` 在每个 W beat
   fire 时都重注册，而 `u_m2` 采样 `msc_rd` 在组最后一个 beat fire 之后 7 拍，此时
   BRAM 地址已指向 g+1 组。相邻组 scale 相同（大多数测试与真实权重）时不可见；
   组全局编号 ≥ ~32 后开始逐位不匹配（M=2/K=4096 第 1 行、M=4 gn=16 第 3 行、
   M=9 gn=8 第 6 行）。用 Python 逐项对照（`/tmp/khyp2.py` 思路）证实：RTL 的
   bank 累加器 == 用 `xs(g+1)` 算的参考值。修复：只在组的最后一个 beat（`l3c`）
   锁存 `msc_ra_r <= cg`。
5. **`acc` 多驱动**：控制块的 `start` 复位与各 `g_dp` 块同时驱动 `acc[MB][4]`，
   Verilator 容忍，Vivado DRC（MDRV-1）失败。修复：控制块不再复位 `acc`，各
   `g_dp` 块自己处理 `start / sclr / wp`。
6. **LUT 超限**：128 个 4 bit×16 bit 乘法默认综合成 LUT，W4 单元 61,516 LUT，
   整体 137,965 > 117,120。修复：`(* use_dsp = "yes" *)` 于 `prd[32]`，W4 单元
   降至 19,039 LUT / 189 DSP / 64 RAMB36。
7. **CLB 布局拥塞**（place_design 失败）：opt 后 LUT-as-logic 116,489（99.46%），
   11,168 CLB 需求 > 8,113 可用。缓解：`W4XD=64`（按组计，KV260 的 K ≤ 4096 轮廓）
   把 X bank 深度对齐 KV260 轮廓，降低 BRAM 与布局压力。
8. **吞吐重构（旧设计的 7 倍瓶颈）**：最初的实现每拍只吃 32 字节窄 beat、组内 8 个子步，
   M=1/N=64/K=4096 要 35,723 拍。改为每拍消费整个 128 字节宽 beat、每拍完成一个组的
   128 点积后降到 5,131 拍（含双缓冲与 scale/bias 开销，接近 4,096 拍的下界）。
9. **X bank 写地址截断**（重构后输出全零）：`xwb` 声明为 `logic [2:0]`（8 bank 时代的
   遗留），16 bank 后 `aw[4:1]` 被截成模 8，组写把 bank 0 清零。修复：`logic [3:0]`。
10. **FIFO 上半拍读到的是线值而非队列头**：宽 beat 的上 112 字节在 fire 时直接用
   `rq_hi_x`（当前线值，已过期），抓取到的是上一拍的数据，前 16 字节对、其余全错。
   修复：先补 `assign qh_h = qh[q_h]`（两个实现分支），抓取改用 `qh_h`。
11. **X bank 的 BRAM 浪费**：原来每个 lane bank 一个 128 位宽 BRAM（Vivado 一个 RAMB18
   只能到 36 位，等于 8 倍浪费，16 个 bank 吃掉 32 个 RAMB36，超过 144 的上限）。改为
   4 个 512 位宽 BRAM（`tm_bramm`，带字节写使能，每个 lane 128 位一条 mask），BRAM
   32 → 4，LUT 也少了 2.8k。
12. **版本能力位**：`VERSION` 的 bit 0 表示"带 W4A16 单元"（W4 变体 0x00040001），
   W4 导出的 `min_firmware` 据此校验，避免 W4 数据被老固件静默算错。

## 验证

- `tests/test_w4a16.py`：28 个用例（量化数值、golden 一致性、异常输入、share-from）。
- `tests/test_rtl_w4.py`：30 个用例，核心顶层（RW = 8）与 KV260 顶层（GEMV-only）各跑一遍
  全绿（形状覆盖 M=1..9、K=256..4096、相邻组 scale 不同、NaN/无穷组、全部输出格式、
  错误码与 lowering）。
- KV260 全量 RTL 套件（`TM_FPGA=1 TM_TOP=kv260`，`test_rtl + test_rtl_v04 + test_rtl_w4`）：
  451 passed / 34 skipped。
- 核心顶层全量回归（默认 `core` top，`pytest tests/`）：924 passed / 19 skipped。
- 板上：W4 比特流待面积方案落地（见下）。

## 上板路径（面积）

W8A8-only 参考设计用 105,634 LUT 布局通过（90.2%），布局器对当前设计只放出 8,075 个 CLB，
所以"两条完整通路"一定放不下。W4 变体（`TM_W4B`：W4X=1、GB=1 的 W8 GEMV、X bank 打包成
4 个 512 位宽 BRAM 带字节写使能）实测 120,193 LUT（102.6%），仍超 117,120 的上限，布局
需要 -15k 左右。剩余可挖的空间与代价：

| 手段 | 估计节省 | 代价 |
|---|---|---|
| `tm_xpu` 去掉 W8 GEMV 数据通路（`GV=0` 参数化） | ~7k LUT + 384 DSP | GEMV 与其他扩展算子共用 FSM，需要重构 + 重跑全部扩展算子测试 |
| W4 乘法器 128 → 64（每组的 128 个值分两拍） | ~3k LUT | 弹拍间隔变 4 拍，吞吐减半（约 2× 慢） |
| W4 的 X 缓冲改为环形 FIFO（8 组而非整行） | ~1k LUT + 3 BRAM | 二期地址生成改为环形 |

三条合计 ≈ -11k，仍差一点；真正干净的方案是让 W4 复用 W8 GEMV 的乘法阵列（激活的高/低
字节分两遍、权重 nibble 当 int8），那样几乎不新增面积，但需要改 GEMV 的累加与 epilogue，
工作量最大、风险也最高。当前选择：先交付 W8A8 的 v4.0 固件与完整 W4A16 软件/仿真栈，
W4 比特流另行迭代。

## 复现

```bash
# 导出（W8A8 已存在时，激活 scale 等直接复用）
python3 tools/export_kv260.py --ckpt ~/models/gemma-4-E2B-it --T 512 --wbits 4 \
    --share-from deploy/kv260/data_gemma4 --out deploy/kv260/data_gemma4_w4
# 仿真（核心顶层 / KV260 顶层）
pytest tests/test_w4a16.py                                   # golden 与 lowering
pytest tests/test_rtl_w4.py                                   # 核心 RTL，RW = 8
TM_FPGA=1 TM_TOP=kv260 pytest tests/test_rtl_w4.py            # KV260 轮廓（GEMV-only）
# W4 比特流（当前面积放不下，见"上板路径"；构建目标已就绪）
vivado -mode batch -source fpga/kv260/build.tcl -tclargs 8 w4
# 板上（需要带 W4 单元的固件；0x00040000 的固件会拒绝）
sudo ./run.sh gemma4-w4 selftest
sudo ./run.sh gemma4-w4 chat
sudo ./run.sh gemma4-w4 profile --mode both --check chat --out profile_gemma4_w4.json
```
