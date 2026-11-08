# Gemma 3 1B（gemma-3-1b-it）在 KV260 上的部署

日期：2026-11-08。固件不变：使用板上原有的 v3.3 比特流（VERSION 0x00030300，
SHA-256 `c40eb336…`），没有修改 RTL。Qwen 仍是默认模型；Gemma 放在单独的数据目录中。

## 结果

| 项目 | 结果 |
|---|---|
| 板上自检（`run.sh selftest`） | ALL PASS：硬件通路 24/24；两个提示词共 41 个位置，logits 与 PC golden model 逐比特一致（SHA-256） |
| Decode（核心时间，100 MHz） | t=16：93.9 ms；t=128：98.5 ms；t=250：103.4 ms（约 9.7–10.6 token/s），读带宽约 10.6–11.1 GB/s |
| Prefill | 17 个 token、分 4 个 token 一块：684 ms（约 40 ms/token） |
| 对话实测（含 Python 开销） | 9.4–10.2 token/s；多轮上下文正确（第二轮“its population”指东京） |
| 内存 | `image.bin` 1008 MiB（T=256），1 GB 地址空间余约 16 MiB；T≥384 放不下 |
| W8A8 与 FP32 原模型（teacher-forced） | 英文 106 token：top-1 一致 86.8%，平均 KL 0.060，PPL 17.69→18.53；中文 128 token：83.6%，0.162，46.42→48.91；对话 84 token：89.3%，0.190，25.38→26.07 |

板上输出示例（greedy）：

```
you> 用一句话介绍长城。
tars> 长城是中国古代一项伟大的军事防御工程，也是世界文化遗产，雄伟壮丽，绵延数千米。
```

## 映射方式

Gemma 3 1B 用到的算子在现有指令集中都已具备，只改了 lowering 和运行时：

- 26 层全部按全注意力执行：滑动窗口 512 不小于上下文 256，结果相同（lowering 断言 T ≤ 窗口）。
- 局部层 / 全局层各用一张 RoPE 表（base 1e4 / 1e6）。
- 每层 4 个 RMSNorm（权重为 1 + w）：注意力输出和 MLP 输出在残差相加前再做一次 norm；q_norm、k_norm 同 Qwen3。
  注意力缩放 `query_pre_attn_scalar^-0.5 = 1/16` 并入 q_norm 的 gamma（精确）。
- GeGLU（tanh 近似 GELU，`ACT_GELU`）。
- `down_proj` 的 K=6912 超过 GEMV 单元上限 4096：权重按整行做 per-channel 量化，存为两个 K=3456 的块，
  共用 scale，两个部分积以 FP32 相加。
- Embedding 表按原 bf16 存储；主机查表后乘 `fp32(sqrt(1152))`（manifest 的 `embed_scale`），与 transformers 的 FP32 计算一致。
  Head 使用未缩放的 tied embedding，词表 262144 分成 8 块，每块 32768 行。
- 对话：manifest 的 `bos_ids=[2]` 只在位置 0 加入；`turn_end="<end_of_turn>\n"` 关闭模型的上一轮；
  `eos_ids=[1, 106]`。运行时拼出的多轮 token 序列与官方 chat template 完全一致（已核对 83 个 token）。

## 验证

- `tests/test_v04.py`：Gemma 小配置（局部 + 全局层，K 分块的 down_proj，GEMV-only 规则）在 golden model 上
  greedy 解码，logits 与 transformers `Gemma3TextModel`（同样的 W8A8 线性层和 head）相差 ≤ 1e-5，token 相同；
  各层更新 ≤ 2e-5。变异检查：交换两张 RoPE 表或去掉注意力后的 norm，误差分别升到 6e-2 和 0.38。
- `tests/test_rtl_v04.py::test_gemma3_model_greedy_on_rtl`：KV260 顶层 RTL 与 golden model 写出的字节完全相同。
- 回归：主机测试 423 passed / 4 skipped；FPGA RTL（`TM_FPGA=1 TM_TOP=kv260`）419 passed / 34 skipped。
- 板上：自检、对话和性能日志见 `validation/`。

## 复现

```bash
# 下载（需先登录 Hugging Face 并接受 Gemma 许可）
hf download google/gemma-3-1b-it --local-dir ~/models/gemma-3-1b-it
python3 tools/export_kv260.py --ckpt ~/models/gemma-3-1b-it --T 256 --out build/data_gemma3
# 板上：导出拷到 ~/tars/data_gemma3，run.sh 第一个参数选模型
sudo ./run.sh gemma3 selftest
sudo ./run.sh gemma3 chat
sudo ./run.sh gemma3 profile --mode both --check chat --out profile_gemma3.json
```

板上位置：`~/tars/data_gemma3`，测量记录在 `~/tars/results/gemma3`。Qwen 仍是默认模型（`sudo ./run.sh`），固件未改动。

## 已知限制

- 上下文最多 256 token（1 GB 内存的限制）。
- W8A8 对 Gemma 的影响比 FP32 状态与 FP16 状态的差异大：困惑度升高 3–5%，约 11–16% 位置的 top-1 改变。
- 在多轮英文对话后用中文提问，模型（包括 FP32 原模型）会用英文回答，这是模型本身的行为。
- Gemma 的数据和缓存约 1.6 GB，会让内存碎片化；运行时在 hugepage 不足时会先释放干净的页缓存、整理内存，再重试申请。
