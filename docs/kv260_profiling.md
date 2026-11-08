# KV260 whole-queue profiling

The profiler observes the normal `Qwen.step` / `Qwen.prefill` paths. It does **not**
split a descriptor queue, reset between its operators, or change the dependency schedule.
Counter readout happens after the core halts and the bridge is idle. Numeric formats,
descriptors, model images and golden logits are unchanged.

Actual KV260 measurements for both models, original-firmware comparisons, raw JSON
and FP16 estimates: [v3.3 board report](profiling_v33/README.md).

## Build and run

The checked-in v3.3 bitstream has aggregate counters only. The new RTL adds an
independent capability register, without changing the v3.3 numeric contract/version:
`PROFILE_CAP` at `0x05c` is `0x50524601`. Old firmware reads zero.
**Updating Python alone does not add operator counters.**

Build on the PC with Vivado 2025.2, then package and install as in the board README:

```sh
vivado -mode batch -source fpga/kv260/build.tcl
fpga/kv260/package.sh
# Copy deploy/kv260 to the board, keeping your exported data directories.
```

On the board, with the new software and firmware:

```sh
sudo ./setup.sh
sudo ./run.sh selftest
sudo ./run.sh profile --data data --positions 16,64,128,192,250 \
    --require-operators --out qwen35-profile.json
sudo ./run.sh profile --data data_qwen3 --positions 16,64,128,192,250 \
    --require-operators --out qwen3-profile.json
sudo ./run.sh profile --data data --mode prefill --check chat \
    --require-operators --out qwen35-prefill.json
```

Without `--require-operators`, the old v3.3 firmware is accepted and produces
**aggregate-only** measurements. Use `--mode both` to collect prefill and decode;
the decode measurement starts a fresh conversation. Prefill uses the selected
exported prompt's actual length and the runtime's default chunking, not a synthetic
20-token claim. `--check plain` is the default. Decode starts at position zero,
teacher-forces that prompt, then uses greedy tokens to reach the requested positions.
Validate positions against your export's context (default 256).

The script checks every visited position present in the selected `expected.json`
sequence, including warm-up. A mismatch aborts. For later positions the report includes
a logits SHA-256 and `golden_verified: null`, **not** a claim of independent verification.
Run the ordinary selftest before benchmarking.

`--mhz` defaults to 100; `--ddr-gbs` defaults to 12.79, the separate four-HP-port,
250 MHz **read-only** traffic-generator result. Change these if the hardware changes.
Do not compare a mixed read/write workload to that reference as an achievable mixed ceiling.

## Counters and attribution

`0x100 + 16*class + 4*metric`, all read-only, cleared by core reset:

| Class | Operator | Includes |
|---|---|---|
| 0 | overhead | descriptor fetch, CRC, validation, queue transitions and their read waits |
| 1 | gemv | INT8 GEMM, including internal row batches and output drain (KV260 has ARR=0) |
| 2 | elt | FP32 elementwise operations, including fused activation |
| 3 / 4 | fdot_nt / fdot_nn | FP32 dot products |
| 5 | rope | rotary embedding |
| 6 | quant | per-row INT8 quantization |
| 7 | norm | LayerNorm, RMSNorm, L2Norm |
| 8 | softmax | plain or causal softmax |
| 9 | act | standalone activation descriptors |

| Metric | Meaning |
|---|---|
| 0 | busy cycles in this class |
| 1 | cycles a core read request is asserted but not accepted |
| 2 | cycles a core write request is asserted but not accepted |
| 3 | cycles the bridge's pending read overlaps an unacknowledged write |

Class busy cycles partition the core's `CYCLES` exactly. The script verifies the sum.
The three wait counters can overlap. **Do not add them, subtract their sum to get
compute time, or interpret an accepted read as data having returned.** A wait is
attributed to the class active when it was observed; outstanding bridge transactions
may cross a descriptor boundary. These are not exclusive root-cause or arithmetic-active counters.

The bank uses 40 x 32-bit counters. Every individual queue must finish in fewer than
`2^32` core cycles (42.95 seconds at 100 MHz); software rejects longer profiled queues.
Normal prefill may contain several queues; software sums **all** of them, rather than
only reporting `last_perf`. Aggregated totals use Python integers.

Existing global counters remain unchanged:

- `RD_BEATS` (`0x028`): completed **128-byte** bridge read beats, including gathered
  reads for narrow requests, descriptors and padding.
- `WR_BEATS` (`0x02c`): acknowledged **32-byte** write bus beats, including masked bytes.
- Read bursts, address-queue wait, read/write request wait, hazard cycles and the
  read-occupancy integral (`0x040..0x058`).

The JSON reports total bytes, GB/s using **core busy time** as denominator, mean
read latency, wait ratios, operator cycles and percentages, and descriptor MAC counts.
This is not instantaneous DDR utilization or FPGA-wide compute utilization. The
core timer excludes final bridge drain; the host timer includes counter reads,
hashing and report bookkeeping. Do not compare the profiled host timer to normal
chat latency as if instrumentation were free.

Per-operator DDR bytes are deliberately **not** inferred from return-time counters:
requests can outlive a descriptor. Total DDR traffic is measured; descriptor MAC
counts are static work, not measured lane activity.

`PROFILE=0` on `tm_kv260_top` removes the counter bank and reports capability zero.
This allows an area/timing comparison. RTL tests compare on/off cycle counts and DDR
bytes at the same stalls/latency, test every class and verify AXI-Lite accounting.
**Verilator equivalence does not establish Vivado timing closure.** Recheck utilization
and setup/hold timing for the instrumented build before using it on the board.

## FP16: what this measurement can decide

The INT8 weights and integer dot products do not become smaller when FP32 state
and vector arithmetic become FP16. A pure FP16 substitution also need not improve
operations/cycle: the datapath must pack more elements or exploit the saved area.

For Qwen3.5, DeltaNet state storage is 18 MiB in FP32, potentially 9 MiB in FP16.
The six full state reads/writes per decode token would shrink from about 113.25 MB
to 56.62 MB if actually packed in DDR; expanding values back into 32-bit storage
would not save that bandwidth. Changing only state storage leaves FP32 arithmetic
throughput unchanged and adds conversion costs.

The existing `docs/fp16_state/` experiment is **software precision evidence**, not
an FPGA performance or area measurement:

- `s16`: state rounded to FP16 after decay/update, arithmetic FP32.
- `f16`: FP16 recurrence products/sums/state; exp/softplus remain FP32.
- `f16a32`: FP16 operands/state with FP32 dot accumulation.
- The checked-in English/Chinese teacher-forced runs show close logits for these
  variants, but not identical logits/tokens. This is not a broad quality guarantee.
- The experiment compares variants to its transformers FP32 reference; it does not
  prove bit-exact equivalence to the current W8A8 RTL. It does not test converting all
  core operators (e.g. softmax, normalization and SFU) to FP16.

Token-weighted top-1 agreement with that FP32 reference in the saved JSON:

| Recurrence variant | English, 2048 tokens | Chinese, 1642 tokens |
|---|---:|---:|
| FP16 state, FP32 operations (`s16`) | 99.07% | 99.94% |
| FP16 recurrence (`f16`) | 98.93% | 99.82% |
| FP16 recurrence, FP32 dot accumulation (`f16a32`) | 99.17% | 99.82% |
| W8A8 linears, FP32 recurrence | 87.99% | 94.70% |
| W8A8 linears, FP16 recurrence | 87.60% | 94.34% |

These last two rows share the FP32 reference; they are **not** agreement of FP16
against the W8A8 baseline. Small top-1 changes can still change greedy generation.
The saved English window-weighted NLL increases from about 0.711 (W8A8) to 0.761
(W8A8+FP16 recurrence). Do not infer negligible additional quality loss merely
because the full-precision recurrence experiment looks close.

First inspect `elt`, `fdot_*`, `norm`, `softmax`, `act`, and GEMV shares separately
for prefill and decode. If a fraction `f` of measured time is genuinely made twice as
fast, Amdahl's estimate is `1 / (1 - f + f/2)`, **not** 2x for the whole model.
Do not apply it to overlapping wait counts or assume halved bytes imply halved time.
The safest next experiment is packed FP16 state / selected vector operations with
FP32 accumulation and FP32 SFU/norm retained, then validate against the current W8A8
baseline and remeasure synthesized area, timing and board performance.

FP16 changes the numerical contract. A new FP16 golden model can still be matched
bit-for-bit by new RTL, but it cannot claim unchanged logits against the old FP32
golden model. This profiling change does not introduce FP16.
