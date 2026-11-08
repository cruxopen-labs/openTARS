# KV260 v3.3 operator profile, 2026-11-08

Measured on the KV260, Kria Ubuntu 22.04 / Linux 5.15, core 100 MHz, HP interfaces
250 MHz, INT8 W8A8, greedy decode. Same exported model data and numerical contract
before/after instrumentation; ordinary runtime paths, no descriptor splitting.

## Build and correctness

- Original installed bitstream SHA-256:
  `c40eb3365d641bcf831a8351429ccebfaa4dcd36e2e4847376074497f9dcebaa`.
- Instrumented bitstream SHA-256:
  `7cf799fb8ca7cab9907f0148384f53f8a1d5d7329eaca61bbdbfdc23c387eb9a`.
- New build: 104,967 LUTs (89.62%), 58,724 registers, 903 DSPs (72.36%),
  118 BRAM tiles (81.94%), 0 URAM.
- Setup WNS +0.044 ns / TNS 0; hold WHS +0.010 ns / THS 0; pulse-width slack
  +0.500 ns. All 19 post-route bus-skew constraints have positive slack (minimum
  +2.418 ns). Timing summary retains one
  `TIMING-9 Unknown CDC Logic` methodology warning; do not equate timing closure
  with a complete CDC audit.
- Verilator: 191 existing FPGA operator tests passed; 25 profiler tests passed.
  Contract/golden/compiler/profiler host run: 290 passed, 3 RTL-only tests skipped.
- Board selftest for **each** model: 24 random hardware-path programs (including
  invalid descriptors) passed, plain/chat prompt + decode logits matched the
  exported golden SHA-256 values; `ALL PASS`.
- Each profile validates that operator busy cycles sum exactly to core cycles.
  Hashes at later context positions are recorded but are not independently golden
  checked unless `golden_verified` is true.
- All 11 common original/instrumented sample logits hashes match, including the
  ten sampled decode positions across the two models. Repeat-run sample hashes also match.

The original repo's v3.3 resource record was 104,879 LUTs (89.55%). The new
implementation is only 88 LUTs larger after synthesis/place/physical optimization,
but this is a comparison of separate implementations, not an isolated counter
bank area estimate.

## Decode: t=16 and t=250

Values below use the first instrumented run. Times are **core busy** time, not
wall-clock chat latency. Read bandwidth includes all 128-byte bridge reads, not
just weights. Write bandwidth counts 32-byte bus beats including masked bytes.

| Model / position | Core ms/token | Read GB/s | Write GB/s | GEMV | ELT | FDOT NT+NN | Norm | Overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B t=16 | 67.104 | 9.503 | 0.268 | 75.88% | 1.52% | 7.15% | 3.39% | 6.82% |
| Qwen3-0.6B t=250 | 95.186 | 7.851 | 0.202 | 53.47% | 1.12% | 31.82% | 2.40% | 4.80% |
| Qwen3.5-0.8B t=16 | 131.639 | 6.938 | 0.479 | 48.61% | 36.28% | 6.03% | 2.14% | 4.96% |
| Qwen3.5-0.8B t=250 | 135.831 | 6.897 | 0.465 | 47.13% | 35.16% | 8.73% | 2.07% | 4.80% |

The other categories are RoPE, quantization, softmax and standalone activation.
See the JSON for all classes, raw counts and the intermediate context positions.

Original / instrumented / repeat core ms/token:

| Model / position | Original | Instrumented | Repeat |
|---|---:|---:|---:|
| Qwen3 t=16 | 67.047 | 67.104 | 67.150 |
| Qwen3 t=250 | 95.133 | 95.186 | 95.707 |
| Qwen3.5 t=16 | 131.531 | 131.639 | 131.629 |
| Qwen3.5 t=250 | 135.717 | 135.831 | 135.797 |

The first instrumented readings differ from the original by 0.005–0.103% at
the five common positions (16, 64, 128, 192, 250); the t=250 Qwen3
repeat is ~0.60% above the original. This is not a statistical proof of zero
instrumentation overhead. Qwen3 t=8 also had a one-off slower GEMV run. Raw
measurements are preserved rather than replacing them with a favorable average.

## Prefill: real 20-token exported chat prompt, default chunks of four

| Model | Total core ms | ms/token | Read GB/s | Write GB/s | GEMV | ELT | FDOT NT+NN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 | 577.235 | 28.862 | 5.933 | 0.494 | 65.77% | 3.37% | 8.93% |
| Qwen3.5 | 1778.295 | 88.915 | 4.103 | 0.642 | 26.45% | 53.90% | 8.57% |

GEMV write-request backpressure is 36.70% of **GEMV cycles** for Qwen3 and 38.01%
for Qwen3.5, about 24.1% and 10.1% of total prefill cycles respectively. It is
nearly zero in single-row decode. Thus prefill GEMV is not merely a DDR weight-read
problem: multi-row output/epilogue/write acceptance deserves investigation too.

Qwen3's t=250 FDOT NN read-request wait is 57.86% of that class's cycles, about
9.19% of total token cycles. This is evidence of pressure on the request path, not
a measurement of all memory stalls. Qwen3.5 ELT has nearly zero measured
read-request/write-request waits despite taking ~47.8 ms/token: accepted reads,
operand collection and the narrow execution schedule still consume time. Zero
request backpressure does **not** mean all operands are available or that arithmetic
lanes are fully occupied.

## FP16 decision

FP16 is more promising for **Qwen3.5**, especially prefill, than for short-context
Qwen3. It does not reduce the existing INT8 weight traffic or integer GEMV work.

If, hypothetically, *all* ELT and FDOT execution became twice as fast with other
costs unchanged, the measured shares imply:

| Case | ELT+FDOT share | Conditional whole-model speedup |
|---|---:|---:|
| Qwen3.5 decode t=16 | 42.31% | ~1.27x |
| Qwen3.5 20-token prefill | 62.47% | ~1.45x |
| Qwen3 decode t=16 | 8.67% | ~1.05x |
| Qwen3 decode t=250 | 32.94% | ~1.20x |

These are Amdahl scenarios, **not measured FP16 gains or guarantees**. Changing only
storage width does not halve ELT execution; conversion, packing, instruction overhead
and the retained FP32 units matter. Some Attention waits may remain.

Recommended bounded experiment: packed FP16 DeltaNet state / selected ELT and FDOT
operands, FP32 accumulation and FP32 SFU/norm retained, explicitly increasing useful
elements per cycle where possible. Check quality against the **current W8A8 FP32**
baseline before accepting the numerical-contract change. The earlier software
precision experiment is informative, not a substitute for that comparison.

The state footprint can fall from 18 to 9 MiB and six state passes from 113.25 to
56.62 MB/token, but total bytes do not halve. Full FP16 arithmetic across the core
has not been implemented or synthesized by this profiling change.

## Raw reports and reproduction

- `qwen35_baseline.json`, `qwen3_baseline.json`: original firmware, aggregate only.
- `qwen35_profile.json`, `qwen3_profile.json`: classes and contexts; include additional
  short independently checked steps and a five-token prefill sample.
- `qwen35_prefill20.json`, `qwen3_prefill20.json`: independently checked chat prefill.
- `qwen35_repeat.json`, `qwen3_repeat.json`: a fresh process/reload per model.

Commands and counter semantics: [../kv260_profiling.md](../kv260_profiling.md).
Profiling software was staged separately from the existing runtime, with read-only
links to its exported model data. Original firmware was backed up before replacement;
model files and the existing runtime scripts were not overwritten.

After collecting these reports, the original installed bitstream was restored,
its SHA-256 checked, and its full Qwen3.5 board selftest passed again. At the end of
this profiling experiment the board ran the original, not instrumented firmware. The staged
profiler software, results, backup and instrumented `profile.bit.bin` remain under
`/home/ubuntu/tars_profile_20261108`. The rebuilt local image is
`build/profile_v33_artifacts/tars_kv260.bit.bin` (preserved before the v3.4 rebuild);
checked-in deployment binaries were not replaced.

To reactivate the staged profiling image, first ensure no TARS run is active, then
on the board (loading interrupts FPGA use):

```sh
cd /home/ubuntu/tars_profile_20261108
sudo install -m 644 profile.bit.bin /lib/firmware/xilinx/tars_kv260/tars_kv260.bit.bin
sudo ./run.sh profile --data data --positions 16,64,128,192,250 \
    --require-operators --out results/qwen35-new.json
```
