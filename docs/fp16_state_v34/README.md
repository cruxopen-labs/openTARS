# Packed FP16 DeltaNet state, firmware v3.4

## Scope

Only the DeltaNet matrix state `S` uses IEEE binary16 storage. Loads expand to
FP32; products, additions, dot accumulators, normalization and SFU remain FP32.
Both the decay store and the update store round to FP16, round-to-nearest-even.
Binary16 subnormals and signed zero are preserved; overflow produces infinity
and NaNs are canonicalized. Existing FP32 arithmetic still uses its existing
flush-to-zero behavior.

KV caches, convolution history, temporary vectors and INT8 weights are unchanged.
An MTP export compresses both copies of `S`.

The descriptor version remains 2. Previously reserved encodings now select
FP16 ELT A/output/C or FDOT NN B; see [the contract](../v04_contract.md).
FP16 exports require firmware `0x00030400`; the runtime rejects older hardware
before loading their image. FP32 exports remain supported.

## Real Qwen3.5-0.8B exports

Both exports use the same checkpoint, W8A8 lowering and 256-token context.

| Item | FP32 state | FP16 state |
|---|---:|---:|
| Matrix state `S`, without MTP | 18 MiB | 9 MiB |
| `image.bin` | 930,086,912 bytes (887 MiB) | 920,649,728 bytes (878 MiB) |
| End of state/scratch region | 173,238,400 | 163,801,216 |

The entire 756,848,512-byte weights/table/padding tail after that region is
byte-identical between exports. Its SHA-256 is
`849fd39270c482548155d2d06c70477bfa6737f09ed7505c1d61aad2d066a06a`.
The FP32 golden hashes at positions 0, 4 and 8 also reproduce the earlier
FP32-state export.

### Numerical caution

This is **not lossless**. Golden-model greedy decoding already changes:

* `The capital of France is`: FP32 continues ` Paris.`; FP16 continues
  ` the capital of the country.`
* The chat example changes from `a monumental testament to human ingenuity`
  to `a monumental and enduring symbol of Chinese`.

Different greedy continuations do not quantify the error on identical inputs.
Use the teacher-forced comparison below. Its fixed English and Chinese samples
are an engineering check, not a corpus-quality or perplexity guarantee.

## Reproduction

```sh
python tools/export_kv260.py --help
# Export the same checkpoint/context twice, with --state-dtype fp32 / fp16.
# Exports default to fp32 (FP16 measured slower); Qwen3 (no DeltaNet S) is always fp32.

python tools/check_state_precision.py \
  --fp32 build/data_fp32 --fp16 build/data_fp16 \
  --tokens 256 --out build/state_precision_golden.json

# On the board, with v3.4 installed:
sudo ./run.sh selftest --data data_fp16
sudo ./run.sh selftest --data data_fp32_v34
sudo ./run.sh precision --fp32 data_fp32_v34 --fp16 data_fp16 \
  --tokens 256 --out results/state_precision.json
sudo ./run.sh profile --data data_fp16 --mode decode \
  --positions 16,64,128,192,250 --require-operators \
  --out results/fp16_decode.json
sudo ./run.sh profile --data data_fp16 --mode prefill --check chat \
  --require-operators --out results/fp16_prefill.json
# Repeat both profile commands with data_fp32_v34 and distinct output filenames.
```

The comparison completes all FP32 runs before loading the FP16 image into the
same physical device. It then replays the same tokens, comparing FP32 logits
using FP64 metrics. Reported metrics include top-1 agreement, maximum absolute
and relative-L2 error, mean KL and next-token NLL.

On the board, `data_fp32_v34` is a fresh upload of this experiment's FP32 export.
The existing `data_fp32` link points to the original deployment and is not used
as the numerical baseline: despite matching short golden sequences, its image
tail has a different hash. The original model directory remains unchanged.
Both fresh uploads passed every file's SHA-256 check on the board. Their
inventories and completed regression logs are saved in [validation/](validation/).

## Performance limits and validation status

Storage savings are exact, but do not imply a 2x throughput improvement.
FDOT NN streams packed half values without adding arithmetic lanes. ELT still
handles four values per step; adjacent half groups currently reread an aligned
beat, and masked write beats still count at their physical bus width.

Conversion tests include every binary16 bit pattern and midpoint rounding
boundaries. Independent NumPy recurrence checks distinguish FP32 accumulation
from half accumulation. RTL checks cover wide/fallback FDOT, strided and
in-place ELT, range validation, prefill and decoding.

An independent shadow-model check also passed: an unchanged FP32 descriptor
program rounds only `S` writes with NumPy. Its layer outputs, final normalization
and expanded final states match the packed path bit for bit, across a three-token
prefill followed by one decode token on the small model.

Final FPGA-target regression: **261 passed**, including both state precisions
for MTP speculative hit/miss and ping-pong state. The targeted MTP rerun also
passed both cases. Final host regression: **352 passed, 3 skipped**.
Generic-core validation initially exposed a missing observation-port
wire in `tb_core.sv`, now connected: **43 passed** in the generic-core rerun.

Vivado finished with all timing constraints met at 100 MHz: WNS +0.008 ns,
WHS +0.010 ns; all 19 bus-skew constraints passed (minimum slack 2.676 ns).
LUT usage is 105,992 (90.50%), BRAM 118 tiles (81.94%), DSP 902 (72.28%).
Timing margin is very small. The existing TIMING-9 CDC methodology warning remains.
See [build_summary.json](build_summary.json); full implementation artifacts are
preserved under `build/fp16_v34_artifacts`, separately from v3.3.

The new firmware was temporarily installed and both fresh FP32/FP16 exports
passed the full board self-test: 24 hardware programs each and real-model
chunked prefill/decode logits bit-identical to their respective goldens.
The first attempt stopped before running descriptors because hugepage allocation
found only 513 of 528 requested pages. Cache reclamation and compaction resolved
this without rebooting or changing the test.

## Measured results on the KV260 (v3.4 firmware, 100 MHz)

### Teacher-forced precision (FPGA, identical tokens)

[validation/state_precision.json](validation/state_precision.json); every position.

| Sample | Tokens | Top-1 agreement | Max relative L2 | Max abs | Mean KL | Mean NLL FP32 → FP16 |
|---|---:|---:|---:|---:|---:|---:|
| English | 241 | 90.87% | 0.159 | 2.52 | 0.0193 | 4.023 → 3.986 |
| Chinese | 229 | 89.08% | 0.194 | 2.64 | 0.0223 | 4.600 → 4.610 |

About one position in ten has a different argmax. Mean NLL is not systematically
worse on these two samples, but this is far from bit-equivalent behavior.

### Performance (same v3.4 bitstream, FP32 vs FP16 export)

Core time per step, read bytes and write bus bytes
([validation/](validation/) `fp32_*.json`, `fp16_*.json`):

| Workload | FP32 ms | FP16 ms | Change | Read MB | Write MB |
|---|---:|---:|---:|---:|---:|
| decode t=16 | 131.571 | 135.678 | +3.12% | 913.4 → 909.5 | 63.1 → 45.1 |
| decode t=64 | 132.538 | 136.590 | +3.06% | 918.4 → 914.6 | 63.1 → 45.1 |
| decode t=128 | 133.713 | 137.752 | +3.02% | 925.1 → 921.2 | 63.2 → 45.1 |
| decode t=192 | 134.837 | 138.958 | +3.06% | 931.8 → 927.9 | 63.2 → 45.2 |
| decode t=250 | 135.769 | 139.901 | +3.04% | 936.8 → 933.0 | 63.2 → 45.2 |
| chat prefill, 20 tokens | 1785.851 | 1824.982 | +2.19% | 7295.7 → 7230.8 | 1141.9 → 782.3 |

**FP16 state is slower, not faster, in this implementation.** Per decode step,
ELT rises from 47.76 to 51.93 ms (+8.7%); FDOT NN falls by only 0.1 ms; GEMV and
other classes are unchanged. Reads fall by only about 3.9 MB per step because narrow
ELT still issues one 16-byte read per four half values (the same request count as
FP32), and the half-to-float conversion adds latency. Writes fall by 18 MB.
A speed benefit would require ELT to consume a full packed beat per request
(8 or more half values per read), which is further RTL work under a 90.5% LUT budget.

After measurement the original firmware (SHA-256 `c40eb336…`, version 0x30300)
was reinstalled and the production `/home/ubuntu/tars` self-test passed again
([validation/selftest_restored.log](validation/selftest_restored.log)).
The v3.4 bit, both uploaded exports and raw results remain in
`/home/ubuntu/tars_fp16_20261108`.

The CPU golden-model 256-token comparison was stopped after one hour (one baseline
sample done); the FPGA results above are backed by bit-exact board self-tests.
