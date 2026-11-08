# Contract v0.4: what Qwen3.5's LA and FA blocks need

Target: one Gated DeltaNet block ("LA", linear attention) and one gated full-attention block
("FA") of Qwen3.5-0.8B, prefill and decode, single core, every state in DDR. Source of the
structure: transformers 5.19 `models/qwen3_5/modeling_qwen3_5.py` and the 0.8B `config.json`
(hidden 1024, 24 layers in a 3 LA : 1 FA pattern, LA 16 heads x 128, FA 8 query / 2 KV heads x
256, partial RoPE on 64 of 256 dims, theta 1e7, SwiGLU 3584, zero-centered RMSNorm).

Every addition takes bits that v0.3 required to be 0 (or codes v0.3 rejected), and 0 keeps the
old meaning: every v0.3 program and test is unchanged. The new OPCODEs are not in V1.1's table
(0x01-0x05); they are pending ratification against the main spec.

## 1. Encodings

| Item | Field | v0.3 | v0.4 |
|---|---|---|---|
| ELT | OPCODE 0x06 | E_OPCODE | elementwise, below |
| ROPE | OPCODE 0x07 | E_OPCODE | rotary embedding |
| FDOT | OPCODE 0x08 | E_OPCODE | fp32 dot products on the row engine |
| activations | EPI ACT codes 5, 6, 7 | E_FIELD | sigmoid, softplus, exp |
| row ops | CTRL[2:0] of OPCODE 0x04 | 3-7 E_FIELD | 3 L2NORM, 4 QUANT (5-7 E_FIELD) |
| causal softmax | K of a softmax row op | must be 0 | row m keeps elements [0, K + m) |
| W8A8 GEMM | CTRL QM = 3 (V1.3.1 "group") | E_FIELD | QM_ROWCOL: per-row scales at BASE + 16 HDR_RSV |

**ELT (0x06)**: y = act(A) op B, op = CTRL[1:0]: 0 MUL, 1 ADD, 2 MULADD (y = C + act(A) * B, C at
BIAS_OFF with EPI.bias = 1 and the output's row stride; fp32, or FP16 in v3.4 below), 3 SUB (act(A) + -B).
HDR_RSV[1:0] = B mode: dense (row stride GROUP_SIZE * 16, 0 = dense), a row vector B[n], one value
per row B[m], one value in all B[0]; HDR_RSV[2] = A as a row vector A[n]. B at W_OFF; 16-byte
aligned except the per-row and scalar modes (4-byte). An ELT may run in place: A (dense) or C may
be exactly the output (same offset, stride and format); any other overlap stays illegal.

**ROPE (0x07)**: rows of N fp32; R = K rotary dims (a multiple of 32, <= N); table rows at W_OFF,
R fp32 each: cos[0:R/2] then sin[0:R/2]; row m uses table row m // G, G = GROUP_SIZE >= 1 (heads
per position). With h = R/2: y[i] = x[i] c + -(x[i+h] s), y[i+h] = x[i+h] c + x[i] s, y[n >= R] =
x[n] (torch's rotate_half form; each product and sum rounded once).

**FDOT (0x08)**: y[m][n] = sum over k of A[m][k] * B(k, n), A rows of K fp32 (K a multiple of 16).
CTRL[0] = 0 (NT): B is [N][K] and the sum is the contract's 16-lane order (element k to lane
k mod 16, in order, then the pairwise fold), as the row sums. CTRL[0] = 1 (NN): B is [K][N] and
the sum runs in k order from +0. B row stride GROUP_SIZE * 16 (0 = dense). No scale, bias or
activation; the output conversion of EPI applies.

### v3.4: packed FP16 DeltaNet state

Firmware `0x00030400` additionally accepts:

- ELT `HDR_RSV[3] = 1`: A is binary16, two bytes/element. B remains FP32.
  Bits [2:0] retain their broadcast meanings; bits [23:4] must be zero.
- ELT `EPI.out = 3`: Y is binary16. MULADD C has Y's format and row stride.
  A and C may alias Y only with exactly the same format and layout as Y.
- FDOT **NN only**, `HDR_RSV = 8`: B is binary16; A, products, accumulators and
  output arithmetic remain FP32. NT and other HDR_RSV encodings remain rejected.
- All row strides and range checks use **bytes**, including packed state views.
  Half-state conversions use IEEE binary16 round-to-nearest-even, preserve signed
  zero and half subnormals, overflow to signed infinity, and canonicalize NaNs
  (`0x7e00` on store, `0x7fc00000` on load). Half subnormals expand exactly to
  normal FP32 numbers; the existing FP32 FTZ arithmetic itself is unchanged.

The numeric sequence is FP32 decay -> round/store FP16 -> expand for FP32
`S^T k` -> FP32 delta/update -> round/store FP16 -> expand for FP32 `S^T q`.
No FMA or FP16 accumulator is introduced. SFU, normalization, attention KV cache,
convolution history, intermediate vectors and INT8 weights do not change.
The 128-byte FDOT NN reader consumes four groups of 16 half values per beat,
retaining 16 FP32 products/cycle. ELT still executes four elements per step;
packing state is not a promise of twice the execution throughput.

Old descriptors (`HDR_RSV[3] = 0`, original output codes) keep their semantics.
Descriptor VERSION remains `0x0002`; the new encodings used previously rejected
bits/codes. The exporter marks `state_dtype` and `min_firmware` and the runtime
rejects FP16-state exports on old firmware before loading their image.

**Row ops**: L2NORM = RMSNorm without the 1/N (rs = rsqrt(sum16(x x) + eps)). QUANT: amax =
max16(|x|), q = q8(x, 127 * recip(amax)), s = amax * fp32(1/127); INT8 rows to Y (EPI.out = INT8,
OUT_SCALE = 0) and the M row scales (fp32) to SCALE_OFF; an all-zero row gives q = 0, s = 0.

**QM_ROWCOL GEMM**: y = ((i2f(acc) * srow[m]) * scol[n]) (+ bias), each product rounded once;
column scales at SCALE_OFF (one per channel), row scales at BASE + 16 HDR_RSV. Not for conv.

**sigmoid** = recip(1 + exp(-x)) (SiLU's factor). **softplus**: x > 20 gives x (torch's
threshold); else relu(x) + ln(1 + t), t = exp(-|x|), ln(1 + t) = 2 atanh(t / (t + 2)) by a
degree-8 odd series (coefficients 1/17 ... 1/3, 1) - within 6 ulp of ln(1 + e^x). **exp** =
the v0.3 exp.

## 2. Lowering (tars_mini/llm.py)

| Model step | Descriptors |
|---|---|
| RMSNorm (1 + w) | row RMSNORM, gamma = fp32(1 + w) folded by the host |
| any linear | ROW QUANT (per token) + GEMM QM_ROWCOL (per-channel INT8 weights) |
| beta = sigmoid(b), softplus(a + dt_bias) | GEMM with the activation fused (dt_bias = the GEMM bias) |
| g = -exp(A_log) softplus, decay = exp(g) | ELT MUL (row vector) + activation EXP |
| causal depthwise conv1d (k = 4) + SiLU | ELT MUL + 3 ELT MULADD on row-shifted views of a history buffer, activation SILU |
| L2 norm of q, k (q times dk^-0.5) | row L2NORM per head, the scaling in gamma |
| gated delta rule, per token and head | ELT MUL (decay, scalar, in place), FDOT NN (S^T k), ELT SUB, ELT MUL (beta), ELT MULADD (S += k (x) delta, in place), FDOT NN (S^T q) |
| gated RMSNorm, * silu(z) | row RMSNORM per head + ELT MUL with SiLU on A |
| q / k norm, partial RoPE | row RMSNORM per head (q's gamma times hd^-0.5, exact), ROPE (K written into the cache) |
| scores, causal softmax, P V | FDOT NT per query head (GQA by pointer), causal softmax, FDOT NN |
| attention * sigmoid(gate) | ELT MUL with SIGMOID on A |
| SwiGLU, residuals | GEMM with SiLU fused, ELT MUL (in place), ELT ADD |

The q projection's interleaved [q | gate] head layout and the fused qkv projection are split into
separate GEMMs by permuting weight rows on the host, so every per-head view is a uniform stride.
The recurrence runs token by token for prefill too (transformers uses a chunked form for prefill;
the two agree to fp32 rounding).

## 3. Results on the golden model

0.8B dimensions, random weights (the checkpoint is not reachable from this environment), inputs
N(0, 1) with 1 % of channels 20x. Prefill 16 tokens then 4 decode steps: 2355 descriptors.

| Reference | LA block update (out - in) | FA block update |
|---|---|---|
| transformers, same W8A8 linears | max rel error <= 1.2e-6 | <= 5.5e-7 |
| transformers, fp32 | cos 0.9966-0.9978 | cos 0.9957-0.9983 |

The first line is the lowering's check (only fp32 rounding order differs); the second is what
per-token W8A8 costs with outlier channels (a smoothing step on the host would recover some).

## 4. RTL (rtl/tm_xpu.sv, rtl/tm_core.sv)

**Where each op runs**

| Op | Unit | How |
|---|---|---|
| GEMM QM_ROWCOL | GEMM loader + epilogue | the loader fetches the tile's R row scales as vector beats beside the column scales; each epilogue lane gets one more fmul stage (row scale, 1.0 otherwise: exact), lane latency 12 / 133 / 163 cycles |
| sigmoid, softplus, exp | every fpu_act (epilogue lanes, row-engine lanes, extension unit) | sigmoid shares GELU/SiLU's exp and recip (ysig, 122 cycles); exp = the exp unit + delay (122); softplus its own 152-cycle pipeline (exp 51, add, recip, the series as 8 fused multiply-add steps, relu add, the x > 20 select) |
| L2NORM, causal softmax | row engine | L2: RMSNorm with 1/N = 1.0 (exact); causal: element n of row m enters as -inf (out of the max) and its exp as +0 when n >= K + m; a causal descriptor does not overlap rows |
| ELT, ROPE, FDOT NT / NN, ROW QUANT | extension unit tm_xpu (new) | below |

**The extension unit.** One descriptor at a time, its own reader on the core's read port (up to
VQ = 128 beats in flight, in-order returns) and its outputs into the core's output queue (the same
slots, writer and latency tap as the epilogue). The reader walks the descriptor's 16-byte beats
in a fixed order with running pointers (no multiply) and tags each beat: its operand (A, B, C,
partner x, cos, sin), and on a step's last beat what the step means (output address, lane group,
pass end); a candidate register sits between the walker and the port, so the port handshake
never reaches the walker's compares and adds. The consumer collects a step's beats and fires it on the last one, so it needs no
walker of its own. A step is one 4-element group:

| Op | Beats per step | Datapath (cycles from the issue registers) |
|---|---|---|
| ELT | A, B (+ C) | fpu_act on A (1 / 122 / 152), B and C through 256-entry delay lines (rtl/tm_vdly.sv: one write, one registered read, i.e. a small two-port SRAM), fmul / fadd 3, MULADD's fadd 3 more; per-row and scalar B: the word of the beat, broadcast |
| ROPE | x, partner x, cos, sin (rotary groups); x (the rest) | two fmuls, a fadd (the partner product negated in the first half): 6; the rest passes through 6 registers |
| FDOT NT | A, B per group of 4 k | products into 16 lane accumulators (lane k mod 16, 3-cycle fadd, a lane is touched every 8 cycles or more); at the end of each output: drain, the pairwise fold (12), one element out |
| FDOT NN | A (once per k), then 4 B beats (16 columns) | A[m][k] broadcast, products into lane = column; after K: the 16 lanes out as 4 groups |
| QUANT | pass 0: x; pass 1: x | pass 0: running max of abs per slot; then amax, recip (38), 127 * recip (3); the row scale goes out raw (fp32) to SCALE_OFF, pass 1 converts x with 127 / amax as the INT8 scale |

Each descriptor's outputs leave at one fixed latency after their launch (element steps launch as
they issue, readouts as they enter the conversion stage), so the core's existing shift-register
tap places them in the queue in order. ELT in place (A or C = Y) is safe without any check: each
group is read before its own output is written and no group is read twice.

**Checks.** All new rules evaluate in the existing check sequencer and priority pick: opcodes
0x06-0x08, the per-op field rules (E_FIELD), R a multiple of 32 and <= N, K a multiple of 16, the
B stride, the 4-byte alignment of per-row / scalar B (E_ALIGN), and the extents of B by mode, C,
the ROPE table ((M - 1) / G + 1 rows: a sixth 17-step divider), the QUANT scales and the ROWCOL row
scales (E_RANGE; two more shift-add products).

## 5. RTL results

**Bit-exact.** tests/test_rtl_v04.py (112 tests): every ELT op x B mode (with activations and
INT8 / BF16 outputs), in place and strided, ROPE (3 shapes x 2 formats), FDOT NT / NN x 3 output
formats on strided operands, QUANT + ROWCOL GEMM (softplus fused) + L2NORM + causal softmax, the
v0.4 rules table, every new extent at exactly-fits and one-over, 24 random v0.4 queues (3 array
shapes, stalls 0-70 %, injected field errors, NaN / inf / denormal inputs, in-place and overlap
hazards between descriptors), a 2-slot output queue under 85 % write backpressure, causal
rows whose first masked element is the row's largest, and the small-config Qwen3.5 LA + FA blocks (prefill 3 + decode 1,
534 descriptors): RTL DDR == golden DDR, byte for byte.

**0.8B decode on LPDDR4X** (tools/perf_qwen.py, 8x8 array, DRAMsim3): the golden model runs a
16-token prefill, then the RTL runs one decode token, block by block, each block's DDR checked
against the golden model (all identical):

| block | descriptors (ops) | cycles (1 GHz) | GEMM | row engine + XPU | DDR bytes (r + w) | DRAM roofline | x roofline |
|---|---|---:|---:|---:|---:|---:|---:|
| LA | 136 (ROW 9, GEMM 10, ELT 81, ACT 4, FDOT 32) | 4,613,690 | 3,657,341 | 912,035 | 28.7 MB | 1,681,574 | 2.7 |
| FA | 46 (ROW 16, GEMM 8, ROPE 2, FDOT 16, ELT 4) | 3,262,312 | 3,142,682 | 101,720 | 19.2 MB | 1,126,612 | 2.9 |
| final norm | 1 (ROW 1) | 1,379 | 0 | 848 | 0.0 MB | 720 | 1.9 |
| token | 183 | 7,877,381 (7.9 ms) | | | 47.9 MB | 2,808,906 | 2.8 |

The decode token is GEMV-bound: with M = 1 an output-stationary R x C array uses one row (C MACs
a cycle), so the 8x8 array takes ~K N / 8 cycles per linear, ~2.8x the DRAM roofline for the
weights. The recurrence (16 heads x 6 descriptors over the 128 x 128 state) is ~0.9 M cycles of
LA's ~4.6 M. Next steps for decode: a GEMV mode (the R rows take different K chunks of the same
output, summed at the end; or W-stationary), and keeping the state on chip across a head's six
descriptors.

**Timing** (run_core_h.sh, R = C = 2, VG = 1, ASAP7, fp units and the delay lines as black boxes):
every register path within 650 ps, worst 604 ps (the epilogue's running address, the A range end
of the check, the walker's ROPE flags). The first run had 19 groups over: two fmax levels in one
cycle for QUANT's amax (783 ps, now two stages), the queue's latency tap computed from the
descriptor's fields (741 ps, now registered), the walker behind the port handshake and the
state decode (724 ps: a candidate register, and the read-port select as its own register), the
ROPE half / rotary compares and the 3-input sin address (now registered beside the group index),
and the B extent's 4-term sum (one more sequencer step).

**Mutation check** (tools/mutate_rtl.py): 31 new v0.4 mutants (the extension unit's datapath,
walker and pass ends, the new rules and extents, ROWCOL, L2NORM, causal masking, sigmoid and
softplus) and the 8 P8 mutants whose code v0.4 rewrote: all 39 caught; the other 93 P8 mutants
target unchanged logic (132 in the file). Five first drafts were equivalent and are recorded with
the reason: MULADD's C one or two cycles early (the issue registers hold it for the 3-beat step),
the scalar B word from the per-row pointer (it only moves in 16-byte steps), FDOT NN's A word
re-read (the A beat stays collected through its four steps), causal softmax keeping one more exp
(that element is already -inf), and softplus's threshold at 20 inclusive (ln(1 + e^-20) is far
below half an ulp). Two survivors were real: the unit's output-queue credit was only tested at 128
slots, and with a small queue it was wrong (it reserved 3 slots by subtraction, which wraps below 4
slots; now it counts the launch in progress, tested with 2 slots under backpressure); and no test
put the largest value just past a causal row's valid range.

**Full regression**: 719 tests (8 h 40 min on 2 cores, most of it one ~10 min Verilator build per
array configuration); the queue-credit fix was then rerun on every v0.4 RTL test and the v0.3
fuzz, error and queue tests.

## 6. GEMV: decode's M = 1 linears at the read port's rate

A GEMM with M = 1 (every linear of a decode step, and the head) used one row of the R x C
output-stationary array: C INT8 MACs a cycle, each weight byte used once, so the weights streamed
at C bytes a cycle (8 at 8x8), half of what the 16-byte read port and LPDDR4X (17 GB/s) carry.
Splitting K across the array's rows does not help an output-stationary array (a K step takes one
cycle whatever the row), and a weight-stationary tile is load-bound the same way. So M = 1 now
runs on the extension unit (no contract change: the integer sum is exact, so any order gives the
same bits):

- dispatch: GEMM, M = 1, K <= 4096 (x fits a 256-beat buffer; Qwen3.5's largest K is 3584); else
  the array as before
- reader: the row scale (QM 3), x's K / 16 beats into the buffer, then per output channel its
  scale beat (every 4th channel, or once per tensor), its bias beat (every 4th), its K / 16 W beats,
  contiguous over the whole weight matrix (one long burst stream)
- datapath per W beat: 16 INT8 x INT8 products, two adder levels, the beat's sum into the
  channel's INT32 accumulator (restarted at the channel's first beat); at its last beat the
  contract's epilogue on one lane (i2f, row scale, column scale, bias, the activation, conversion):
  16 + activation cycles, one output element per channel

Bit-exact on the RTL against the golden model: every QM x bias x 4 activation / format pairs, K
from 16 to 624 at 64 KB, K = 16 (a channel ends every cycle), K = 4096 (the buffer exactly full)
and K = 4112 (falls back to the array), and every M = 1 GEMM of the model runs below.

| 0.8B decode, LPDDR4X, 8x8 | before (array) | GEMV | speed-up | GEMM part | weight bytes per cycle |
|---|---:|---:|---:|---:|---:|
| LA layer | 4,613,690 | 2,635,602 | 1.75x | 3,657,341 -> 1,658,924 | 13.3 |
| FA layer | 3,262,312 | 1,531,270 | 2.13x | 3,142,682 -> 1,409,517 | 13.5 |
| head row block (62080 x 1024) | ~7.9 M (C MACs a cycle) | 5,027,747 | ~1.6x | | 12.7 |

The weights now stream at 75-78 % of the DRAM's 17.06 bytes a cycle. What is left in LA is the
recurrence (0.93 M cycles a layer: 16 heads x 6 descriptors over a 128 x 128 state that goes
through DDR four times a token).

## 7. The whole model: 24 layers, the head, greedy decoding

`QwenLowering(..., head=True)` lowers all 24 layers (18 LA, 6 FA, `LAYERS_08B`), the final norm
and the tied head (vocabulary 248320; the 248320 x 1024 INT8 embedding is the head's weight, run as
four 62080-row GEMV blocks because N is 16 bits) into one DDR image; the host does what is not
the NPU's: the embedding lookup (one row copy per token) and the argmax. `tools/run_qwen_full.py`:

    python3 tools/run_qwen_full.py golden      # prompt 8 tokens, 3 + 1 greedy steps (151 s)
    python3 tools/run_qwen_full.py ref         # transformers W8A8 / fp32, logits and layer by layer
    python3 tools/run_qwen_full.py rtl         # the first decode step on the RTL, byte compare

Layout (the contract's field widths decide it): states, queue, scratch and the row-scale
scratch first (HDR_RSV is 24 bits: the row scale must sit below BASE + 256 MB), the weights last;
753 MiB in a 1 GiB image. Each step's descriptors: 14,826 for the 8-token prefill, 2,730 for a
decode token (232 GEMM, all GEMV; 1,482 ELT; 672 FDOT; 260 ROW; 72 ACT; 12 ROPE).

Against transformers (random weights with the model's shapes; results in `docs/qwen_full.json`):

| check | result |
|---|---|
| logits vs W8A8 transformers (same tokens, 4 steps) | cosine 0.995-0.999, argmax equal at 2 of 4 steps |
| logits vs fp32 transformers | cosine 0.79-0.84 (W8A8 per-token on random weights: quantization, not the lowering) |
| each layer alone, fed the golden model's input, vs W8A8 transformers | 22 of 24 layers <= 3.6e-7 relative |
| layers 11 and 22 | 9e-4 / 4e-3: one INT8 activation rounding flips (fp32 sum order); the reference flips the same element under 3e-7 input noise |

So the lowering is the model; what separates the logits from W8A8 transformers is per-token
INT8 rounding sensitivity accumulated over 24 layers of random weights, which a real checkpoint
(smaller activations' outliers aside) would show as well.

On the RTL (the core with an ideal port, the 1 GiB image read from a file, one decode step, all
2,730 descriptors, 3 h 50 min of Verilator) the final 1 GiB is **byte for byte the golden model's**:

| one decode token, all 24 layers + head, 8x8 | cycles | at 1 GHz |
|---|---:|---:|
| ideal 16-byte port (RTL, measured) | 57,339,535 | 57 ms (17 tok/s) |
| LPDDR4X-4266 x32 (RTL, measured; 4 h 35 min) | 76,461,463 | 76 ms (13 tok/s) |
| LPDDR4X, from the per-block runs (section 6) | 76,740,823 (estimate) | 77 ms |

With the ideal port the step is the read port: 56.8 M read beats in 57.3 M cycles (99 %); the
GEMVs are 47.2 M of them (755 MB of INT8 weights, head included, at 16 bytes a cycle) and the
row / extension work 9.9 M. A decode token cannot go faster than its weights through the port;
from here the levers are the port's width (or a second core's), weights below 8 bits, and the
recurrence's state traffic.

Under LPDDR4X the same 909 MB of reads take 76.5 M cycles (11.9 bytes a cycle, 70 % of the DRAM's
17.06): the GEMVs 58.5 M (their 755 MB at 12.9 bytes a cycle, 76 %), the row / extension work
17.1 M (its small, scattered reads pay the latency the weight streams hide). The per-block estimate
of section 6 was within 0.4 % of the measured whole step.

Regression after the GEMV mode and the lean harness: 748 tests pass (10 h on 2 cores).

**Timing with GEMV** (run_core_h.sh, as in section 5): the first GEMV put the descriptor decode
(`op == GEMV`) in front of lane 0's activation, 726 ps. The lane-0 select is now a registered bit
and the GEMV result is registered before the activation (one cycle more latency, 16 + activation);
the next worst, the epilogue's first-row address (x_y + m0 * ldy + n0 << esl, 654 ps), is now
computed a cycle ahead (m0 / n0 settle at the tile's end, many cycles before the epilogue takes the
tile). Worst path 644 ps, every register path within 650 ps. The RTL tests were rerun on the final
RTL: 415 RTL tests (v0.4, core, fp units) pass, and the 9 GEMV mutants are all caught again. The whole-model cycle counts above were measured just before this retime, which adds
at most one cycle per GEMV descriptor (232 a decode step).
