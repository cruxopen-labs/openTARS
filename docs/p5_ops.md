# P5: the TARS operators (contract v0.3)

TARS-mini now runs every operator in the V1.1 OPCODE table, plus the two Qwen3.5 needs, bit-exact
between the golden model and the RTL. The descriptor VERSION stays 0x0002: v0.3 only gives
meaning to encodings that v0.2 rejected, so every v0.2 program and test keeps its result.

| OPCODE | Operator (V1.1 name) | Unit in TARS | TARS-mini implementation |
|---|---|---|---|
| 0x01 | Conv | MXU (+ VPU im2col) | implicit GEMM: the A loader generates im2col addresses, padding = zero beats without a read |
| 0x02 | GEMM | MXU | as before (A3 + B + reuse) |
| 0x03 | ReLU/GELU (activation) | VPU | row engine, one pass, 4 elements per cycle (was element-serial) |
| 0x04 | Softmax / LayerNorm (+ RMSNorm) | VPU | row engine, 2-3 passes, row kept in the hold buffer |
| 0x05 | Conv+ReLU (fused) | MXU+VPU | 0x01 with a required activation |
| (EPI) | GELU, SiLU activations | VPU | in every GEMM / Conv epilogue lane and in 0x03 |

## Contract increments

**Activation code** = EPI[1:0] | EPI[5] << 2 (EPI[5] was must-be-zero): 0 none, 1 relu, 2 relu6,
3 GELU, 4 SiLU; 5-7 and EPI[15:6] are E_FIELD.

**Math primitives** (numerics.py; tm_fp_pkg.sv matches step by step). Fixed sequences of the
contract's fmul / fadd (each rounded once, RNE, FTZ, no FMA) plus exact bit operations.

| Primitive | Sequence | Specials | Error (measured) |
|---|---|---|---|
| exp(x) | n = rint(x * log2e) (ties even); r = (x + n*(-ln2_hi)) + n*(-ln2_lo); degree-6 Horner with 1/720, 1/120, 1/24, 1/6, 1/2, 1, 1; add n to the exponent field (>= 255 inf, <= 0 +0) | x >= 89 inf, x <= -104 +0, NaN canonical | <= 2.9 ulp |
| recip(x) | m = mantissa in [1, 2); y0 = bits(0x7EF311C7 - bits(m)); 3 x (e = 1 - m*y; y = y + y*e); subtract the exponent | +-0 -> +-inf, +-inf -> +-0, NaN canonical | <= 1.13 ulp; exact for powers of two |
| rsqrt(x) | m in [1, 4) (even exponent); y0 = bits(0x5F3759DF - (bits(m) >> 1)); 3 x (e = 0.5 - 0.5*((m*y)*y); y = y + y*e); subtract half the exponent | +-0 -> +-inf, +inf -> +0, x < 0 NaN | <= 1.21 ulp |
| GELU | x3 = (x*x)*x; u = c*(x + a*x3); t = 1 - 2*recip(exp(2u) + 1); y = (0.5x)*(1 + t); a = 0.044715, c = sqrt(2/pi) | GELU(+-1e20) = 1e20 / -0; GELU(-inf) NaN | <= 4.3e-7 abs vs the tanh form |
| SiLU | x * recip(1 + exp(-x)) | SiLU(-1e20) = -0; SiLU(-inf) NaN | <= 1.2e-6 abs |

Two first drafts failed their own tests and were changed before any RTL: plain Newton
`y*(2 - x*y)` left recip(1) = 0.99999994, which made GELU(-1e20) = -5.96e12 instead of -0; and
`h = 0.5*x` underflowed for x near 2^-126, giving rsqrt = NaN. Correction-form steps on a
normalized mantissa fixed both.

**Row ops (0x04)**: CTRL[1:0] = 0 softmax, 1 LayerNorm, 2 RMSNorm. A_OFF / LDA16 = fp32 rows,
SCALE_OFF = gamma, BIAS_OFF = beta (EPI.bias), W_OFF = eps as fp32 bits; softmax takes none of
them; activation must be 0; any output format.

- sum16: element j goes to lane j mod 16; each lane adds in order from +0; the lanes fold pairwise
  (0+1, 2+3, ...) in four levels. fmax: NaN if either is NaN; of equal values the sign is the AND
  (order-free). 1/N = recip(i2f(N)).
- softmax: mx = max; e = exp(x + -mx); y = e * recip(sum16(e)).
- LayerNorm: mean = sum16(x) * 1/N; d = x + -mean; rs = rsqrt(sum16(d*d) * 1/N + eps);
  y = (d*rs)*gamma (+ beta). RMSNorm: rs = rsqrt(sum16(x*x) * 1/N + eps); y = (x*rs)*gamma (+ beta).

**Conv (0x01, 0x05)**: NHWC INT8 image of H x W pixels (LDA16 = pixel stride, default Cin), weights
[n][ky][kx][cin] (N x K, K = KH*KW*Cin), M = OH*OW, OH = (H + 2P - KH)/S + 1. Geometry: GROUP_SIZE
= W; HDR_RSV = H[23:12] | KH[11:8] | KW[7:4] | (S-1)[3:2] | P[1:0]. E_FIELD: zero geometry, empty
output, M != OH*OW, 0x05 without an activation; E_ALIGN: K not KH*KW*Cin or Cin % 16; E_RANGE: the
image extent. Dilation 1 and one group only.

## Decisions taken by default, pending ratification

| # | Decision | Taken | Alternatives |
|---|---|---|---|
| D1 | GELU form | tanh approximation | erf (exact GELU; needs an erf sequence, ~5e-4 apart) |
| D2 | Primitive sequences | magic constant + 3 correction-form Newton steps; degree-6 exp | table lookups; fewer steps with looser bounds |
| D3 | Reduction order | 16 lanes round-robin + fixed 4-level pairwise fold | another lane count (TARS VPU has 128 lanes x 8 sublanes); sequential |
| D4 | Where eps lives | W_OFF field (fp32 bits) | a fixed contract constant; a new field |
| D5 | Conv geometry fields | GROUP_SIZE = W, HDR_RSV = H/KH/KW/S/P | V1.3.1's own conv fields (not in this project's copy of the spec) |
| D6 | 0x05 Conv+ReLU | 0x01 with a required activation (any) | ReLU only; drop 0x05 |
| D7 | Softmax / norm algorithms | multi-pass (max, sum, write) | single pass: online softmax, Welford (fewer passes, different rounding) |
| D8 | GELU(-inf), SiLU(-inf) | NaN (as the sequences give) | define -0 |

## Verification

| Gate | Result |
|---|---|
| tests | 513 (P5 added: primitive accuracy bounds and specials; fp vectors incl. constructed rint ties; row-op and conv error codes; fuzz with all 6 opcodes and 5 activations; conv coverage (padding, stride 2, 1x1, both opcodes, strided images); compiled conv chains on 3 array shapes; hold-buffer fuzz on a 3x16 array; compiled row ops on normal data incl. all-negative softmax rows; queue end under 90 % write backpressure) |
| mutants | 48 / 48 caught, each with its own test subset. The first full run caught 41; of the 7 left, 1 was stale (its text had become non-unique), 1 was equivalent (2*recip(d) = recip(d/2) bit for bit, since recip normalizes the mantissa) and was replaced, and 5 were test holes, closed and re-run: activation code 5 (another rule fired first in the error test), the 16-lane order and the fold pairing (random-byte rows span 1e-38..1e38, so their sums hardly depend on order), the queue ending before a row op's last write (needs heavy write backpressure on the last descriptor), softmax's max starting from 0 (needs an all-negative row) |
| found by tests during P5 | recip(1) inexact; rsqrt NaN near 2^-126; RTL syntax limit (non-blocking writes to arrays in loops); a `//` comment that swallowed the eps and 1/N latches (caught at once by the fuzz); coverage drops from generator changes (fixed by targeting, not by lowering thresholds) |

## Performance (no backpressure; machine: R*C MACs/cycle, 16-byte read + 16-byte write channels)

Roofline = max(MACs / R*C, max(min bytes read, min bytes written) / 16). Reproduce:
`python3 tools/perf_p5.py`.

| workload | 8x8, lat 2 | 8x8, lat 16 | 16x16, lat 2 | 16x16, lat 16 |
|---|---:|---:|---:|---:|
| softmax 64x1024 -> bf16 | 32.8 % | 32.2 % | 32.8 % | 32.2 % |
| LayerNorm 64x1024 (+beta) | 33.5 % | 32.9 % | 33.5 % | 32.9 % |
| RMSNorm 64x1024 | 49.4 % | 48.0 % | 49.4 % | 48.0 % |
| GELU elementwise 64x1024 | 96.8 % | 91.8 % | 96.8 % | 91.8 % |
| GEMM 64x256x256 + ReLU | 90.3 % | 86.2 % | 76.3 % | 73.2 % |
| GEMM 64x256x256 + GELU | 90.3 % | 86.2 % | 76.3 % | 73.2 % |
| Conv 3x3, 28x28x32 -> 32, + ReLU | 92.5 % | 88.9 % | 86.2 % | 82.7 % |

- GELU fused into a GEMM costs no cycles (same as ReLU): the parallel epilogue has a GELU per lane.
- Conv runs at the GEMM path's efficiency (83-93 %).
- The standalone activation went from 9-14 % (element-serial) to 92-97 % via the row engine; a
  row-reader depth of 8 left it at 49 % at latency 16, depth 32 hides the latency.
- Row ops are bound by their passes: each pass streams the row at one beat per cycle, so 3 passes
  reach ~33 % and 2 passes ~49 %. Next steps: read 2-4 beats per cycle from the hold buffer in the
  later passes (microarchitecture), or single-pass algorithms (D7, a contract change).
- The P3-reuse tables (perf_report.py) put reads and writes on one 16-byte port; the DDR model and
  AXI have separate channels, so those memory rooflines are conservative for output-heavy shapes.
