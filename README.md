# TARS-mini

The first end-to-end slice of the TARS NPU: one core, INT8 GEMM and ReLU, 512-bit descriptors
in the V1.3.1 format (VERSION 0x02), bit-exact from descriptor to DDR result, in Python (golden
model) and in SystemVerilog (RTL).

The contract ("TARS-mini 契约 v0.2") is the source of truth. `tars_mini/golden.py` is its
executable form; the RTL must match it bit for bit on every DDR byte and the end status.

## Status: v0.4 done (Qwen3.5-0.8B whole model, 24 layers + head, on one core: RTL == golden; contract v0.4; 1 GHz pre-layout)

| Phase | State | Exit criterion |
|---|---|---|
| P0 contract | done (v0.2) | layout covers 512 bits once; worked example generated and self-checked |
| P1 golden + compiler | done | a trained MLP on the golden model matches float64 within quantization error |
| P2 RTL | **done** | RTL == golden bit for bit: 5 array shapes, random queues, random stalls; every invariant tested |
| P3 performance | done | cycle counts vs roofline, where the cycles go (`docs/p3_perf.md`) |
| A3 chunk streaming | **done** | reads in flight + chunk buffers + continuous feed; bit-exact, 1.6-11.3x (`docs/a3_perf.md`) |
| B parallel epilogue | **done** | row-parallel epilogue, scale/bias once per column block, writer overlaps the next tile; 1.1-10.9x over A3 (`docs/b_reuse_perf.md`) |
| On-chip reuse | **done** | hold buffer (16 KB) keeps A or W rows across the inner tile loop, loop order per descriptor; 16x16 1.5-1.75x |
| P5 operators (v0.3) | **done** | Conv / Conv+act (implicit GEMM), Softmax / LayerNorm / RMSNorm, GELU / SiLU; bit-exact (`docs/p5_ops.md`) |
| P6 overlap | **done** | row ops VG beats/cycle + overlapped rows (97-98 % of roofline); double-buffered accumulators, epilogue beside the next tile (`docs/p6_overlap.md`); D7 = multi-pass |
| P7 prefetch | **done** | cross-tile loader prefetch: read latency hidden; GEMM 16x16 87 %, 8x8 95 %, conv 92-96 % (`docs/p7_prefetch.md`) |
| P8 DRAM + 1 GHz | **done** | LPDDR4X-4266 x32 (DRAMsim3) behind the ports; fp32 units pipelined (fadd / fmul 3 stages, exp 51, GELU 122 cycles) and every core path within 650 ps of ASAP7 logic; under DRAM the pipeline costs -1 to +2.4 %; 101/101 mutants (`docs/p8_dram_1ghz.md`) |
| v0.4 LLM blocks | **done** | contract v0.4 (ELT, ROPE, FDOT, L2NORM / QUANT, causal softmax, W8A8 ROWCOL GEMM, sigmoid / softplus / exp); extension unit tm_xpu; Qwen3.5 LA + FA blocks (small config, and one 0.8B decode token on LPDDR4X) RTL == golden byte for byte; every core path <= 650 ps (worst 604); 39/39 new or rewritten mutants caught (`docs/v04_contract.md`) |
| v0.4 GEMV + whole model | **done** | M = 1 GEMMs on the extension unit at the port's 16 B/cycle (LA 1.75x, FA 2.13x); all 24 layers + final norm + tied head (vocab 248320) in one 1 GiB image, greedy decoding with the host doing embedding and argmax; one decode step (2,730 descriptors) on the RTL byte for byte the golden model's, 57.3 M cycles with an ideal port, 76.5 M under LPDDR4X (76 ms a token at 1 GHz); every core path <= 650 ps (worst 644); 9/9 GEMV mutants caught (`docs/v04_contract.md` sections 6-7) |
| Next (to decide) | - | real checkpoint weights (accuracy); weights below 8 bits or a wider port (decode is weight-bandwidth-bound); the recurrence state on chip; SRAM macros; timing at full size with a commercial flow |

### P2 result

| Check | Result |
|---|---|
| fp32 units vs `numerics.py` | 8 vector sets, ~0.6 M vectors (random bits, subnormal boundary, ties, inf/NaN), all bit-exact |
| random queues, RTL vs golden (full DDR + status) | 60 queues x 5 array shapes (8x8, 16x16, 4x8, 8x4, 3x5), stalls 0/30/70 %, read latency 1-6; 6 long queues at 85 % stalls |
| fuzz coverage (asserted) | >= 4 error codes, canonical-NaN outputs, >= 30 cross-descriptor read-after-write hazards |
| every error code, its priority, exact-fit and one-beat-over DDR bounds | bit-exact on RTL |
| compiled programs (fused / unfused, M- and N-split, strided outputs) | 3 array shapes, bit-exact |
| the P1 MLP (2 GEMMs, 360 images) | 8x8, 16x16 + N-split, 8x4 unfused + M-split: whole 4 MB DDR identical; RTL accuracy = golden accuracy (97.5 %) |
| worst-case accumulator (K = 65520, all -128) | exact in INT32 |
| mutation check (`tools/mutate_rtl.py`) | 11 / 11 injected RTL bugs caught |

Performance baseline (MLP, no stalls, read latency 2): 8x8 array 633,610 cycles (9.1 % of the
array's peak MACs), 16x16 403,000 cycles (3.6 %). The P2 core keeps one read in flight and runs
the epilogue one element per cycle; that is where P3 starts.

## Contract v0.2 (changes from v0.1)

- **Row strides.** LDA16 (bits 111:96) and LDY16 (30:16), in 16-byte units, 0 = dense. The compiler
  can now split a GEMM along N and write column slices of one buffer (concatenation).
- **Canonical NaN.** Every NaN a multiply or add produces is 0x7FC00000 (v0.1 left it open; numpy
  on x86 gives 0xFFC00000 for inf x 0, and FP32 outputs store it as is).

## Layout

```
tars_mini/      contract, numerics, golden model, compiler, disasm, rtlsim (Verilator harness), llm.py (Qwen3.5 lowering), qwen_ref.py (transformers reference)
rtl/            tm_fp_pkg.sv (fp32 steps), tm_fpu.sv (pipelined units), tm_pe.sv, tm_array.sv (R x C systolic), tm_vpu.sv (row-engine lanes),
                tm_xpu.sv + tm_vdly.sv (v0.4 extension unit), tm_core.sv
sim/            tb_core.sv (DDR model with random backpressure and latency), tb_fp.sv, dram/ (LPDDR4X via DRAMsim3, DPI)
examples/       mlp_digits.py (trained MLP -> TARS-mini)
tests/          748 tests: contract, numerics, golden, compiler, MLP, v0.4 ops + Qwen blocks (golden vs transformers), RTL fp units, RTL core, RTL v0.4
tools/          mutate_rtl.py (inject bugs into the RTL; every one must be caught), perf_report.py, perf_p5.py,
                perf_dram.py, fetch_dramsim3.sh, timing/ (ASAP7 flows: units, stages, whole core; sta.py)
tars_mini_ref.py  the P0 reference script (v0.1), kept as is
```

## Run

```sh
pip install numpy scikit-learn pytest      # and Verilator >= 5 for the RTL tests
python3 -m pytest -q                       # 748 tests, ~9 h on 2 cores from scratch (each array configuration is a ~10 min Verilator build; cached after)
python3 examples/mlp_digits.py             # P1 report
python3 tools/mutate_rtl.py                # mutation check (141 mutants; a fast tier, then each one's own test subset), ~1 day on 2 cores; --only 'a|b' reruns some
sh tools/fetch_dramsim3.sh && python3 tools/perf_dram.py   # cycles on LPDDR4X-4266 vs the ideal port
python3 tools/perf_qwen.py                 # Qwen3.5-0.8B: one decode token (LA, FA, norm, a head block) on the RTL under LPDDR4X, checked vs golden
python3 tools/run_qwen_full.py golden|ref|rtl [--dram]   # the whole 0.8B model: greedy decoding on golden, vs transformers, one step on the RTL (~4 h)
sh tools/timing/run_core_h.sh <asap7.lib> <sv2v>           # every core path vs the 650 ps budget (Yosys, ABC, sta.py)
python3 tools/perf_report.py --baseline docs/p3_perf.json   # cycles vs roofline and vs P2
```

## Lessons from P2

- Random vectors almost never hit a rounding tie (about 1 in 2^23 for a product). A "round half up"
  multiplier passed 200 k random vectors; constructed tie vectors caught it at once.
- A coverage assertion on the fuzz (error codes hit, NaN outputs, cross-descriptor hazards) found
  that the first generator produced no NaN at all and almost no hazards.
- Two of the first mutants were equivalent (no behaviour change), and one directed test missed
  its target because the read cache holds a single 16-byte beat. Hazard tests have to be built on
  the microarchitecture's actual state.
- The array drain is not needed in this core (the element-serial epilogue is slower than the
  array's last products); it stays for when the epilogue becomes parallel.

## Lessons from A3

- A systolic array cannot be stalled at its edges alone: a and w meet in PE (i, j) only if every
  edge is delayed by the same amount at the same virtual step. The stall has to freeze the whole
  array (one enable per PE register).
- The old fuzz (K <= 64, 4 chunks) never wrapped the 4 chunk buffers, so slot-reuse bugs were
  invisible to it. The new tests shrink NB and OT and lengthen K until every slot and every
  credit is reused; the reuse mutants were caught only by those tests.
- The projection was right where its assumptions held (decode within 2-7 %) and optimistic where
  it had folded in an unbuilt part (epilogue read latency). Write down what each projection
  assumes, so a shortfall points at the missing piece.

## Lessons from B and reuse

- Removing a stage needs a timing argument, not a test run: the drain was unnecessary once rows
  are read in order (row i is final at step K+i+C-1), and the argument says exactly when that
  would stop being true (an epilogue that reads rows out of order or faster than one per step).
- A mutant that used to be caught can become equivalent after a refactor: once GEMM had its own
  writer, the RELU packer only ever wrote whole beats, so its strobes no longer mattered.
- Projections undercount changes that remove work (the drain) and overlap phases (the writer under
  the next tile): B was projected at 1.06-6.4x and measured 1.1-10.9x.
- Reuse needs a coverage assertion of its own: fuzz shapes decide the loop order and whether K
  fits, so the test asserts that held tiles occur in both orders and that too-large GEMMs occur.

## Lessons from P5

- Write the transcendental functions as fixed fmul/fadd sequences in the golden model first, and
  measure them: two first drafts failed (recip(1) = 0.99999994 broke GELU's tail by 13 orders of
  magnitude; rsqrt underflowed to NaN near 2^-126). Normalizing the mantissa before Newton steps,
  and correction-form steps, fixed both before any RTL existed.
- A coverage assertion can fail because the generator changed, not because the design did. Measure
  coverage on the programs the fuzz actually runs, and add a configuration that produces the
  missing case (a 3x16 array for m-inner reuse) instead of lowering the threshold.
- The fuzz caught an edit whose `//` comment swallowed two latches on the same line. Whole-program
  random queues find what targeted tests and mutants were not looking for.
- A roofline must model the machine's channels: the DDR model and AXI read and write in parallel,
  and a one-port model showed an activation running at 145 % of its "roofline".
- Mutation runs need per-mutant test subsets and build cleanup once the core is large: 48 mutants
  times a dozen builds filled the session's disk once.
- The mutants found five test holes the fuzz could not: random-byte data makes sums insensitive to
  the reduction order the contract fixes, error tests can be rejected by an earlier rule than the
  one they target, and some behaviour shows only on the last descriptor under write backpressure or
  on all-negative rows. Directed tests on well-conditioned data closed all five.
- Splitting one signal into a feed copy and a loader copy (for prefetch) silently weakened two
  mutants that used to break both: each now breaks only one side, and one side's damage is a race
  or extra traffic. A rerun of the full mutation set after a structural change is what found it;
  the fixes were a directed race test and a traffic bound.

## Lessons from P8

- A DRAM model changes what is worth optimizing: the loader's row interleave was harmless on the
  ideal port and cost 2.7x read amplification on LPDDR (one 64 B burst per 16 B used); four
  consecutive chunks per row visit fixed it. Little's law sized the reads in flight (100 ns x
  17 GB/s ~ 1.7 KB, so 128 beats, not 32).
- ABC's area recovery makes every path that is not the worst look as slow as the worst: ~100
  "violations" in the first core report were artifacts (i2f 893 ps in the core, 601 ps alone).
  Map without area recovery and time every endpoint (tools/timing/sta.py) before redesigning.
- The fixes that closed timing were mostly bookkeeping, not arithmetic: counters kept beside the
  indices they replace compares of (left-to-issue, rows-in-tile, the queue address), flags
  registered one step ahead (the feed window), a candidate register between choosing a read and
  issuing it, and the descriptor check's products in a sequencer hidden under the CRC.
- A non-stalling pipeline needs its output buffer sized for the lane latency: 64 slots for a
  122-cycle GELU lane halved the activation's throughput; the DRAM run showed it as a 17 %
  regression that the ideal-port numbers alone would have blamed on the memory.
- 1 GHz costs the row operators 1.3-1.45x on an ideal port (a 3-cycle fadd makes the contract's
  ordered 16-lane sum one step per 3 cycles, and each pass ends with a ~100-cycle fold and
  scalar) and at most 2 % on LPDDR4X, which the memory hides.
- Pipelines make equivalent mutants: a delay line off by a few cycles changes nothing when the
  value it carries is constant over the gap (a tile's bias, a row's 1/sum). Five such mutants were
  rewritten to break what the pipeline must keep apart (columns, slots) instead; one mutant that
  survived honestly found the lane spacing a cycle slower than needed, a free 3-6 % on row ops.
- A mutation run on 2 cores needs triage: a fast tier (one 8x8 fuzz build) before each mutant's
  full subset, a cycle cap so a mutant that hangs the core fails in seconds, and ccache.

## Lessons from v0.4 (Qwen3.5 LA + FA blocks)

- Decide the numerics against the real reference before any RTL: the golden model's lowering
  was checked against transformers with the same W8A8 linears (block updates within 1.2e-6) before
  a line of tm_xpu existed, so every later RTL difference was an RTL bug, not a modelling question.
  The first softplus (log of 1 + e) was 17 % off near 0; ln(1 + t) as 2 atanh(t / (t + 2)) fixed it.
- A tagged read stream keeps a new unit small: the reader tags each beat with its operand and, on
  a step's last beat, the step's meaning; the consumer only collects and fires, with no second
  walker to keep in step with the first.
- Fields that default to 0 bite: a register that only the causal-softmax path set (the valid
  length) left the overlapped norm path masking every element (all outputs inf) until it got a
  dispatch default. And three tests failed for the right reason: they asserted that v0.3 codes
  were illegal (ROW CTRL 3, activations 5 and 7 in a test bench that dropped act early).
- Simulation and synthesis cost grew faster than the logic: per-lane 256-entry delay lines and
  128-deep tag FIFOs made each Verilator build ~10 min and Yosys run out of memory beside a build;
  the delay lines are now a separate two-port memory (an SRAM in a real design, a black box in the
  timing flow).
- The end-to-end number is GEMV-bound: one 0.8B decode token is ~7.9 M cycles at 8x8, mostly the
  M = 1 linears on one array row. That, not the new operators, is what to fix next for decode.
- A GEMV is not a GEMM with M = 1: an output-stationary array then uses one row and streams the
  weights at C bytes a cycle. A dedicated dot-product path at the port's 16 bytes a cycle needed no
  contract change (the INT32 sum is exact in any order) and made decode port-bound (99 % of cycles
  are read beats with an ideal port).
- Full-model runs are a memory problem before a time problem: a 1 GiB image held three times by
  the harness (input, output, expected) plus a Verilator build exceeds 6.5 GB. The harness now
  streams the image from a file and the comparison is chunked; builds and long simulations run apart.
- Field widths shape the memory map: a 24-bit row-scale offset (HDR_RSV) puts state and scratch
  below the 1 GiB of weights, a 16-bit N splits the 248320-row head into four GEMVs.
- Mutation found the one real RTL bug the tests had missed: the extension unit reserved its
  output-queue slack by subtracting 3 from the queue size, which wraps for queues under 4 slots
  and was only ever tested at 128. Counting the launch already under way is exact at any size.
