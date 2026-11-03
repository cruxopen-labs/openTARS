# P8: LPDDR4X timing model and a 1 GHz pipeline (ASAP7)

No contract change: every result bit is the golden model's, as before. What changed is how
the cycles are counted (a DRAM model instead of an ideal port) and what one cycle may contain
(at most one ~650 ps stage of logic, npu_clk 1 GHz per spec V1.1).

## 1. DRAM model (sim/dram)

| Part | Choice |
|---|---|
| DRAM | LPDDR4X-4266, x32 (two x16 dies ganged into one 32-bit channel), JESD209-4B timings (tRCD/tRP 18 ns, tRAS 42 ns, RL 36 / WL 18, BL16 -> 64 B per access, tRFCab 280 ns, tREFI 3.9 us) |
| Simulator | DRAMsim3 (pinned commit, built by tools/fetch_dramsim3.sh), open-page, per-bank queues |
| Front end (tb_dram.cpp, DPI) | AXI-like: in-order returns, reads to an in-flight 64 B burst merged, a 4-line read buffer, posted writes combined in an 8-line buffer (full / evicted / idle 64 cycles), NoC + controller + PHY latency `fabric` = 40 ns |
| Data | stays in the testbench: a read returns memory as it is when it returns, a write lands when accepted (as with the ideal port), so the model decides timing only |

Approximations to keep in mind: LPDDR4 x32 is really two independent x16 channels with 32 B
bursts; no system-level cache; the fabric latency is a parameter, not a measurement.

**What the model exposed, and the core changes it led to**

| Finding | Change |
|---|---|
| ~100 ns average read latency; 32 beats in flight cap the core at ~5 GB/s (Little) | 128 reads in flight (loader OT and row reader VQ) |
| the loader fetched 16 B of one row, then of the next row: one 64 B burst per 16 B used (amplification 2.7x on GEMM) | the loader takes LG = 4 consecutive chunks of a row per visit (one burst), with NB = 8 chunk buffers |
| GEMM 16x16 is then DRAM-bound: burst use 100 %, but the other operand is re-streamed per tile | (architecture: more on-chip reuse) |
| implicit-GEMM conv re-reads each input pixel up to 9 times from DRAM | (architecture: load the input tile on chip once, as TARS's VPU im2col from VMEM does) |
| LPDDR's data bus is half duplex | the DRAM roofline charges reads + writes against 17.06 GB/s |

## 2. Timing flow (tools/timing)

Pre-layout logic delay on ASAP7 RVT TT (open 7 nm predictive library). Budget: 650 ps of logic
per 1 ns cycle (the rest for clock-to-q, setup, skew and wires).

- **FP units** (run_units.sh, run_stages.sh): sv2v -> Yosys >= 0.38 (adders as Kogge-Stone, Booth
  multipliers) -> ABC (dch, delay mapping, buffering to fanout 4, sizing) -> `stime`. Calibration:
  a 32-bit adder 167 ps, a 24 x 24 Booth multiplier 578 ps, one gate 15 ps.
- **Whole core** (run_core_h.sh + sta.py): Yosys keeps the hierarchy (PEs, row-engine lanes and
  fold, delay lines), ABC maps one module at a time (dch, `map -r`, buffering, `upsize -W 50`), the
  fp units are black boxes. ABC's `stime` reports only the worst path, so `sta.py` times the
  flattened mapped netlist itself: NLDM tables (rise and fall apart, by unateness, slew
  propagated), every register group's worst arrival; a path into a black-box unit is charged the
  unit's first-stage delay; a net that drives more than 4 fF gets an ideal fanout-4 buffer tree
  (20 ps a level), since ABC cannot buffer a net past its boundary (register enables). sta.py
  reads 12-17 % above ABC's own `stime` on the same netlist (checked on i2f, an adder, bf16).

Lessons from building the flow:

- ABC's area recovery relaxes every path that is not the worst up to the worst path's delay: with
  it, the first core runs showed ~100 register groups "over budget" whose logic was fine (i2f at
  893 ps in the core, 601 ps alone). `map -r` (no area recovery) plus upsizing near-critical paths
  gives per-path numbers that mean something. Likewise `dc2` / `rewrite` turn a prefix adder back
  into a 590 ps chain, and `map -D` stops at its target.
- The flattened core does not fit dch in 4-5 GB; per-module mapping does (at R = C = 2). Larger
  configurations exceed the session's memory in Yosys itself.
- Open-source sizing is weaker than a commercial flow, so stage counts are upper bounds.

## 3. FP units

| Unit | Before (one cycle) | After |
|---|---|---|
| fadd | 1949 ps | rewritten (log-tree normalize, shifted-ones masks, narrow exponent math): 3 stages, 483 / 605 / 218 ps |
| fmul | 1689 ps | 3 stages: radix-4 partial products + 3:2 tree (586 ps), final add + normalize (241), round (468) |
| i2f | 657 ps | 619 ps (log-tree normalize); 2 stages in the GEMM epilogue (sign / magnitude, then normalize and round) |
| rint, set_exp, bf16, q8 tail, fmax | | 1 stage each: 402 / 214 / 178 / 418 / 346 ps |
| exp | 40 ns | 51 cycles |
| recip / rsqrt | | 38 / 56 cycles |
| GELU / SiLU (shared exp, recip) | | 122 cycles |

Each pipelined unit equals its function bit for bit (tests/test_rtl_fp.py streams every fp vector
through it). Core simulation uses cycle-identical behavioral twins (function + L-cycle delay;
the units never stall); one core test runs the structural units as a cross-check.

## 4. Core changes for 1 GHz

| Path (first core run) | Change |
|---|---|
| GEMM epilogue: i2f, fmul, fadd, act, conversion in one cycle | C pipelined lanes (9 to 134 cycles) into a 128-slot output queue (slot taken at launch, so the lanes never stall) |
| row engine: lane sums, folds, scalars in one cycle | steps carry their own state through 4 VG pipelined lanes; 16 lane accumulators with 3-cycle fadds (result forwarded), a lane takes the next summed step 3 cycles after the last; passes end with a drain and a pipelined fold + scalar (up to 74 cycles); stage A has its own accumulators and fold; the contribution tap is registered |
| descriptor check and dispatch: 32 x 32 products, divisions, 4-input 66-bit sums, a 14-deep rule chain | a check sequencer beside the 64-cycle CRC: five 17-step dividers (K / (KH KW), ceil(M / R), ceil(N / C), the output size), one shift-add multiplier (4 bits a cycle, 13 products), then the range ends and every dispatch value registered; the rules evaluated in parallel with a priority pick |
| array enable: per-row K-window compares and slot lookups feeding every PE register | the window flags, chunk slots and row liveness registered as the feed steps; feed_go is an AND of registered terms |
| loader: live-position scan, slot lookup, padding test and address in front of the read port | two stages: a generate stage puts the next read (address, slot, padding flag) in a candidate register, the issue stage sends it once its slot is free; live mask and next tile registered per tile |
| PE: 8 x 8 multiply + 32-bit accumulate | product registered with its flags (the epilogue waits one more array step) |
| array edges: byte selects and K-range compares before the PE | registered edges (one more array step for every row and column alike) |
| no runtime multiply / divide outside the check | loader row-address tables and a conv chunk walker (adds only), running epilogue / row / queue addresses, row-engine counters (left-to-issue, left-to-consume) beside their indices |

## 5. Results

**Timing** (run_core_h.sh, R = C = 2, VG = 1; fp units as black boxes with their measured first
stages): every register-to-register path within 650 ps; the worst is the row engine's lane
accumulator update (650 ps), then the epilogue's running address (631 ps) and the loader's chunk
walker (625 ps). The FP units themselves: every stage within 650 ps (section 3).

**Cycles under LPDDR4X-4266** (perf_dram.py; `docs/dram_p8.md`, P8a = DRAM model before the 1 GHz
pipeline, `docs/dram_p8a.md`):

| workload | array | ideal port P8a -> P8 | LPDDR4X P8a -> P8 | change | % of DRAM roofline |
|---|---|---:|---:|---:|---:|
| softmax | any | 16,785 -> 24,148 | 30,240 -> 30,883 | +2.1 % | 74.6 % |
| layernorm | any | 17,335 -> 23,171 | 30,857 -> 30,802 | -0.2 % | 76.4 % |
| rmsnorm | any | 16,995 -> 22,610 | 30,784 -> 31,437 | +2.1 % | 74.1 % |
| gelu (eltwise) | any | 17,854 -> 17,906 | 31,291 -> 30,930 | -1.2 % | 74.5 % |
| gemm + relu | 8x8 | 69,116 -> 69,185 | 89,532 -> 89,900 | +0.4 % | 72.9 % |
| gemm + relu | 16x16 | 19,064 -> 19,553 | 30,552 -> 30,535 | -0.1 % | 53.7 % |
| gemm + gelu | 8x8 | 69,116 -> 69,306 | 89,532 -> 91,719 | +2.4 % | 71.5 % |
| gemm + gelu | 16x16 | 19,064 -> 19,674 | 30,552 -> 30,656 | +0.3 % | 53.4 % |
| conv 3x3 | 8x8 | 117,648 -> 117,744 | 140,741 -> 140,650 | -0.1 % | 80.3 % |
| conv 3x3 | 16x16 | 30,926 -> 32,848 | 41,785 -> 42,675 | +2.1 % | 66.1 % |

Under LPDDR4X the 1 GHz pipeline costs -1 to +2.4 %: the memory hides the longer latencies. With an
ideal port the row operators are 1.3-1.45x slower than before: a 16-lane sum in the contract's order
can take a step only every 3 cycles (the fadd's latency, one dependent add per lane), and each pass
ends with a drain and the fold + scalar latency (up to ~110 cycles per pass). The first run used a
64-slot output queue and GELU ran at half rate (a slot is held for the 122-cycle lane); 128 slots
restored it. The mutation check found the lane spacing one cycle more conservative than the
forwarding needs (a step every 4 cycles where 3 is exact); tightening it took softmax from +5.0 %
and LayerNorm from +5.6 % to the numbers above.

**Verification**: 547 tests (RTL == golden byte for byte on every DDR byte and the end status,
ideal port with random stalls and the DRAM model, the structural fp units in the core once);
mutation check 101 / 101 injected RTL bugs caught, 62 of them new or rewritten for the P8
structures. Five first-draft mutants were equivalent and are recorded in tools/mutate_rtl.py with
the reason (a lane step at the forwarding's exact spacing, a bias or scale delay off by a few
cycles when every row of a tile shares them, the softmax r tap, a waiting loader candidate, the
check's wait for its sequencer). One survivor was a real test hole: a tile handed to a busy
epilogue engine (now covered by a 2-slot output queue under heavy write backpressure).

## 6. What the numbers do and do not say

- Cycle counts are exact for this RTL against this DRAM model (DRAMsim3 timing, the front end in
  tb_dram.cpp, a fixed 40 ns fabric). They are not a system measurement.
- Timing is pre-layout logic delay at one small configuration: no wires, placement or clock tree;
  open-source mapping and sizing. Larger R, C, VG widen the muxes and fanouts (the array enable,
  the hold-buffer read, the lane mapping); the flow ran out of memory at R = C = 4, VG = 4.
- The hold buffer (16 KB) and the output queue (128 x RB bytes) are flip-flops in this RTL; a
  real design would use SRAM macros with a registered read, which changes those paths' timing and
  adds a cycle where they are read.
- Area grew: every row-engine slot has its own exp and GELU/SiLU pipelines (4 VG of each), the
  epilogue has C GELU/SiLU pipelines; sharing them between slots would trade throughput for area.
- Stage counts are conservative (upper bounds from an open flow); a commercial flow would likely
  merge some (fmul and fadd in 2 stages each, exp / recip / rsqrt shorter).
