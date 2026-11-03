# A3: chunk streaming (reads in flight, chunk buffers, continuous feed)

P3 chose memory-level parallelism first. This is that change, measured.

## What changed (microarchitecture only; contract, golden model and tests' expectations unchanged)

- **Loader.** A/W K-chunks are fetched by a loader that runs beside the FSM: one 16-byte read per
  cycle, up to `OT` (default 32) in flight, slots issued A0 W0 A1 W1 ... so they free up in the
  order the skewed feed releases them. The read port now returns data in request order with any
  number in flight; the DDR model gives each read its own random 1..lat latency.
- **Chunk buffers.** `NB` (default 4) buffers of R A-rows and C W-rows. Each slot has a full and a
  pending bit; a slot is reused for chunk c + NB only after its row has taken the last byte of
  chunk c, and its previous read has returned.
- **Continuous feed.** One feed per output tile: row i takes k = step - i for k = 0..K-1, across
  chunks, so the skew fills once per tile instead of once per chunk. The feed advances only when
  every live row and column has its current chunk; otherwise an array-wide enable freezes every
  PE (`tm_pe.en`), so a stall delays all edges alike and the skew alignment survives. (Pausing
  only the edges would misalign a and w inside the array.)
- **Not changed:** the descriptor fetch and the epilogue's scale/bias/RELU reads still go one at a
  time with the full latency; the epilogue is still one element per cycle. Both belong to B.

## Verification

| Gate | Result |
|---|---|
| full test suite (P2 gates + new) | 387 passed |
| new: loader fuzz | 6 (NB, OT, array) configs incl. NB=2/OT=2, K up to 192 (12 chunks, every slot reused), latency 1-16, stalls 0-80 %, 4 seeds each: bit-exact |
| new: reads in flight | measured max in flight <= OT always, and >= min(OT, latency) |
| mutation check | 16 / 16 caught (5 new: slot freed one byte early, feed not waiting for W, slot reissued while pending, credit limit + 1, array not frozen on stall) |
| the P1 MLP on RTL | bit-exact, accuracy unchanged |

## Performance (no backpressure; baseline = P2 core, `docs/p3_perf.json`)

| workload | array | lat | baseline cycles | cycles | measured speedup | projected A3 | % of roof | load wait | feed | epilogue | other | max reads in flight | then B |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_l1 | 8x8 | 2 | 525,641 | 260,023 | 2.02x | 3.10x | 17.7% | 4% | 20% | 69% | 8% | 2 | 2.60x |
| mlp_l1 | 8x8 | 16 | 1,010,331 | 431,137 | 2.34x | 5.97x | 10.7% | 5% | 12% | 79% | 5% | 16 | 3.31x |
| mlp_l1 | 16x16 | 2 | 339,265 | 213,380 | 1.59x | 2.58x | 5.4% | 6% | 7% | 83% | 4% | 2 | 4.61x |
| mlp_l1 | 16x16 | 16 | 663,926 | 377,543 | 1.76x | 5.04x | 3.1% | 4% | 4% | 90% | 3% | 16 | 6.37x |
| mlp_l2 | 8x8 | 2 | 107,990 | 39,789 | 2.71x | 3.80x | 29.0% | 3% | 31% | 58% | 8% | 2 | 2.09x |
| mlp_l2 | 8x8 | 16 | 207,067 | 60,927 | 3.40x | 7.27x | 18.9% | 4% | 20% | 70% | 6% | 16 | 2.67x |
| mlp_l2 | 16x16 | 2 | 63,768 | 31,841 | 2.00x | 2.95x | 14.0% | 9% | 10% | 73% | 7% | 2 | 3.23x |
| mlp_l2 | 16x16 | 16 | 124,603 | 52,200 | 2.39x | 5.76x | 8.5% | 6% | 6% | 83% | 5% | 16 | 4.54x |
| decode | 8x8 | 2 | 681,625 | 146,763 | 4.64x | 4.73x | 45.1% | 1% | 90% | 8% | 2% | 2 | 1.06x |
| decode | 8x8 | 16 | 1,201,026 | 151,999 | 7.90x | 8.33x | 43.6% | 2% | 87% | 10% | 2% | 16 | 1.06x |
| decode | 16x16 | 2 | 601,392 | 92,831 | 6.48x | 6.63x | 71.4% | 5% | 72% | 21% | 3% | 2 | 1.22x |
| decode | 16x16 | 16 | 1,094,942 | 97,266 | 11.26x | 12.07x | 68.1% | 6% | 68% | 23% | 3% | 16 | 1.21x |
| prefill | 8x8 | 2 | 534,137 | 141,632 | 3.77x | 4.88x | 46.3% | 3% | 48% | 45% | 5% | 2 | 1.67x |
| prefill | 8x8 | 16 | 1,050,002 | 202,845 | 5.18x | 9.59x | 32.3% | 4% | 33% | 60% | 4% | 16 | 2.12x |
| prefill | 16x16 | 2 | 281,958 | 102,051 | 2.76x | 3.86x | 16.1% | 16% | 17% | 62% | 4% | 2 | 2.44x |
| prefill | 16x16 | 16 | 568,582 | 160,350 | 3.55x | 7.79x | 10.2% | 11% | 11% | 75% | 3% | 16 | 3.46x |
| square | 8x8 | 2 | 1,922,671 | 338,274 | 5.68x | 6.28x | 77.5% | 1% | 78% | 19% | 2% | 2 | 1.20x |
| square | 8x8 | 16 | 3,815,437 | 399,339 | 9.55x | 12.47x | 65.6% | 2% | 66% | 30% | 2% | 16 | 1.37x |
| square | 16x16 | 2 | 922,221 | 198,361 | 4.65x | 5.45x | 33.0% | 33% | 34% | 32% | 2% | 2 | 1.43x |
| square | 16x16 | 16 | 1,896,030 | 256,338 | 7.40x | 11.20x | 25.6% | 26% | 26% | 47% | 1% | 16 | 1.78x |

*load wait* = cycles the feed waits for a chunk (plus the tile set-up); *then B* = projected
further speedup of a parallel epilogue on top of this core (estimate, from these counters).
Reproduce: `python3 tools/perf_report.py --baseline docs/p3_perf.json`.

MLP (whole queue, latency 2): 8x8 633,610 -> 299,845 cycles (2.1x); 16x16 403,000 -> 245,198 (1.6x).

## Reading

1. **Loads are no longer the bottleneck.** Load wait fell from 38-87 % of cycles to 1-9 % on
   most runs. The exception is 16x16 on long K (square 26-33 %, prefill 11-16 %): there each
   16-cycle chunk needs R + C = 32 beats on a 16-byte port, the 16 / (R + C) = 50 % ceiling P3
   predicted. Only on-chip reuse (A or W held across tiles) moves that.
2. **Measured speedup 1.6-11.3x vs projected 2.6-12.5x.** Where loads dominated (decode, square
   8x8) measurement is within 2-23 % of the projection (decode 2-7 %). The short-K layers fall further short
   (mlp_l1 16x16: 1.6x vs 2.6x) because the A3 projection also assumed the epilogue's scale/bias
   reads would hide their latency, and they still do not.
3. **The epilogue is now the largest phase in 12 of 20 runs** (up to 90 % on mlp_l1 16x16). B
   on top of this core is projected at 1.06-6.4x: largest on short-K and 16x16, about 1.06x on
   decode 8x8.
4. Decode reaches 44-71 % of its memory roofline; square 8x8 66-78 %.

## Next

B (parallel epilogue, scale/bias read once per tile and streamed) is now the largest lever for
everything except decode; for decode and long-K 16x16, on-chip reuse is. Both are
microarchitecture; the same gates apply.
