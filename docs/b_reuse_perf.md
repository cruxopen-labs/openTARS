# B (parallel epilogue) and on-chip reuse

Both steps after A3, measured. Microarchitecture only: the contract, the golden model and every
test's expectation are unchanged.

## Where it stands (P2 -> A3 -> B -> reuse; no backpressure)

| workload | array | lat | P2 cycles | now | total speedup | % of roof P2 -> now |
|---|---|---|---:|---:|---:|---|
| mlp_l1 | 8x8 | 2 | 525,641 | 62,918 | 8.4x | 8.8% -> 73.2% |
| mlp_l1 | 8x8 | 16 | 1,010,331 | 72,140 | 14.0x | 4.6% -> 63.9% |
| mlp_l1 | 16x16 | 2 | 339,265 | 19,555 | 17.3x | 3.4% -> 58.9% |
| mlp_l1 | 16x16 | 16 | 663,926 | 22,156 | 30.0x | 1.7% -> 52.0% |
| mlp_l2 | 8x8 | 2 | 107,990 | 14,334 | 7.5x | 10.7% -> 80.4% |
| mlp_l2 | 8x8 | 16 | 207,067 | 15,494 | 13.4x | 5.6% -> 74.4% |
| mlp_l2 | 16x16 | 2 | 63,768 | 5,073 | 12.6x | 7.0% -> 87.9% |
| mlp_l2 | 16x16 | 16 | 124,603 | 5,429 | 23.0x | 3.6% -> 82.2% |
| decode | 8x8 | 2 | 681,625 | 133,724 | 5.1x | 9.7% -> 49.5% |
| decode | 8x8 | 16 | 1,201,026 | 135,457 | 8.9x | 5.5% -> 48.9% |
| decode | 16x16 | 2 | 601,392 | 67,739 | 8.9x | 11.0% -> 97.8% |
| decode | 16x16 | 16 | 1,094,942 | 68,669 | 15.9x | 6.1% -> 96.5% |
| prefill | 8x8 | 2 | 534,137 | 72,603 | 7.4x | 12.3% -> 90.3% |
| prefill | 8x8 | 16 | 1,050,002 | 76,028 | 13.8x | 6.2% -> 86.2% |
| prefill | 16x16 | 2 | 281,958 | 21,467 | 13.1x | 5.8% -> 76.3% |
| prefill | 16x16 | 16 | 568,582 | 22,397 | 25.4x | 2.9% -> 73.2% |
| square | 8x8 | 2 | 1,922,671 | 269,275 | 7.1x | 13.6% -> 97.4% |
| square | 8x8 | 16 | 3,815,437 | 272,698 | 14.0x | 6.9% -> 96.1% |
| square | 16x16 | 2 | 922,221 | 76,827 | 12.0x | 7.1% -> 85.3% |
| square | 16x16 | 16 | 1,896,030 | 77,757 | 24.4x | 3.5% -> 84.3% |

The P1 MLP, whole queue, latency 2: 8x8 633,610 -> 77,252 cycles (8.2x); 16x16 403,000 -> 24,626
(16.4x).

## B: parallel epilogue

- **Row-parallel epilogue.** C lanes, each the contract's steps 2-6 in order (i2f, fmul, fadd,
  act, convert; each rounded once, no FMA); one tile row per cycle.
- **Scale/bias once per column block.** The loader fetches the tile's scale and bias beats first
  (they are few), into vector registers; when the next tile has the same n0 they are kept.
- **Segment writer.** A row's C x es bytes, at any byte offset, become byte-strobed beats (one
  per accepted cycle); it takes the next row the cycle its last beat is accepted and runs on into
  the next tile's feed. A descriptor ends only after its last row is written.
- **No drain.** PE (i, j) adds a[i][K-1] at step K-1+i+j, so row i is final from step K+i+C-1;
  the epilogue reads row i no earlier than step K+max(R,C)-1+i. The R+C+2-cycle drain per tile
  is gone.

Beat-level safety: inputs and outputs are disjoint byte ranges (golden rejects overlap), Y rows
are whole aligned beats, so no output beat shares a beat with an input byte; prefetching scale
and bias, and writing while the next tile loads, cannot change a result.

| workload | array | lat | baseline cycles | cycles | measured speedup | projected B | % of roof | load wait | feed | epilogue | other | traffic / min | max reads in flight |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_l1 | 8x8 | 2 | 260,023 | 71,371 | 3.64x | 2.60x | 64.6% | 18% | 72% | 9% | 1% | 11.2x | 2 |
| mlp_l1 | 8x8 | 16 | 431,137 | 81,473 | 5.29x | 3.31x | 56.6% | 28% | 63% | 8% | 1% | 11.2x | 16 |
| mlp_l1 | 16x16 | 2 | 213,380 | 31,971 | 6.67x | 4.61x | 36.0% | 44% | 45% | 10% | 1% | 5.6x | 2 |
| mlp_l1 | 16x16 | 16 | 377,543 | 34,581 | 10.92x | 6.37x | 33.3% | 48% | 42% | 9% | 1% | 5.6x | 16 |
| mlp_l2 | 8x8 | 2 | 39,789 | 15,391 | 2.59x | 2.09x | 74.8% | 11% | 79% | 9% | 1% | 3.0x | 2 |
| mlp_l2 | 8x8 | 16 | 60,927 | 16,685 | 3.65x | 2.67x | 69.0% | 17% | 73% | 9% | 1% | 3.0x | 16 |
| mlp_l2 | 16x16 | 2 | 31,841 | 7,825 | 4.07x | 3.23x | 57.0% | 39% | 42% | 18% | 1% | 1.6x | 2 |
| mlp_l2 | 16x16 | 16 | 52,200 | 8,181 | 6.38x | 4.54x | 54.5% | 41% | 40% | 17% | 2% | 1.6x | 16 |
| decode | 8x8 | 2 | 146,763 | 133,851 | 1.10x | 1.06x | 49.5% | 1% | 99% | 0% | 0% | 1.1x | 2 |
| decode | 8x8 | 16 | 151,999 | 135,592 | 1.12x | 1.06x | 48.9% | 2% | 97% | 0% | 0% | 1.1x | 16 |
| decode | 16x16 | 2 | 92,831 | 71,771 | 1.29x | 1.22x | 92.3% | 7% | 93% | 0% | 0% | 1.1x | 2 |
| decode | 16x16 | 16 | 97,266 | 72,701 | 1.34x | 1.21x | 91.1% | 8% | 91% | 0% | 0% | 1.1x | 16 |
| prefill | 8x8 | 2 | 141,632 | 74,587 | 1.90x | 1.67x | 87.9% | 6% | 90% | 3% | 0% | 9.4x | 2 |
| prefill | 8x8 | 16 | 202,845 | 78,205 | 2.59x | 2.12x | 83.8% | 10% | 86% | 3% | 0% | 9.4x | 16 |
| prefill | 16x16 | 2 | 102,051 | 36,827 | 2.77x | 2.44x | 44.5% | 47% | 47% | 6% | 0% | 4.8x | 2 |
| prefill | 16x16 | 16 | 160,350 | 37,757 | 4.25x | 3.46x | 43.4% | 48% | 46% | 5% | 1% | 4.8x | 16 |
| square | 8x8 | 2 | 338,274 | 271,195 | 1.25x | 1.20x | 96.7% | 2% | 97% | 1% | 0% | 15.2x | 2 |
| square | 8x8 | 16 | 399,339 | 274,813 | 1.45x | 1.37x | 95.4% | 3% | 96% | 1% | 0% | 15.2x | 16 |
| square | 16x16 | 2 | 198,361 | 134,171 | 1.48x | 1.43x | 48.8% | 50% | 50% | 1% | 0% | 7.6x | 2 |
| square | 16x16 | 16 | 256,338 | 135,101 | 1.90x | 1.78x | 48.5% | 50% | 49% | 1% | 0% | 7.6x | 16 |

Measured 1.10-10.9x over A3, above the 1.06-6.4x projection: the projection had neither the
drain removal nor the writer running into the next tile.

## On-chip reuse: a hold buffer across the inner tile loop

- **Hold buffer** of HB beats (default 1024 = 16 KB; a VMEM region in TARS terms). One operand's
  rows for the current outer step, one beat per K-chunk.
- **Loop order per descriptor.** n-inner (hold the A rows of m0) or m-inner (hold the W rows of
  n0), whichever has fewer outer steps (ceil(M/R) vs ceil(N/C)). The first tile of each outer step
  loads the operand from DDR and copies it into the buffer; the others read it there and stream
  only the other operand: C (or R) beats per 16-cycle chunk instead of R + C, inside the port's
  16 beats for arrays up to 16 wide.
- **Fallback.** K / 16 > HB / max(R, C): every tile loads both, as before.

| workload | array | lat | baseline cycles | cycles | measured speedup | % of roof | load wait | feed | epilogue | other | traffic / min | max reads in flight |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_l1 | 8x8 | 2 | 71,371 | 62,918 | 1.13x | 73.2% | 7% | 81% | 10% | 1% | 6.0x | 2 |
| mlp_l1 | 8x8 | 16 | 81,473 | 72,140 | 1.13x | 63.9% | 19% | 71% | 9% | 1% | 6.0x | 16 |
| mlp_l1 | 16x16 | 2 | 31,971 | 19,555 | 1.63x | 58.9% | 9% | 74% | 16% | 1% | 3.1x | 2 |
| mlp_l1 | 16x16 | 16 | 34,581 | 22,156 | 1.56x | 52.0% | 19% | 66% | 14% | 1% | 3.1x | 16 |
| mlp_l2 | 8x8 | 2 | 15,391 | 14,334 | 1.07x | 80.4% | 4% | 85% | 10% | 1% | 1.6x | 2 |
| mlp_l2 | 8x8 | 16 | 16,685 | 15,494 | 1.08x | 74.4% | 11% | 78% | 9% | 1% | 1.6x | 16 |
| mlp_l2 | 16x16 | 2 | 7,825 | 5,073 | 1.54x | 87.9% | 5% | 65% | 27% | 2% | 1.0x | 2 |
| mlp_l2 | 16x16 | 16 | 8,181 | 5,429 | 1.51x | 82.2% | 11% | 61% | 26% | 3% | 1.0x | 16 |
| decode | 8x8 | 2 | 133,851 | 133,724 | 1.00x | 49.5% | 1% | 99% | 0% | 0% | 1.0x | 2 |
| decode | 8x8 | 16 | 135,592 | 135,457 | 1.00x | 48.9% | 2% | 97% | 0% | 0% | 1.0x | 16 |
| decode | 16x16 | 2 | 71,771 | 67,739 | 1.06x | 97.8% | 1% | 98% | 0% | 0% | 1.0x | 2 |
| decode | 16x16 | 16 | 72,701 | 68,669 | 1.06x | 96.5% | 3% | 97% | 0% | 0% | 1.0x | 16 |
| prefill | 8x8 | 2 | 74,587 | 72,603 | 1.03x | 90.3% | 4% | 93% | 3% | 0% | 5.1x | 2 |
| prefill | 8x8 | 16 | 78,205 | 76,028 | 1.03x | 86.2% | 8% | 89% | 3% | 1% | 5.1x | 16 |
| prefill | 16x16 | 2 | 36,827 | 21,467 | 1.72x | 76.3% | 9% | 81% | 10% | 1% | 2.7x | 2 |
| prefill | 16x16 | 16 | 37,757 | 22,397 | 1.69x | 73.2% | 13% | 77% | 9% | 1% | 2.7x | 16 |
| square | 8x8 | 2 | 271,195 | 269,275 | 1.01x | 97.4% | 1% | 98% | 1% | 0% | 8.1x | 2 |
| square | 8x8 | 16 | 274,813 | 272,698 | 1.01x | 96.1% | 2% | 97% | 1% | 0% | 8.1x | 16 |
| square | 16x16 | 2 | 134,171 | 76,827 | 1.75x | 85.3% | 12% | 87% | 1% | 0% | 4.3x | 2 |
| square | 16x16 | 16 | 135,101 | 77,757 | 1.74x | 84.3% | 13% | 86% | 1% | 0% | 4.3x | 16 |

16x16 gains 1.5-1.75x (the port-bound shapes); 8x8 gains up to 1.13x (it was not port-bound).

## Verification

| Gate | Result |
|---|---|
| tests | 412 passed (387 before + 25 hold-buffer: 6 HB/array configs x 4 seeds with HB small enough that some GEMMs fit and some do not, and a coverage assertion that both loop orders hold tiles and that GEMMs too big to hold occur) |
| mutants | 25 / 25 caught (B: bias order, writer strobes, writer takes a row early, descriptor ends before its last row, vectors kept across column blocks or descriptors; reuse: held A read while still filling, held W keyed on the wrong index, hold used when K does not fit, wrong operand captured, m-inner loop ends early) |
| equivalent mutant found | RELU ignoring its write strobes: RELU output rows are whole aligned beats, so its strobes are always full. Dropped. |
| the P1 MLP on RTL | bit-exact, accuracy = golden |

## What limits it now

1. **Decode 8x8 (49 %)** sits on its schedule floor (131,072 cycles: one of the 8 rows works when
   M = 1). Only a dataflow change helps: V1.3.1's sub-array partitioning, or a GEMV mode that
   spreads K across rows. Decode 16x16 is at 97 % of its memory roofline.
2. **Short K (mlp_l1 52-73 %, mlp_l2 74-88 %)**: per tile the skew fills (max(R,C)-1 steps) and
   the epilogue runs after the feed (R rows, more when an fp32 row needs 4 beats). Double-buffered
   accumulators would let tile t's epilogue run under tile t+1's feed, and a feed continuing
   across tiles would fill the skew once per outer step.
3. **Traffic is still 2.7-8x the minimum** on prefill/square: one operand is held, the other is
   re-read per outer step. Holding both (an output block of several tiles) is the next reuse
   level, and with a 128x128 array it is the only way to keep the port at 16 / (R + C) or better.
