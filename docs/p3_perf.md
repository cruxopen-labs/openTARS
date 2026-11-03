# P3: where the cycles go, and what to build next

Measured on the P2 RTL with the per-phase counters (tm_core `perf_*`, printed by tb_core; the
phases sum to the total). One GEMM descriptor per run, no backpressure, read latency 2 cycles
(on-chip-like) and 16 cycles (DDR-like). Reproduce: `python3 tools/perf_report.py`.

Machine model for the roofline: R*C int8 MACs per cycle, 16 bytes per cycle on the memory port
(one 128-bit beat). Minimum traffic = A, W, scale, bias and Y once each.

## Results

| workload | array | lat | cycles | roofline | % of roof | load | feed | epilogue | other | traffic / min | traffic floor | schedule floor | A | A2 | A3 | B | AB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_l1 | 8x8 | 2 | 525,641 | 46,080 (compute) | 8.8% | 50% | 13% | 34% | 4% | 15.3x | 74,884 | 46,080 | 2.28x | 2.85x | 3.10x | 1.44x | 6.57x |
| mlp_l1 | 8x8 | 16 | 1,010,331 | 46,080 (compute) | 4.6% | 58% | 7% | 34% | 2% | 15.3x | 74,884 | 46,080 | 4.38x | 5.48x | 5.97x | 1.42x | 12.62x |
| mlp_l1 | 16x16 | 2 | 339,265 | 11,520 (compute) | 3.4% | 38% | 7% | 52% | 3% | 10.0x | 49,220 | 11,776 | 2.24x | 2.63x | 2.58x | 1.97x | 8.47x |
| mlp_l1 | 16x16 | 16 | 663,926 | 11,520 (compute) | 1.7% | 44% | 3% | 51% | 1% | 10.0x | 49,220 | 11,776 | 4.37x | 5.15x | 5.04x | 1.92x | 16.56x |
| mlp_l2 | 8x8 | 2 | 107,990 | 11,520 (compute) | 10.7% | 60% | 15% | 21% | 3% | 3.6x | 15,844 | 11,520 | 2.43x | 3.29x | 3.80x | 1.24x | 6.52x |
| mlp_l2 | 8x8 | 16 | 207,067 | 11,520 (compute) | 5.6% | 70% | 8% | 21% | 2% | 3.6x | 15,844 | 11,520 | 4.66x | 6.30x | 7.27x | 1.22x | 12.48x |
| mlp_l2 | 16x16 | 2 | 63,768 | 4,460 (memory) | 7.0% | 51% | 9% | 36% | 4% | 2.3x | 10,148 | 2,944 | 2.36x | 3.00x | 2.95x | 1.53x | 7.02x |
| mlp_l2 | 16x16 | 16 | 124,603 | 4,460 (memory) | 3.6% | 59% | 5% | 35% | 2% | 2.3x | 10,148 | 2,944 | 4.61x | 5.85x | 5.76x | 1.49x | 13.67x |
| decode | 8x8 | 2 | 681,625 | 66,244 (memory) | 9.7% | 70% | 28% | 2% | 0% | 1.1x | 74,372 | 131,072 | 2.49x | 3.40x | 4.73x | 1.01x | 5.03x |
| decode | 8x8 | 16 | 1,201,026 | 66,244 (memory) | 5.5% | 83% | 16% | 1% | 0% | 1.1x | 74,372 | 131,072 | 4.38x | 5.99x | 8.33x | 1.01x | 8.87x |
| decode | 16x16 | 2 | 601,392 | 66,244 (memory) | 11.0% | 75% | 21% | 3% | 0% | 1.1x | 70,276 | 65,536 | 2.78x | 4.09x | 6.63x | 1.03x | 8.16x |
| decode | 16x16 | 16 | 1,094,942 | 66,244 (memory) | 6.1% | 86% | 12% | 2% | 0% | 1.1x | 70,276 | 65,536 | 5.05x | 7.45x | 12.07x | 1.02x | 14.85x |
| prefill | 8x8 | 2 | 534,137 | 65,536 (compute) | 12.3% | 69% | 18% | 12% | 1% | 10.4x | 75,780 | 65,536 | 2.65x | 3.92x | 4.88x | 1.12x | 6.88x |
| prefill | 8x8 | 16 | 1,050,002 | 65,536 (compute) | 6.2% | 79% | 9% | 11% | 1% | 10.4x | 75,780 | 65,536 | 5.20x | 7.70x | 9.59x | 1.11x | 13.51x |
| prefill | 16x16 | 2 | 281,958 | 16,384 (compute) | 5.8% | 65% | 11% | 23% | 2% | 5.9x | 43,012 | 16,384 | 2.72x | 3.91x | 3.86x | 1.27x | 7.10x |
| prefill | 16x16 | 16 | 568,582 | 16,384 (compute) | 2.9% | 72% | 6% | 21% | 1% | 5.9x | 43,012 | 16,384 | 5.48x | 7.89x | 7.79x | 1.25x | 14.31x |
| square | 8x8 | 2 | 1,922,671 | 262,144 (compute) | 13.6% | 77% | 20% | 3% | 0% | 15.6x | 272,388 | 262,144 | 2.82x | 4.59x | 6.28x | 1.03x | 7.01x |
| square | 8x8 | 16 | 3,815,437 | 262,144 (compute) | 6.9% | 87% | 10% | 3% | 0% | 15.6x | 272,388 | 262,144 | 5.60x | 9.11x | 12.47x | 1.03x | 13.91x |
| square | 16x16 | 2 | 922,221 | 65,536 (compute) | 7.1% | 79% | 14% | 7% | 0% | 8.0x | 140,292 | 65,536 | 3.12x | 5.48x | 5.45x | 1.07x | 6.73x |
| square | 16x16 | 16 | 1,896,030 | 65,536 (compute) | 3.5% | 87% | 7% | 6% | 0% | 8.0x | 140,292 | 65,536 | 6.42x | 11.26x | 11.20x | 1.06x | 13.84x |


Columns: *load* = A/W chunk loads incl. waiting for reads; *epilogue* = per-element epilogue incl.
scale/bias reads; *traffic / min* = bytes the core moved over the minimum; *traffic floor* =
those bytes / 16 (no schedule change at this tiling beats it); *schedule floor* = tiles x K (each
R x C output tile needs K feed cycles, full or not). A, A2, A3, B, AB are **projected** speedups
(estimates from the counters, not measurements):

| | change |
|---|---|
| A | many reads in flight: each beat costs one port cycle, latency hidden |
| A2 | A + the next K-chunk loads while the current one feeds |
| A3 | A2 + continuous feed: chunks back to back, the skew fills once per tile |
| B | parallel epilogue: one output row per cycle, scale/bias once per tile |
| AB | A3 and B |

## Decision: memory-level parallelism first (A -> A3), parallel epilogue second

1. Loads are 38-87 % of all cycles; the epilogue exceeds 25 % only for short-K layers
   (mlp_l1/l2, prefill on 16x16).
2. A3 is projected at 2.6-12.5x on every workload; B at 1.0-2.0x, and 1.01-1.03x on decode,
   the main edge-LLM workload.
3. B becomes worth it after A: on mlp_l1 16x16, A3 alone is 2.6x but A3+B is 8.5x, because the
   epilogue is then the largest phase.
4. Neither changes the contract: both are microarchitecture, so the P2 gates (fuzz on 5 array
   shapes, 11 mutants) judge them unchanged.

## What comes after: reuse, then the decode dataflow

- **No cross-tile reuse caps the array.** The output-stationary schedule reads R + C beats per
  16-byte K chunk and feeds for 16 cycles. At one beat per cycle the utilization ceiling is
  16 / (R + C): 100 % at 8x8, 50 % at 16x16, 25 % at 32x32, 6.2 % at 128x128. Traffic today is
  up to 15.6x the minimum. TARS's 128x128 array therefore needs the A or W tile held on chip
  (VMEM) across output tiles; that is the step after A.
- **Decode wastes rows.** With M = 1 only one of the R rows does work, so the schedule floor
  (tiles x K = 131,072 cycles at 8x8) sits above the memory roofline (66,244). At 16x16 the two
  floors meet. This is the effect V1.3.1's sub-array partitioning (2x64x128, 4x32x128) targets
  for decode.
