# P7: cross-tile loader prefetch

No contract change. The loader no longer restarts at each tile: it fetches the next tile's
first chunks while the current tile feeds its last ones.

## Mechanism

- The loader has its own tile (lm0, ln0), its own hold flags and its own conv addressing, and
  moves on as soon as every chunk (and vector beat) of its tile is issued.
- Chunk c of a tile lives in buffer (base + c) mod NB; the base runs on by K/16 per tile
  (l_base for the loader, f_base for the feed). Buffer b receives chunks in global order, so a
  slot freed by the current tile is refilled with the next chunk mapped to it, and the existing
  full / pending bits need no clearing between tiles (they are cleared once per descriptor).
- At most one tile ahead (l_tidx <= f_tidx). That bounds three hazards:
  - scale / bias vectors are double-buffered by accumulator bank; the loader fills the next
    tile's bank while the epilogue latches the current one's;
  - the next tile's bank is the one the tile before the current tile used, already latched;
  - no prefetch into a tile that refills the hold buffer while the current tile still reads it.

## Performance (latency 16; latency 2 is now within 0.5 %)

| workload | array | P6 | P7 |
|---|---|---:|---:|
| GEMM 64x256x256 + ReLU/GELU | 8x8 | 88.5 % | 95.4 % |
| | 16x16 | 80.1 % | 86.8 % |
| Conv 3x3 28x28x32 -> 32 | 8x8 | 91.2 % | 96.1 % |
| | 16x16 | 86.6 % | 91.5 % |

Per-tile load stalls at 16x16 GEMM fell from ~42 to ~18 cycles. What is left at 16x16:

- the read port: 17.9 k beats in 18.9 k cycles (95 % busy). With hold-buffer reuse the other
  operand is still streamed once per tile, 256 beats per 271-cycle tile; the stalls are bandwidth,
  not latency. Further gains need more reuse (both operands held, larger tiles) or a wider port.
- the skew fill: SK - 1 = 15 cycles per tile (~5 %), removable only by per-row feed counters
  that let the next tile's row 0 start while row R-1 finishes.

## Verification

- 531 tests. The 525 earlier ones pass unchanged (prefetch changes timing, never bits); new:
  - a hold-filling tile under R > C, heavy read backpressure and latency 16: waiting for the W
    columns does not imply the A rows are there, so a feed that read the hold buffer on that tile
    would see unfilled rows;
  - read traffic with hold reuse: W read once per column block. This one checks performance, not
    bits: a loader fetching a held operand it does not need gives the right result, only extra reads.
- Mutants: 75 / 75 caught. The first run caught 72 of 74. Both survivors were holes opened by
  prefetch (the feed's and the loader's hold flags are now separate signals): the feed reading
  the hold buffer on the tile that fills it (a race the old tests rarely lost) and the loader
  using the feed's hold flag (performance only). The two tests above close them; one more mutant
  (the loader skipping A on the filling tile) was added; a rerun caught all three.
