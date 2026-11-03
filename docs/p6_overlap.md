# P6: row-op throughput and double-buffered accumulators

No contract change: the golden model and every test vector are as in v0.3. D7 is decided:
softmax / LayerNorm / RMSNorm keep their multi-pass algorithms (no online softmax, no Welford).

## Row ops: VG beats per cycle, overlapped rows

| Mode (chosen per descriptor) | When | Per row |
|---|---|---|
| overlapped | two rows (+ gamma, beta) fit in the hold buffer | pass 0 (DDR -> hold buffer, 1 beat/cycle) of row r+1 runs beside the later passes of row r |
| sequential | one row (+ gamma, beta) fits | pass 0 from DDR, later passes VG beats/cycle from the hold buffer |
| DDR re-read | the row does not fit | every pass from DDR, 1 beat/cycle (as in P5) |

- VG (core parameter, 1/2/4, default 4) beats per step in hold-buffer passes. Steps start at
  multiples of VG, so the 4·VG elements of a step go to distinct lanes and every lane still sees
  its elements in order: the contract's 16-lane order is untouched. N is a multiple of 16, so a
  step is never partial.
- Overlap: stage A (pass 0 of row a_row into region a_buf, its own 16 lanes and fold) and stage B
  (the later passes of the previous row from the other region). A folded row is handed to B when
  B is idle; A then starts the next row in the other region. The reader runs ahead across rows.
- Cost: the hold buffer needs 3·VG read ports in hold-buffer passes (row, gamma, beta), 16·VG
  elements of exp / fmul datapath, and a second set of 16 lane registers plus fold tree.

| workload (64 x 1024, -> bf16) | P5 | VG = 4 only | + overlap |
|---|---:|---:|---:|
| softmax | 32.2-32.8 % | 54.0-55.7 % | 97.6-97.9 % |
| LayerNorm (+ beta) | 32.9-33.5 % | 54.7-56.5 % | 97.5-97.9 % |
| RMSNorm | 48.0-49.4 % | 63.1-65.4 % | 97.9-98.3 % |

(Roofline: the row's fp32 read at 16 B/cycle; reads and writes on separate channels.)

## Double-buffered accumulators

- Each PE has two banks. The activation carries its tile's bank (tile parity) and a K-step-0
  flag; the first product of a tile replaces the bank's old value, so there is no clear cycle.
- The epilogue is an engine of its own: at the end of a tile's feed the tile is handed over
  (bank, m0, n0, live columns, and the C scale / bias values latched), and the core starts the
  next tile at once. The engine writes one row per cycle the writer can take.
- Safety: row i of the old bank is final once the array has advanced i times after the old
  tile's last feed step (PE (i, j) adds its last product at step K-1+i+j <= K+SK-2+i). The engine
  counts array advances (not cycles: the next tile's feed may freeze the array). A tile is handed
  over only when the engine is idle, so tile t+2 cannot write the bank tile t is being read from.

| workload | array | lat 2 before -> after | lat 16 before -> after |
|---|---|---:|---:|
| GEMM 64x256x256 + ReLU/GELU | 8x8 | 90.3 -> 92.9 % | 86.2 -> 88.5 % |
| | 16x16 | 76.3 -> 84.0 % | 73.2 -> 80.1 % |
| Conv 3x3 28x28x32 -> 32 | 8x8 | 92.5 -> 95.0 % | 88.9 -> 91.2 % |
| | 16x16 | 86.2 -> 90.5 % | 82.7 -> 86.6 % |

Epilogue waiting fell from 32 to 1 cycle per tile (16x16 GEMM). What is left, per tile at
16x16, lat 16: the loader restarting at the tile's first chunk (~42 cycles) and the skew fill
(SK - 1 = 15 cycles). Next step: cross-tile prefetch (the loader fetches the next tile's first
chunks into free buffers while the current tile feeds its last ones).

## Verification

- 525 tests. New: compiled row ops in all three modes (HB 1024 / 40 / 16 / 10) and VG 1, 2, 4.
- Mutants: 65 / 65 caught. The run had 66; 65 were caught, including all 18 new ones (bank
  selection, the first-product flag, the advance-counted read gate, cycle- instead of advance-
  counting, handoff to a busy engine, the latched bias, region selection, the region swap, handoff
  while the previous row runs, gamma placement, the overlap capacity check, the wide write, lane
  maps and folds of both stages, both -inf initialisations). One was equivalent (GEMM end without
  the epilogue-idle check: the writer is busy without a gap until the engine has gone idle) and
  was removed with a note.
- Equivalent by construction, so no mutant: the partial-step guard (N % 16 = 0 makes every
  hold-buffer step full) and VG itself (VG = 1 gives the same bits, only slower).
