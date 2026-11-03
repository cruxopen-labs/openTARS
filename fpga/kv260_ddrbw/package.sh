#!/bin/bash
# Package the DDR bandwidth test: deploy/kv260_ddrbw/firmware/tgen<MHz>/ (same bitstream, one dtbo
# per pl_clk0 rate) for xmutil, plus the board scripts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$ROOT/fpga/kv260_ddrbw
XB=/opt/Xilinx/2025.2/Vitis/bin
B=$ROOT/build/kv260_ddrbw
OUT=$ROOT/deploy/kv260_ddrbw
mkdir -p "$B/pkg" "$OUT/firmware"
cp "$B/tgen.bit" "$B/pkg/tgen.bit"
printf 'all:\n{\n  [destination_device = pl] tgen.bit\n}\n' > "$B/pkg/bit.bif"
(cd "$B/pkg" && "$XB/bootgen" -image bit.bif -arch zynqmp -process_bitstream bin -w on)
for m in 200:199998000 250:249997500 300:299997009; do
  d=$OUT/firmware/tgen${m%%:*}; mkdir -p "$d"
  cp "$B/pkg/tgen.bit.bin" "$d/"
  sed "s/@RATE@/${m##*:}/" "$HERE/tgen.dtso.in" > "$B/pkg/tgen${m%%:*}.dtso"
  "$XB/dtc" -@ -I dts -O dtb -o "$d/tgen.dtbo" "$B/pkg/tgen${m%%:*}.dtso"
  printf '{\n  "shell_type" : "XRT_FLAT",\n  "num_slots": "1"\n}\n' > "$d/shell.json"
done
cp "$HERE/ddrbw.py" "$HERE/run_ddrbw.sh" "$OUT/"
ls -laR "$OUT"
