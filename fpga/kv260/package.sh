#!/bin/bash
# Package the bitstream for Kria Ubuntu (xmutil / fpgautil): deploy/kv260/firmware/tars_kv260/
#   tars_kv260.bit.bin (bootgen -process_bitstream bin), tars_kv260.dtbo, shell.json
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
XB=/opt/Xilinx/2025.2/Vitis/bin
OUT=$ROOT/deploy/kv260/firmware/tars_kv260
B=$ROOT/build/kv260
mkdir -p "$OUT" "$B/pkg"
cp "$B/tars_kv260.bit" "$B/pkg/tars_kv260.bit"
printf 'all:\n{\n  [destination_device = pl] tars_kv260.bit\n}\n' > "$B/pkg/bit.bif"
(cd "$B/pkg" && "$XB/bootgen" -image bit.bif -arch zynqmp -process_bitstream bin -w on)
cp "$B/pkg/tars_kv260.bit.bin" "$OUT/"
"$XB/dtc" -@ -I dts -O dtb -o "$OUT/tars_kv260.dtbo" "$ROOT/fpga/kv260/tars_kv260.dtso"
printf '{\n  "shell_type" : "XRT_FLAT",\n  "num_slots": "1"\n}\n' > "$OUT/shell.json"
cp "$B/tars_kv260.xsa" "$ROOT/deploy/kv260/firmware/" 2>/dev/null || true
(cd "$OUT" && sha256sum tars_kv260.bit.bin tars_kv260.dtbo shell.json > SHA256SUMS)
ls -la "$OUT"
