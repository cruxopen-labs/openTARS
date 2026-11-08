#!/bin/bash
# One-time setup on the KV260 (Kria Ubuntu 22.04), from this directory:  sudo ./setup.sh
#  1. the Python packages from wheels/ unpacked into pylib/ (no pip, no network needed)
#  2. the bitstream into /lib/firmware/xilinx/tars_kv260 (for xmutil)
#  (the hugepages for the core's 1 GB buffer are taken by run.sh, after the bitstream is loaded)
set -euo pipefail
cd "$(dirname "$0")"
PY=$(python3 -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
echo "python: $PY, $(uname -m)"
[ "$(uname -m)" = aarch64 ] || { echo "run this on the KV260 (aarch64)"; exit 1; }
rm -rf pylib && mkdir pylib
for w in wheels/*.whl; do
  case "$w" in
    *-cp3[0-9]*-cp3[0-9]*-*) [[ "$w" == *"-$PY-"* ]] || continue ;;
  esac
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$w" pylib
done
PYTHONPATH=pylib python3 -c "import numpy, tokenizers; print('numpy', numpy.__version__, 'tokenizers', tokenizers.__version__)"
mkdir -p /lib/firmware/xilinx/tars_kv260
cp firmware/tars_kv260/tars_kv260.bit.bin firmware/tars_kv260/tars_kv260.dtbo firmware/tars_kv260/shell.json \
   /lib/firmware/xilinx/tars_kv260/
echo "firmware installed: /lib/firmware/xilinx/tars_kv260"
(cd data && sha256sum -c ../SHA256SUMS.data --quiet) && echo "data files: checksums ok"
for f in SHA256SUMS.data_*; do                # the other models' exports (data_gemma3, ...), if present
  [ -f "$f" ] && [ -d "${f#SHA256SUMS.}" ] || continue
  (cd "${f#SHA256SUMS.}" && sha256sum -c "../$f" --quiet) && echo "${f#SHA256SUMS.} files: checksums ok"
done
