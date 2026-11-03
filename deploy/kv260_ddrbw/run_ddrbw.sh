#!/bin/bash
# DDR bandwidth test:  sudo ./run_ddrbw.sh [MHz ...]   (default: 200 250 300)
# Installs the tgen<MHz> apps, loads each in turn, runs ddrbw.py, then puts TARS-mini back
# (xmutil loadapp tars_kv260). Refuses to run while a TARS chat or self-test holds the PL.
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0 $*"; exit 1; }
if pgrep -f "sw/(qwen_chat|selftest)\.py" >/dev/null; then
  echo "ERROR: TARS-mini is in use:"; pgrep -af "sw/(qwen_chat|selftest)\.py"; exit 1
fi
for d in firmware/tgen*; do
  mkdir -p /lib/firmware/xilinx/$(basename $d); cp $d/* /lib/firmware/xilinx/$(basename $d)/
done
loaded() {
  [ "$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null)" = operating ] &&
    ls /sys/bus/platform/devices/ 2>/dev/null | grep -q "$1"
}
restore() {
  echo 0 > /proc/sys/vm/nr_hugepages || true
  xmutil unloadapp >/dev/null 2>&1 || true
  xmutil loadapp tars_kv260 >/dev/null 2>&1 || true
  loaded tars_mini && echo "TARS-mini bitstream restored" || echo "WARNING: TARS-mini not restored; sudo ./run.sh reloads it"
}
trap restore EXIT
for mhz in "${@:-200 250 300}"; do
  for f in $mhz; do
    echo "=== tgen$f ==="
    echo 0 > /proc/sys/vm/nr_hugepages
    xmutil unloadapp >/dev/null 2>&1 || true
    xmutil loadapp tgen$f >/dev/null || true
    sleep 1
    if ! loaded tgen; then echo "ERROR: tgen$f did not load; not touching the PL"; dmesg | tail -5; exit 1; fi
    echo 16 > /proc/sys/vm/nr_hugepages
    python3 ddrbw.py --out ddrbw_$f.json ${DDRBW_ARGS:-}
  done
done
