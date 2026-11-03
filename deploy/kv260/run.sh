#!/bin/bash
# Load the bitstream and start:   sudo ./run.sh [chat|selftest|quick|hw]   (default: chat)
# The bitstream is (re)loaded every time: never touch 0xA0000000 while the PL is not programmed
# (a read there with nothing to answer hangs the bus). The hugepages are released before the load:
# the FPGA manager needs ~8 MB of CMA for the bitstream, and hugepages can take the CMA area.
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0 $*"; exit 1; }
FW=/lib/firmware/xilinx/tars_kv260
# another run holds the core (its page table, its hugepages): reloading the PL under it would
# leave it driving a reset core
if pgrep -f "sw/(qwen_chat|selftest)\.py" >/dev/null; then
  echo "ERROR: TARS-mini is in use by another run:"; pgrep -af "sw/(qwen_chat|selftest)\.py"
  echo "quit it first (/quit in the chat)"; exit 1
fi
loaded() {
  [ "$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null)" = operating ] &&
    ls /sys/bus/platform/devices/ 2>/dev/null | grep -q tars_mini
}
echo 0 > /proc/sys/vm/nr_hugepages || true
xmutil unloadapp >/dev/null 2>&1 || true
xmutil loadapp tars_kv260 || true
sleep 1
if ! loaded; then
  echo "xmutil did not load the bitstream; trying fpgautil"
  xmutil unloadapp >/dev/null 2>&1 || true
  fpgautil -R >/dev/null 2>&1 || true
  fpgautil -b $FW/tars_kv260.bit.bin -o $FW/tars_kv260.dtbo || true
  sleep 1
fi
if ! loaded; then
  echo "ERROR: the bitstream did not load (FPGA manager: $(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null))."
  echo "Not touching the PL. Kernel messages:"
  dmesg | tail -15
  exit 1
fi
echo "bitstream loaded"
case "${1:-chat}" in
  chat)     python3 sw/qwen_chat.py "${@:2}" ;;
  selftest) python3 sw/selftest.py "${@:2}" ;;
  quick)    python3 sw/selftest.py --quick "${@:2}" ;;
  hw)       python3 sw/selftest.py --hw-only "${@:2}" ;;
  *) echo "usage: $0 [chat|selftest|quick|hw]"; exit 1 ;;
esac
