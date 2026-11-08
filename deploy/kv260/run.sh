#!/bin/bash
# Load the bitstream and start:  sudo ./run.sh [model] [command] [args]
#   model    qwen3.5 (default: data/), qwen3.5-mtp (data_mtp/), qwen3 (data_qwen3/), gemma3 (data_gemma3/)
#   command  chat (default) | selftest | quick | hw | bench | profile | precision
# e.g. sudo ./run.sh gemma3 chat;  sudo ./run.sh qwen3 selftest;  sudo ./run.sh quick
# The bitstream is (re)loaded every time: never touch 0xA0000000 while the PL is not programmed
# (a read there with nothing to answer hangs the bus). The hugepages are released before the load:
# the FPGA manager needs ~8 MB of CMA for the bitstream, and hugepages can take the CMA area.
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0 $*"; exit 1; }
FW=/lib/firmware/xilinx/tars_kv260
MODELS="qwen3.5 qwen3.5-mtp qwen3 gemma3"
USAGE="usage: $0 [$(echo $MODELS | tr ' ' '|')] [chat|selftest|quick|hw|bench|profile|precision] [args]"
DATA=()                                    # no model given: each program's own default
case "${1:-}" in
  qwen3.5)     DATA=(--data data);        shift ;;
  qwen3.5-mtp) DATA=(--data data_mtp);    shift ;;
  qwen3)       DATA=(--data data_qwen3);  shift ;;
  gemma3)      DATA=(--data data_gemma3); shift ;;
  -h|--help|help) echo "$USAGE"; exit 0 ;;
esac
if [ ${#DATA[@]} -gt 0 ] && [ ! -f "${DATA[1]}/manifest.json" ]; then
  echo "ERROR: ${DATA[1]}/ has no export (manifest.json): copy it from the PC (tools/export_kv260.py)"; exit 1
fi
CMD="${1:-chat}"
case "$CMD" in
  chat|selftest|quick|hw|bench|profile|precision) ;;
  *) echo "$USAGE"; exit 1 ;;
esac
# another run holds the core (its page table, its hugepages): reloading the PL under it would
# leave it driving a reset core
if pgrep -f "sw/(qwen_chat|selftest|spec_bench|perf_profile|check_state_precision|ctx_bench)\.py" >/dev/null; then
  echo "ERROR: TARS-mini is in use by another run:"; pgrep -af "sw/(qwen_chat|selftest|spec_bench|perf_profile|check_state_precision|ctx_bench)\.py"
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
case "$CMD" in
  chat)     python3 sw/qwen_chat.py "${DATA[@]}" "${@:2}" ;;
  selftest) python3 sw/selftest.py "${DATA[@]}" "${@:2}" ;;
  quick)    python3 sw/selftest.py "${DATA[@]}" --quick "${@:2}" ;;
  hw)       python3 sw/selftest.py "${DATA[@]}" --hw-only "${@:2}" ;;
  bench)    python3 sw/spec_bench.py "${DATA[@]}" "${@:2}" ;;
  profile)  python3 sw/perf_profile.py "${DATA[@]}" "${@:2}" ;;
  precision) python3 sw/check_state_precision.py --fpga "${@:2}" ;;
esac
