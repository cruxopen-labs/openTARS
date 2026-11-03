#!/bin/sh
# Fetch and build DRAMsim3 (pinned) into ext_dramsim3/ for the DRAM-timed testbench.
set -e
cd "$(dirname "$0")/.."
REV=29817593b3389f1337235d63cac515024ab8fd6e
[ -d ext_dramsim3 ] || git clone -q https://github.com/umd-memsys/DRAMsim3 ext_dramsim3
cd ext_dramsim3 && git checkout -q $REV
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j"$(nproc)" dramsim3
