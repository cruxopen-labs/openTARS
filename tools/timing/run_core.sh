#!/bin/sh
# Pre-layout timing of tm_core (small parameters) with the fp units as black boxes: reports the
# longest register-to-register / register-to-unit-input path (ABC stime -p, ASAP7).
# Usage: run_core.sh <merged.lib> <sv2v> [chparam settings]
LIB=$1; SV2V=$2; shift 2; PARAMS=${*:-"-set R 2 -set C 2 -set NB 2 -set OT 4 -set HB 4 -set VG 1 -set OQ 4 -set LG 1 -set VQMIN 4 -set MEM_BYTES 65536"}
YN=${YOSYS_CORE:-yosys}; YA=${YOSYS_ABC:-yosys}   # native: the flattened core exceeds the 4 GB of the WebAssembly build
ROOT=$(cd "$(dirname "$0")/../.." && pwd); OUT=$ROOT/build/timing; mkdir -p $OUT
[ -n "$REUSE" ] && [ -f $OUT/core_g.v ] || {
$SV2V -E always $ROOT/rtl/tm_fp_pkg.sv $ROOT/rtl/tm_fpp_pkg.sv $ROOT/tools/timing/fpu_bb.sv $ROOT/rtl/tm_vpu.sv $ROOT/rtl/tm_xpu.sv \
      $ROOT/rtl/tm_pe.sv $ROOT/rtl/tm_array.sv $ROOT/rtl/tm_core.sv > $OUT/core.v 2> $OUT/sv2v.err || { cat $OUT/sv2v.err; exit 1; }
$YN -q -l $OUT/core_syn.log -p "read_verilog -sv $OUT/core.v; chparam $PARAMS tm_core; synth -top tm_core -flatten -noabc -run :fine; techmap -map +/techmap.v -map +/choices/kogge-stone.v; synth -top tm_core -flatten -noabc -run fine:; write_verilog -noattr $OUT/core_g.v" >/dev/null 2>&1 || { tail -20 $OUT/core_syn.log; exit 1; }
}
$YA -q -l $OUT/core_t.log -p "read_verilog $OUT/core_g.v; proc; techmap; opt -fast; dfflibmap -liberty $LIB; abc -liberty $LIB -script +strash;map;topo;buffer,-p,-N,4;upsize;dnsize;stime,-p"   # no dch: its choices exceed 4 GB on the whole core >/dev/null 2>&1
grep -o "Delay = *[0-9.]* ps" $OUT/core_t.log | tail -1
grep -E "Start-point|End-point" $OUT/core_t.log | tail -2
