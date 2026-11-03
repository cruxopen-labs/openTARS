#!/bin/sh
# Like run_units.sh, for the pipeline-stage functions in tools/timing/stages.sv.
LIB=$1; SV2V=$2; shift 2; OPS=${*:-"0 1 2 3 4 5 6 7 8 9 10"}
YN=${YOSYS_NEW:-yowasp-yosys}; YA=${YOSYS_ABC:-yosys}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); OUT=$ROOT/build/timing; mkdir -p $OUT
$SV2V $ROOT/rtl/tm_fp_pkg.sv $ROOT/rtl/tm_fpp_pkg.sv $ROOT/tools/timing/stages.sv > $OUT/stages.v
for op in $OPS; do
  $YN -q -p "read_verilog -sv $OUT/stages.v; chparam -set OP $op s_stage; synth -top s_stage -flatten -booth -noabc -run :fine; techmap -map +/techmap.v -map +/choices/kogge-stone.v; synth -top s_stage -flatten -booth -noabc -run fine:; write_verilog -noattr $OUT/sg$op.v" >/dev/null 2>&1
  $YA -q -l $OUT/s$op.log -p "read_verilog $OUT/sg$op.v; proc; techmap; opt -fast; dfflibmap -liberty $LIB; abc -liberty $LIB -script +strash;dch,-f;map;topo;buffer,-p,-N,4;upsize;dnsize;stime,-p" >/dev/null 2>&1
  echo "STAGE $op: $(grep -o 'Delay = *[0-9.]* ps' $OUT/s$op.log | tail -1)"
done
