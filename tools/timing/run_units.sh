#!/bin/sh
# Pre-layout logic depth of each wrapped unit (tools/timing/units.sv) on an ASAP7 liberty:
#   sv2v -> Yosys >= 0.38 (synth -booth -noabc, adders as Kogge-Stone: the structure a commercial
#   tool would pick) -> Yosys + ABC (delay-optimal map to the liberty, buffering, gate sizing,
#   static timing: stime -p). No wires, no placement: an optimistic pre-layout logic delay.
# Usage: run_units.sh <merged.lib> <sv2v> [ops...]   (YOSYS_NEW=yowasp-yosys, YOSYS_ABC=yosys)
LIB=$1; SV2V=$2; shift 2; OPS=${*:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"}
YN=${YOSYS_NEW:-yowasp-yosys}; YA=${YOSYS_ABC:-yosys}; T=${TARGET_PS:-650}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); OUT=$ROOT/build/timing; mkdir -p $OUT
$SV2V $ROOT/rtl/tm_fp_pkg.sv $ROOT/tools/timing/units.sv > $OUT/units.v
for op in $OPS; do
  $YN -q -p "read_verilog -sv $OUT/units.v; chparam -set OP $op u_reg2; synth -top u_reg2 -flatten -booth -noabc -run :fine; techmap -map +/techmap.v -map +/choices/kogge-stone.v; synth -top u_reg2 -flatten -booth -noabc -run fine:; write_verilog -noattr $OUT/g$op.v" >/dev/null 2>&1
  $YA -q -l $OUT/u$op.log -p "read_verilog $OUT/g$op.v; proc; techmap; opt -fast; dfflibmap -liberty $LIB; abc -liberty $LIB -script +strash;dch,-f;map;topo;buffer,-p,-N,4;upsize;dnsize;stime,-p" >/dev/null 2>&1
  d=$(grep -o "Delay = *[0-9.]* ps" $OUT/u$op.log | tail -1)
  a=$(grep -o "Area = *[0-9.]*" $OUT/u$op.log | tail -1)
  echo "OP $op: $d  $a"
done
