#!/bin/sh
# Pre-layout timing of tm_core, hierarchical: Yosys keeps the submodules (PE, row-engine lanes and
# fold, delay lines), so ABC (with dch, the delay-oriented restructuring the unit runs use) maps
# one module at a time within memory; the mapped netlist is then flattened for tools/timing/sta.py,
# which times every register-to-register path (fp units as black boxes with their first-stage
# delay). Usage: run_core_h.sh <merged.lib> <sv2v> [chparam settings]
LIB=$1; SV2V=$2; shift 2; PARAMS=${*:-"-set R 2 -set C 2 -set NB 2 -set OT 4 -set HB 4 -set VG 1 -set OQ 4 -set LG 1 -set VQMIN 4 -set MEM_BYTES 65536"}
YN=${YOSYS_CORE:-yosys}
# ABC maps without area recovery (map -r) and sizes every path within 50 % of the worst (upsize -W):
# with area recovery it relaxes each path that is not the worst up to the worst path's delay, and
# only the worst path would mean anything.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); OUT=$ROOT/build/timing; mkdir -p $OUT
$SV2V -E always $ROOT/rtl/tm_fp_pkg.sv $ROOT/rtl/tm_fpp_pkg.sv $ROOT/tools/timing/fpu_bb.sv $ROOT/rtl/tm_vpu.sv $ROOT/rtl/tm_xpu.sv \
      $ROOT/rtl/tm_pe.sv $ROOT/rtl/tm_array.sv $ROOT/rtl/tm_core.sv > $OUT/core.v 2> $OUT/sv2v.err || { cat $OUT/sv2v.err; exit 1; }
$YN -q -l $OUT/core_h.log -p "read_verilog -sv $OUT/core.v; chparam $PARAMS tm_core; synth -top tm_core -noabc -run :fine; techmap -map +/techmap.v -map +/choices/kogge-stone.v; synth -top tm_core -noabc -run fine:; abc -liberty $LIB -script +strash;dch,-f;map,-r;topo;buffer,-p,-N,4;upsize,-W,50,-T,900; flatten; opt_clean; write_verilog -noattr $OUT/core_h.v" >/dev/null 2>&1 || { tail -20 $OUT/core_h.log; exit 1; }
python3 $ROOT/tools/timing/sta.py $OUT/core_h.v $LIB --top ${TOP:-60}
