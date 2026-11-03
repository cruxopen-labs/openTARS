# TARS-mini on the KV260: block design (Zynq MPSoC + the core), implementation, bitstream.
#   vivado -mode batch -source fpga/kv260/build.tcl [-tclargs <jobs>]
# Outputs in build/kv260/: tars_kv260.bit, tars_kv260.xsa, reports (utilization, timing).
#
# PS: the KV260 SOM preset (DDR4, MIO); pl_clk0 = 100 MHz (the core), pl_clk1 = 250 MHz (the DDR
# ports); M_AXI_HPM0_FPD (128 bits) -> the registers at 0xA000_0000 (16 KB); the core's four read
# masters (128 bits at 250 MHz, their clock crossing inside the core's top): r0, r1, r2 ->
# S_AXI_HP0_FPD, S_AXI_HP3_FPD, S_AXI_HP2_FPD directly; r3 and the write master (256 bits at
# 100 MHz) -> SmartConnect (width and clock conversion) -> S_AXI_HP1_FPD (measured: 12.5 GB/s of
# reads over the four ports at 250 MHz). Every port reaches all of DDR (low 2 GB and the high
# window at 0x8_0000_0000).
set root [file normalize [file join [file dirname [info script]] .. ..]]
set out  [file join $root build kv260]
set jobs [expr {[llength $argv] > 0 ? [lindex $argv 0] : 6}]
file mkdir $out

create_project -force tars_kv260 [file join $out prj] -part xck26-sfvc784-2LV-c
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

set rtl [file join $root rtl]
add_files [list \
  $rtl/tm_fp_pkg.sv $rtl/tm_fpp_pkg.sv $rtl/tm_fpu.sv $rtl/tm_sfu.sv $rtl/tm_mem.sv $rtl/tm_vpu.sv \
  $rtl/tm_xpu.sv $rtl/tm_vdly.sv $rtl/tm_pe.sv $rtl/tm_array.sv $rtl/tm_core.sv $rtl/tm_kv260.sv \
  $rtl/tm_kv260_wrap.v]
set_property verilog_define {TM_FPGA} [get_filesets sources_1]
update_compile_order -fileset sources_1

create_bd_design tars
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
  CONFIG.PSU__FPGA_PL1_ENABLE {1} CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ {250} \
  CONFIG.PSU__USE__IRQ0 {0} ] $ps

set tars [create_bd_cell -type module -reference tm_kv260_wrap tars]
set scc  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $scc
set scm  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_mem]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {2}] $scm
set rst  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl]
set rsth [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_hp]

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI]    [get_bd_intf_pins tars/s_axil]
connect_bd_intf_net [get_bd_intf_pins tars/m_axi]        [get_bd_intf_pins sc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem/M00_AXI]    [get_bd_intf_pins ps/S_AXI_HP1_FPD]
connect_bd_intf_net [get_bd_intf_pins tars/r0_axi]       [get_bd_intf_pins ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins tars/r1_axi]       [get_bd_intf_pins ps/S_AXI_HP3_FPD]
connect_bd_intf_net [get_bd_intf_pins tars/r2_axi]       [get_bd_intf_pins ps/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins tars/r3_axi]       [get_bd_intf_pins sc_mem/S01_AXI]

set clk [get_bd_pins ps/pl_clk0]
foreach p {ps/maxihpm0_fpd_aclk tars/clk sc_ctl/aclk sc_mem/aclk rst_pl/slowest_sync_clk} {
  connect_bd_net $clk [get_bd_pins $p]
}
set clk1 [get_bd_pins ps/pl_clk1]
foreach p {ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk ps/saxihp3_fpd_aclk sc_mem/aclk1 tars/hclk rst_hp/slowest_sync_clk} {
  connect_bd_net $clk1 [get_bd_pins $p]
}
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_pl/ext_reset_in]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_hp/ext_reset_in]
connect_bd_net [get_bd_pins rst_hp/peripheral_aresetn] [get_bd_pins tars/hrstn]
foreach p {tars/rstn sc_ctl/aresetn sc_mem/aresetn} {
  connect_bd_net [get_bd_pins rst_pl/peripheral_aresetn] [get_bd_pins $p]
}

assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data] -filter {NAME =~ "*tars*"}]
set_property offset 0xA0000000 $seg
set_property range 16K $seg
puts "ADDRESS MAP:"
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] { puts "  ps: $s [get_property offset $s] [get_property range $s]" }
foreach sp {m_axi r0_axi r1_axi r2_axi r3_axi} {
  foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces tars/$sp]] { puts "  tars/$sp: $s [get_property offset $s] [get_property range $s]" }
}
puts "PL CLOCKS: [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps] [get_property CONFIG.PSU__CRL_APB__PL1_REF_CTRL__ACT_FREQMHZ $ps]"
validate_bd_design
save_bd_design

set bd [get_files tars.bd]
generate_target all $bd
make_wrapper -files $bd -top
add_files -norecurse [file join $out prj tars_kv260.gen sources_1 bd tars hdl tars_wrapper.v]
set_property top tars_wrapper [current_fileset]
update_compile_order -fileset sources_1
# the read ports' clock-crossing FIFOs (tm_afifo): a gray pointer reaches its first synchronizer
# stage within the faster clock's period (the clocks are unrelated for these paths)
set xdc [file join $out cdc.xdc]
set f [open $xdc w]
puts $f {set_max_delay -datapath_only 4.0 -from [get_cells -hier -filter {NAME =~ *afg_w_reg*}] -to [get_cells -hier -filter {NAME =~ *afs1_w_reg*}]}
puts $f {set_max_delay -datapath_only 4.0 -from [get_cells -hier -filter {NAME =~ *afg_r_reg*}] -to [get_cells -hier -filter {NAME =~ *afs1_r_reg*}]}
close $f
add_files -fileset constrs_1 -norecurse $xdc
set_property used_in_synthesis false [get_files $xdc]

set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
# area-optimized synthesis (the core, a module reference, is synthesized in its own run): the
# design is near the KV260's LUT count, and the default directive leaves it too dense to place
foreach r [get_runs -filter {IS_SYNTHESIS}] {
  set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high $r
}
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synthesis failed" }
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

open_run impl_1
report_utilization -file [file join $out utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 4 -file [file join $out utilization_hier.rpt]
report_timing_summary -max_paths 20 -file [file join $out timing.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "TIMING WNS=$wns WHS=$whs"
file copy -force [glob [file join $out prj tars_kv260.runs impl_1 *.bit]] [file join $out tars_kv260.bit]
write_hw_platform -fixed -include_bit -force [file join $out tars_kv260.xsa]
puts "DONE"
