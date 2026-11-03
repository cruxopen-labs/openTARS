# DDR bandwidth test for the KV260: four AXI traffic generators (tg_top), one on each of
# S_AXI_HP0..HP3_FPD (128 bits, no interconnect in between), registers at 0xA000_0000 over
# M_AXI_HPM0_FPD. One clock, pl_clk0, closed at 300 MHz (the PS gives 299.997); the dtbo picks the rate at load time
# (200 / 250 / 300 MHz), so one bitstream measures every port clock.
#   vivado -mode batch -source fpga/kv260_ddrbw/build.tcl [-tclargs <jobs>]
set here [file normalize [file dirname [info script]]]
set root [file normalize [file join $here .. ..]]
set out  [file join $root build kv260_ddrbw]
set jobs [expr {[llength $argv] > 0 ? [lindex $argv 0] : 6}]
file mkdir $out

create_project -force tgen [file join $out prj] -part xck26-sfvc784-2LV-c
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]
add_files [list $here/tg_master.v $here/tg_top.v]
update_compile_order -fileset sources_1

create_bd_design tgen
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {300} \
  CONFIG.PSU__FPGA_PL1_ENABLE {0} CONFIG.PSU__USE__IRQ0 {0} ] $ps

set tg  [create_bd_cell -type module -reference tg_top tg]
set scc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $scc
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl]

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI]    [get_bd_intf_pins tg/s_axil]
foreach i {0 1 2 3} {
  connect_bd_intf_net [get_bd_intf_pins tg/m${i}_axi] [get_bd_intf_pins ps/S_AXI_HP${i}_FPD]
}
set clk [get_bd_pins ps/pl_clk0]
foreach p {ps/maxihpm0_fpd_aclk ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk
           ps/saxihp3_fpd_aclk tg/clk sc_ctl/aclk rst_pl/slowest_sync_clk} {
  connect_bd_net $clk [get_bd_pins $p]
}
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_pl/ext_reset_in]
foreach p {tg/rstn sc_ctl/aresetn} {
  connect_bd_net [get_bd_pins rst_pl/peripheral_aresetn] [get_bd_pins $p]
}

assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data] -filter {NAME =~ "*tg*"}]
set_property offset 0xA0000000 $seg
set_property range 16K $seg
puts "ADDRESS MAP:"
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] { puts "  ps: $s [get_property offset $s] [get_property range $s]" }
foreach i {0 1 2 3} {
  foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces tg/m${i}_axi]] { puts "  m$i: $s [get_property offset $s] [get_property range $s]" }
}
puts "PL CLOCK: [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]"
validate_bd_design
save_bd_design

set bd [get_files tgen.bd]
generate_target all $bd
make_wrapper -files $bd -top
add_files -norecurse [file join $out prj tgen.gen sources_1 bd tgen hdl tgen_wrapper.v]
set_property top tgen_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synthesis failed" }
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

open_run impl_1
report_utilization -file [file join $out utilization.rpt]
report_timing_summary -max_paths 10 -file [file join $out timing.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "TIMING WNS=$wns WHS=$whs"
file copy -force [glob [file join $out prj tgen.runs impl_1 *.bit]] [file join $out tgen.bit]
puts "DONE"
