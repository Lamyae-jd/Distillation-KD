# OpenSTA script: post-synthesis static timing analysis for StudentNet
# on SKY130 (typical corner). Determines Fmax by measuring worst-slack
# at a fast-clock probe period, then Fmax = 1 / (period - slack).
#
# Environment inputs (set by run_synth.sh):
#   PDK_LIB     : path to sky130_fd_sc_hd__tt_025C_1v80.lib
#   WORK_DIR    : directory containing synth_netlist.v
#   TOP_MODULE  : top module name (myproject)
#   CLK_PORT    : clock port name (ap_clk for HLS4ML/Vitis output)
#   PROBE_PERIOD_NS : probe period in ns (default 1.0)

set pdk_lib      $::env(PDK_LIB)
set work_dir     $::env(WORK_DIR)
set top_module   $::env(TOP_MODULE)
set clk_port     $::env(CLK_PORT)
set probe_period $::env(PROBE_PERIOD_NS)

read_liberty $pdk_lib
read_verilog $work_dir/synth_netlist.v
link_design  $top_module

#---------register count-------
# all_registers renvoie tous les endpoints séquentiels (les DFF) du design.
# On le fait ici, juste après link_design, avant même la clock, pour
# être sûr que la commande a accès à toute la hiérarchie linked.
set n_registers [llength [all_registers]]

create_clock -name clk -period $probe_period [get_ports $clk_port]

# Drive/load model at IO boundary — matches abc.constr for consistency.
set_input_delay  0.1 -clock clk [delete_from_list [all_inputs] [get_ports $clk_port]]
set_output_delay 0.1 -clock clk [all_outputs]

# ---------- power activity defaults ----------
# report_power a besoin de connaître le taux d'activité de chaque net pour
# estimer le power dynamique (P_dyn = 0.5 * C * V² * f * activity).
# Sans VCD réel, on utilise le default OpenSTA :
#   activity = 0.1  → 10 % des cycles horloge, un net "toggle" (change d'état)
#   duty     = 0.5  → moitié du temps à '1', moitié à '0'
# C'est une convention pour "generic switching" — bon pour un ordre de
# grandeur dans l'abstract. Pour du power précis il faudrait un VCD.

set_power_activity -input -activity 0.1 -duty 0.5

# Report the timing summary in a machine-parseable form.
report_checks -path_delay max -format full -digits 4
puts "-------- summary --------"

# Worst slack across all paths; if positive we still have headroom at
# probe_period ns, else the real critical delay > probe_period.
set worst_slack [sta::worst_slack -max]
set critical_delay [expr $probe_period - $worst_slack]
set fmax_mhz [expr 1000.0 / $critical_delay]

puts [format "PROBE_PERIOD_NS %.4f" $probe_period]
puts [format "WORST_SLACK_NS  %.4f" $worst_slack]
puts [format "CRITICAL_DELAY_NS %.4f" $critical_delay]
puts [format "FMAX_MHZ        %.2f" $fmax_mhz]

# Native OpenSTA summaries
report_worst_slack -max
report_tns
#report_design_area

#----------top-5 critical paths -----------
puts ""
puts "-------Top 5 critical paths --------"
#group_path_count 5 = imprime les 5 pires chemins de setup
report_checks -path_delay max -group_path_count 5 -format full -digits 4


#------------- endpoint / check_type summary --------
puts ""
puts " ---------check types (endpoint counts) ---------"
# Décompose les endpoints (la ou finit un path) par type de check:
#combien de setup checks, combien de holds,etc.
report_check_types -max_delay -min_delay -max_slew -max_capacitance -max_fanout -digits 4


# ---------- design size ----------
puts ""
puts "-------- DESIGN SIZE --------"
# report_design_area n'existe pas dans OpenSTA (c'est une commande OpenROAD) —
# l'aire vient de 'yosys stat -liberty', voir report.txt.
puts [format "N_REGISTERS     %d" $n_registers]

# ---------- power ----------
puts ""
puts "-------- POWER (default activity 10%/50%) --------"
# Décompose : internal (charge/decharge interne des cellules),
# switching (charge/decharge des nets), leakage (statique, courants de fuite),
# et total. Séparé par groupe : combinational, sequential, clock, macros.
report_power

exit 0


