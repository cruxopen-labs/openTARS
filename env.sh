# Local tool environment for TARS-mini on this machine (AlmaLinux 8, no root): source env.sh
#   conda env "tars" (~/miniforge3): Python 3.12, numpy, scikit-learn, pytest, torch (CPU),
#   transformers, Verilator 5, Yosys, gcc 13, cmake, ccache; its activation puts the env's
#   libstdc++ first on LD_LIBRARY_PATH (torch's wheel would otherwise load the system's old one)
#   ~/eda_oss: sv2v (run through conda's glibc 2.34 loader), ASAP7 RVT TT merged liberty
. "$HOME/miniforge3/bin/activate" tars
export PATH="$HOME/eda_oss/bin:$PATH"
export ASAP7_LIB="$HOME/eda_oss/asap7/asap7_rvt_tt_merged.lib"
export SV2V="$HOME/eda_oss/bin/sv2v"
export YOSYS_NEW=yosys YOSYS_ABC=yosys      # tools/timing: the conda Yosys is new enough for both roles
export CMAKE_POLICY_VERSION_MINIMUM=3.5     # tools/fetch_dramsim3.sh under CMake 4
