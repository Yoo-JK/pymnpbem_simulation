#!/bin/bash
#SBATCH --job-name=au10
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --export=NONE

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: Au dimer 1.0 nm gap ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/dimer/au/str/r0.2/config_str_au_r0.2_g1.0.py --sim-conf ./config/dimer/au/sim/r0.2/config_sim_au_r0.2_g1.0.py --verbose

echo "Job finished on $(date)"

