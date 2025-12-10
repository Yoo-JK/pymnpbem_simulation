#!/bin/bash
#SBATCH --job-name=auag02_16
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE
#SBATCH --exclusive

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: AuAg dimer 0.2 nm gap ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/test/config_str.py --sim-conf ./config/test/config_sim_16.py --verbose

echo "Job finished on $(date)"

