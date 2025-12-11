#!/bin/bash
#SBATCH --job-name=4_agg_s
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --export=NONE

module purge
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: Aggregation 4 Sphere(s) ----------"
cd /u/yoojk20/workspace/pymnpbem_simulation
./master.sh --str-conf ./config/agg_sph/w_sub/config_str_4_agg.py --sim-conf ./config/agg_sph/w_sub/config_sim_4_agg.py --verbose

echo "Job finished on $(date)"

