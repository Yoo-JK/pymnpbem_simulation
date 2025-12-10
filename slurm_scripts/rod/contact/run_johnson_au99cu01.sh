#!/bin/bash
#SBATCH --job-name=mnpbem
#SBATCH --account=yoojk20-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

module purge
module load matlab/24.1
module load miniconda3/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnpbem

echo "Job started on $(date)"

echo "---------- Start simulation: Johnson / Au99Cu01 / Distance: Contact ----------"
cd /u/yoojk20/workspace/mnpbem_simulation
./master.sh --str-conf ./config/rod_contact/aucu/johnson/config_str_au99cu01.py --sim-conf ./config/rod_contact/aucu/johnson/config_sim_au99cu01.py --verbose

echo "Job finished on $(date)"

