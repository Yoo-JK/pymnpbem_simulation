#!/bin/bash
#SBATCH --job-name=pymnpbem_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# CPU-only single node (no GPU).
# Submit:
#   sbatch --export=CONFIG=examples/dimer_baseline.yaml slurm_scripts/cpu_only.slurm

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate mnpbem

CONFIG="${CONFIG:-examples/dimer_baseline.yaml}"

echo "[slurm] job_id=${SLURM_JOB_ID} nodes=${SLURM_NNODES} cpus=${SLURM_CPUS_PER_TASK}"
echo "[slurm] config=${CONFIG}"

python run_simulation.py --config "${CONFIG}" --auto
