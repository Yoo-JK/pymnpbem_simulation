#!/bin/bash
#SBATCH --job-name=pymnpbem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Single-node GPU (4 GPUs on one node).
# Submit:
#   sbatch --export=CONFIG=examples/dimer_baseline.yaml slurm_scripts/single_node_gpu.slurm

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate mnpbem

CONFIG="${CONFIG:-examples/dimer_baseline.yaml}"

echo "[slurm] job_id=${SLURM_JOB_ID} nodes=${SLURM_NNODES} gpus_on_node=${SLURM_GPUS_ON_NODE:-?}"
echo "[slurm] config=${CONFIG}"

python run_simulation.py --config "${CONFIG}" --auto
