#!/bin/bash
#SBATCH --job-name=pymnpbem_mn2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Multi-node MPI (2 nodes x 4 GPUs = 8-way wavelength parallelism).
# Each rank uses 4 local GPUs via solve_spectrum_multi_gpu.
# Submit:
#   sbatch --export=CONFIG=examples/dimer_multinode.yaml slurm_scripts/multi_node_2.slurm

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate mnpbem

CONFIG="${CONFIG:-examples/dimer_multinode.yaml}"

echo "[slurm] job_id=${SLURM_JOB_ID} nodes=${SLURM_NNODES} ntasks=${SLURM_NTASKS}"
echo "[slurm] gpus_on_node=${SLURM_GPUS_ON_NODE:-?} config=${CONFIG}"

# srun launches one task per node; each task sees its node's 4 GPUs.
srun --mpi=pmix python run_simulation.py --config "${CONFIG}" --multi-node --auto
