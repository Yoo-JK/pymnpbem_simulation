#!/bin/bash
#SBATCH --job-name=pymnpbem_mn4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Multi-node MPI (4 nodes x 4 GPUs = 16-way wavelength parallelism).
# Submit:
#   sbatch --export=CONFIG=examples/large_mesh_multinode.yaml slurm_scripts/multi_node_4.slurm

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate mnpbem

CONFIG="${CONFIG:-examples/large_mesh_multinode.yaml}"

echo "[slurm] job_id=${SLURM_JOB_ID} nodes=${SLURM_NNODES} ntasks=${SLURM_NTASKS}"
echo "[slurm] gpus_on_node=${SLURM_GPUS_ON_NODE:-?} config=${CONFIG}"

srun --mpi=pmix python run_simulation.py --config "${CONFIG}" --multi-node --auto
