import os

from typing import Tuple


def detect_n_gpus() -> int:
    if 'SLURM_GPUS_ON_NODE' in os.environ:
        return int(os.environ['SLURM_GPUS_ON_NODE'])

    if 'SLURM_JOB_GPUS' in os.environ:
        return len(os.environ['SLURM_JOB_GPUS'].split(','))

    if 'PBS_GPUFILE' in os.environ:
        path = os.environ['PBS_GPUFILE']

        if os.path.exists(path):
            with open(path) as f:
                return len([line for line in f if line.strip()])

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cvd = os.environ['CUDA_VISIBLE_DEVICES']

        if cvd == '':
            return 0

        return len(cvd.split(','))

    return 0


def detect_n_cpus() -> int:
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])

    if 'SLURM_CPUS_ON_NODE' in os.environ:
        return int(os.environ['SLURM_CPUS_ON_NODE'])

    if 'PBS_NP' in os.environ:
        return int(os.environ['PBS_NP'])

    return os.cpu_count() or 1


def auto_compute_plan(n_gpus: int = -1,
        n_cpus: int = -1) -> Tuple[int, int, int]:
    if n_gpus < 0:
        n_gpus = detect_n_gpus()

    if n_cpus < 0:
        n_cpus = detect_n_cpus()

    if n_gpus >= 1:
        n_workers = n_gpus
        n_gpus_per_worker = 1
        n_threads = max(1, n_cpus // n_gpus)
    else:
        n_workers = max(1, n_cpus)
        n_gpus_per_worker = 0
        n_threads = 1

    return n_workers, n_threads, n_gpus_per_worker
