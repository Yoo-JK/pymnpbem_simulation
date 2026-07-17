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
        gpu_list = [x.strip() for x in cvd.split(',') if x.strip()]
        return len(gpu_list)

    # fallback for if cuda visible devices isn't set (typical if there is only one GPU like the lab machine)
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount()
    except (ImportError, Exception):
        return 0


def detect_n_cpus() -> int:
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])

    if 'SLURM_CPUS_ON_NODE' in os.environ:
        return int(os.environ['SLURM_CPUS_ON_NODE'])

    if 'PBS_NP' in os.environ:
        return int(os.environ['PBS_NP'])

    return os.cpu_count() or 1


def detect_multi_node() -> bool:
    if 'SLURM_NNODES' in os.environ:

        try:

            if int(os.environ['SLURM_NNODES']) > 1:
                return True

        except ValueError:
            pass

    if 'SLURM_JOB_NUM_NODES' in os.environ:

        try:

            if int(os.environ['SLURM_JOB_NUM_NODES']) > 1:
                return True

        except ValueError:
            pass

    if 'PBS_NUM_NODES' in os.environ:

        try:

            if int(os.environ['PBS_NUM_NODES']) > 1:
                return True

        except ValueError:
            pass

    if 'OMPI_COMM_WORLD_SIZE' in os.environ:

        try:

            if int(os.environ['OMPI_COMM_WORLD_SIZE']) > 1:
                return True

        except ValueError:
            pass

    if 'PMI_SIZE' in os.environ:

        try:

            if int(os.environ['PMI_SIZE']) > 1:
                return True

        except ValueError:
            pass

    if 'MPI_LOCALNRANKS' in os.environ:

        try:

            if int(os.environ.get('MPI_NUMRANKS', '1')) > 1:
                return True

        except ValueError:
            pass

    if 'PMIX_RANK' in os.environ:

        if 'PMIX_NAMESPACE' in os.environ or 'PMIX_SERVER_URI' in os.environ \
                or 'PMIX_HOSTNAME' in os.environ:
            return True

    if 'PBS_NODEFILE' in os.environ:
        path = os.environ['PBS_NODEFILE']

        if os.path.exists(path):

            try:

                with open(path) as f:
                    unique_nodes = set(
                            line.strip() for line in f if line.strip())

                    if len(unique_nodes) > 1:
                        return True

            except OSError:
                pass

    if 'SLURM_PROCID' in os.environ and 'SLURM_NNODES' in os.environ:

        try:

            if int(os.environ['SLURM_NNODES']) > 1:
                return True

        except ValueError:
            pass

    return False


def detect_mpi_rank() -> int:
    for key in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID', 'MPI_RANK']:

        if key in os.environ:

            try:
                return int(os.environ[key])

            except ValueError:
                continue

    return 0


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
