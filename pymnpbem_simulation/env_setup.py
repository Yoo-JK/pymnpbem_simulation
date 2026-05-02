import os

from typing import Dict, Any


def setup_env(n_threads: int,
        n_gpus_per_worker: int,
        extra: Dict[str, str] = None) -> None:

    n_threads_str = str(n_threads)

    os.environ['OMP_NUM_THREADS'] = n_threads_str
    os.environ['MKL_NUM_THREADS'] = n_threads_str
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads_str
    os.environ['NUMEXPR_NUM_THREADS'] = n_threads_str
    os.environ['NUMBA_NUM_THREADS'] = n_threads_str

    if n_gpus_per_worker >= 1:
        os.environ['MNPBEM_GPU'] = '1'
    else:
        os.environ['MNPBEM_GPU'] = '0'

    os.environ['MNPBEM_NUMBA'] = '1'

    if extra is not None:
        for k, v in extra.items():
            os.environ[k] = v


def assert_pre_import() -> None:
    if 'mnpbem' in __import__('sys').modules:
        raise RuntimeError(
            '[error] mnpbem already imported! setup_env() must run before mnpbem import.')
