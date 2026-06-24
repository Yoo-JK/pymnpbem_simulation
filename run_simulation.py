import sys
import os


def _set_threads_pre_import():
    # numpy(MKL)/numba 는 import 시점에 thread 수를 고정한다. cli.main 의
    # setup_env 는 numpy import(cli.py) 이후라 MKL thread 에 무효 — 여기서
    # numpy import 전에 미리 박는다. 이미 설정된 외부 env 는 존중(setdefault).
    nt = str(os.cpu_count() or 1)
    for i, a in enumerate(sys.argv):
        if a == '--n-threads' and i + 1 < len(sys.argv):
            nt = sys.argv[i + 1]
        elif a.startswith('--n-threads='):
            nt = a.split('=', 1)[1]
    for k in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ.setdefault(k, nt)


_set_threads_pre_import()

from pymnpbem_simulation.cli import main


if __name__ == '__main__':
    sys.exit(main())
