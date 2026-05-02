import os

from typing import Any, Dict, List

import numpy as np

from ..util import print_info, print_error


def dispatch_single_node(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    n_workers = int(cfg['compute']['n_workers'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])
    multi_node = bool(cfg['compute'].get('multi_node', False))

    print_info('dispatch_single_node: n_workers={}, n_gpus_per_worker={}, multi_node={}, n_wl={}'.format(
            n_workers, n_gpus_per_worker, multi_node, len(enei)))

    if multi_node:
        from .mpi_node import dispatch_mpi
        return dispatch_mpi(cfg, p, epstab, enei)

    if n_gpus_per_worker >= 2:
        from .multi_gpu import dispatch_multi_gpu
        return dispatch_multi_gpu(cfg, p, epstab, enei)

    if n_gpus_per_worker == 1:

        if n_workers > 1:
            from .multi_gpu import dispatch_multi_gpu
            return dispatch_multi_gpu(cfg, p, epstab, enei)

        return _dispatch_single_gpu(cfg, p, epstab, enei)

    if n_workers > 1:
        return _dispatch_cpu_pool(cfg, p, epstab, enei)

    return _dispatch_cpu_serial(cfg, p, epstab, enei)


def _dispatch_single_gpu(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    os.environ['MNPBEM_GPU'] = '1'
    print_info('single GPU dispatch (MNPBEM_GPU=1)')

    return _dispatch_cpu_serial(cfg, p, epstab, enei)


def _dispatch_cpu_serial(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    from ..simulation.base import run_simulation

    return run_simulation(cfg, p, epstab, enei)


def _dispatch_cpu_pool(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    import multiprocessing as mp
    import time

    n_workers = int(cfg['compute']['n_workers'])
    n_wl = len(enei)

    if n_workers > n_wl:
        print_info('cpu_pool: n_workers={} > n_wl={}, capping to n_wl'.format(
                n_workers, n_wl))
        n_workers = n_wl

    chunks = _split_indices(n_wl, n_workers)
    print_info('cpu_pool: dispatching {} wavelengths to {} workers'.format(
            n_wl, n_workers))

    cfg_pickle = _strip_unpicklable(cfg)

    ctx = mp.get_context('spawn')
    results_q = ctx.Queue()

    procs = []
    t0 = time.time()

    for w in range(n_workers):
        wl_indices = chunks[w]

        if not wl_indices:
            continue

        enei_chunk = enei[wl_indices].tolist()
        proc = ctx.Process(
                target = _cpu_pool_worker,
                args = (w, wl_indices, enei_chunk, cfg_pickle, results_q))
        proc.start()
        procs.append(proc)

    n_pol = _infer_n_pol(cfg)
    ext = np.zeros((n_wl, n_pol))
    sca = np.zeros((n_wl, n_pol))
    abs_ = np.zeros((n_wl, n_pol))
    warm_s = 0.0
    errors = []

    for _ in procs:
        r = results_q.get()

        if not r.get('ok', False):
            errors.append(r)
            continue

        idxs = r['wl_indices']
        ext[idxs] = r['ext']
        sca[idxs] = r['sca']
        abs_[idxs] = r['abs']

        if r.get('warmup_s', 0.0) > warm_s:
            warm_s = r['warmup_s']

    for proc in procs:
        proc.join(timeout = 30)

    wall_s = time.time() - t0

    if errors:
        for err in errors:
            print_error('cpu_pool worker {} failed: {}'.format(
                    err.get('worker_idx', '?'), err.get('error', 'unknown')))

            if 'traceback' in err:
                print_error(err['traceback'])

        raise RuntimeError(
                '[error] cpu_pool: {} of {} workers failed'.format(
                        len(errors), len(procs)))

    peak_idx = int(np.argmax(ext[:, 0]))
    peak_wl = float(enei[peak_idx])
    peak_ext_x = float(ext[peak_idx, 0])

    print_info('cpu_pool: peak ext_x = {:.3f} at {:.2f} nm'.format(
            peak_ext_x, peak_wl))
    print_info('cpu_pool: total wall = {:.2f} min'.format(wall_s / 60.0))

    return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_pol}


def _cpu_pool_worker(worker_idx: int,
        wl_indices: List[int],
        enei_chunk: List[float],
        cfg: Dict[str, Any],
        queue) -> None:

    import time
    import traceback

    try:
        os.environ['MNPBEM_GPU'] = '0'
        os.environ.setdefault('MNPBEM_NUMBA', '1')

        n_threads = int(cfg['compute'].get('n_threads', 1))
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(n_threads)

        from ..structures import build_structure
        from ..simulation.base import run_simulation

        cfg_struct = cfg['structure']
        cfg_materials = cfg.get('materials', dict())

        p, epstab, _ = build_structure(cfg_struct, cfg_materials)

        enei_arr = np.asarray(enei_chunk, dtype = float)
        t0 = time.time()
        local = run_simulation(cfg, p, epstab, enei_arr)
        wall = time.time() - t0

        queue.put({
                'ok': True,
                'worker_idx': worker_idx,
                'wl_indices': list(wl_indices),
                'ext': np.asarray(local['ext']),
                'sca': np.asarray(local['sca']),
                'abs': np.asarray(local['abs']),
                'wall_s': wall,
                'warmup_s': float(local.get('warmup_s', 0.0))})

    except Exception as exc:
        queue.put({
                'ok': False,
                'worker_idx': worker_idx,
                'wl_indices': list(wl_indices),
                'error': repr(exc),
                'traceback': traceback.format_exc()})


def _split_indices(n_total: int,
        n_workers: int) -> List[List[int]]:

    chunks = [[] for _ in range(n_workers)]

    for i in range(n_total):
        chunks[i % n_workers].append(i)

    return chunks


def _infer_n_pol(cfg: Dict[str, Any]) -> int:
    pol = cfg['simulation'].get('polarizations',
            [[1, 0, 0], [0, 1, 0]])

    return len(pol)


def _strip_unpicklable(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    return copy.deepcopy(cfg)
