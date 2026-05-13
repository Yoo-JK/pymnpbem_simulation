import os

from typing import Any, Dict, List

import numpy as np

from ..util import print_info, print_error


def is_field_calculation(cfg: Dict[str, Any]) -> bool:
    sim_cfg = cfg.get('simulation', dict())
    sim_type = sim_cfg.get('type', 'ret')

    # Explicit type tags: 'field', 'field_ret', 'field_stat'.
    if sim_type in ('field', 'field_ret', 'field_stat'):
        return True

    # Implicit: a 'grid' block in 'simulation' implies a field calculation.
    if 'grid' in sim_cfg:
        return True

    return False


# Issue A (v1.5.1) — auto-redirect simulation.type when compute.iterative=true.
#
# Legacy py->yaml migration emits the original mnpbem_simulation flag
# ``use_iterative_solver=True`` as ``compute.iterative=true`` while keeping
# ``simulation.type='ret'`` (or 'stat' / 'ret_layer'). Previously this
# combo silently fell through to the dense BEMRet solver, defeating the
# user's intent (and OOM-ing on 12k+ face meshes such as the jk-config
# Au@Ag dimer 4nm shell case).
#
# This helper rewrites ``simulation.type`` in-place on the cfg dict so all
# downstream dispatch (single GPU / multi-GPU / CPU pool / MPI) sees the
# correct iter-path runner key in ``REGISTRY``.
_ITER_TYPE_MAP = {
        'ret': 'ret_iter',
        'stat': 'stat_iter',
        'ret_layer': 'ret_layer_iter'}


def _redirect_iterative_to_iter_type(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """If compute.iterative=true, redirect simulation.type to its _iter
    counterpart when one exists. Returns the (possibly modified) cfg.

    No-op when:
      * compute.iterative is false / missing
      * simulation.type is already an _iter variant
      * no _iter variant exists for the current type (e.g. 'eels',
        'dipole', 'field' — these have no iterative runner today)
    """
    if not isinstance(cfg, dict):
        return cfg

    compute = cfg.get('compute', dict())
    iterative = bool(compute.get('iterative', False))

    if not iterative:
        return cfg

    sim = cfg.get('simulation', dict())
    sim_type = sim.get('type', 'ret')

    if not isinstance(sim_type, str):
        return cfg

    if sim_type.endswith('_iter'):
        return cfg

    new_type = _ITER_TYPE_MAP.get(sim_type, None)

    if new_type is None:
        print_info(
                'iterative redirect: simulation.type=<{}> has no _iter variant — left as-is'.format(
                        sim_type))
        return cfg

    print_info(
            'iterative redirect (Issue A): compute.iterative=true detected — '
            'simulation.type <{}> -> <{}>'.format(sim_type, new_type))

    sim['type'] = new_type
    cfg['simulation'] = sim
    return cfg


def dispatch_single_node(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    # v1.5.1 — Issue A — translate compute.iterative=true into _iter type
    # before any dispatch decision. Touch the cfg dict directly so the
    # rewrite reaches all worker subprocesses (cpu_pool / multi_gpu) that
    # snapshot cfg further down.
    _redirect_iterative_to_iter_type(cfg)

    n_workers = int(cfg['compute']['n_workers'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])
    multi_node = bool(cfg['compute'].get('multi_node', False))

    print_info('dispatch_single_node: n_workers={}, n_gpus_per_worker={}, multi_node={}, n_wl={}'.format(
            n_workers, n_gpus_per_worker, multi_node, len(enei)))

    if is_field_calculation(cfg):
        print_info('dispatch_single_node: field calculation detected (grid block) — routing to FieldCalculator')
        return _dispatch_field(cfg, p, epstab, enei)

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


def _dispatch_field(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    import time

    from ..simulation.field_calculator import FieldCalculator

    n_gpus_per_worker = int(cfg['compute'].get('n_gpus_per_worker', 0))

    if n_gpus_per_worker >= 1:
        os.environ['MNPBEM_GPU'] = '1'
        print_info('field dispatch: MNPBEM_GPU=1 (n_gpus_per_worker={})'.format(n_gpus_per_worker))

    fc = FieldCalculator(cfg, p, epstab)

    t0 = time.time()
    result = fc.run(enei)
    wall_s = time.time() - t0

    result['wall_s'] = wall_s
    result['warmup_s'] = 0.0
    result['kind'] = 'field'

    print_info('field dispatch: wall = {:.2f} min'.format(wall_s / 60.0))

    return result


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
    sc_records = []
    sc_mesh = None

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

        if r.get('sc_local', None) is not None:
            sc_local = r['sc_local']

            for k in range(len(sc_local['wavelengths'])):
                sc_records.append({
                        'wavelength': float(sc_local['wavelengths'][k]),
                        'wl_idx': int(sc_local['wl_indices'][k]),
                        'sig2': sc_local['sig2'][k],
                        'sig1': sc_local['sig1'][k]})

            if sc_mesh is None:
                sc_mesh = {
                        'verts': sc_local['verts'],
                        'faces': sc_local['faces'],
                        'centroids': sc_local['centroids'],
                        'normals': sc_local['normals'],
                        'areas': sc_local['areas'],
                        'polarizations': sc_local['polarizations']}

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

    out = {
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

    if sc_records and sc_mesh is not None:
        sc_records.sort(key = lambda r: r['wl_idx'])
        nfaces = sc_records[0]['sig2'].shape[0]
        sig2_all = np.zeros((len(sc_records), nfaces, n_pol), dtype = complex)
        sig1_all = np.zeros((len(sc_records), nfaces, n_pol), dtype = complex)

        for k, rec in enumerate(sc_records):
            n_pol_rec = min(n_pol, rec['sig2'].shape[1])
            sig2_all[k, :, :n_pol_rec] = rec['sig2'][:, :n_pol_rec]
            sig1_all[k, :, :n_pol_rec] = rec['sig1'][:, :n_pol_rec]

        sc_dict = {
                'wavelengths': np.asarray([r['wavelength'] for r in sc_records]),
                'wl_indices': np.asarray([r['wl_idx'] for r in sc_records],
                        dtype = int),
                'sig2': sig2_all,
                'sig1': sig1_all}
        sc_dict.update(sc_mesh)
        out['surface_charge'] = sc_dict
        print_info('cpu_pool: aggregated surface_charge across {} wavelengths'.format(
                len(sc_records)))

    return out


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

        sc_local = local.get('surface_charge', None)

        if sc_local is not None:
            global_indices = np.asarray(wl_indices, dtype = int)
            local_idx = np.asarray(sc_local['wl_indices'], dtype = int)
            sc_local = {
                    'wavelengths': np.asarray(sc_local['wavelengths']),
                    'wl_indices': global_indices[local_idx],
                    'sig2': np.asarray(sc_local['sig2']),
                    'sig1': np.asarray(sc_local['sig1']),
                    'verts': np.asarray(sc_local['verts']),
                    'faces': np.asarray(sc_local['faces']),
                    'centroids': np.asarray(sc_local['centroids']),
                    'normals': np.asarray(sc_local['normals']),
                    'areas': np.asarray(sc_local['areas']),
                    'polarizations': np.asarray(sc_local['polarizations'])}

        queue.put({
                'ok': True,
                'worker_idx': worker_idx,
                'wl_indices': list(wl_indices),
                'ext': np.asarray(local['ext']),
                'sca': np.asarray(local['sca']),
                'abs': np.asarray(local['abs']),
                'wall_s': wall,
                'warmup_s': float(local.get('warmup_s', 0.0)),
                'sc_local': sc_local})

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
