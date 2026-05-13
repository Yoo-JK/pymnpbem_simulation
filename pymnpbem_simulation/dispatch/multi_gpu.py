import os
import time

from typing import Any, Dict, List

import numpy as np

from ..util import print_info, print_error


def dispatch_multi_gpu(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    n_workers = int(cfg['compute']['n_workers'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    if n_gpus_per_worker > 1:
        return _dispatch_vram_share(cfg, p, epstab, enei)

    return _dispatch_wavelength_split(cfg, p, epstab, enei)


def _dispatch_wavelength_split(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    from mnpbem.utils.multi_gpu import solve_spectrum_multi_gpu

    n_workers = int(cfg['compute']['n_workers'])
    n_wl = len(enei)

    if n_workers > n_wl:
        print_info('multi_gpu: n_workers={} > n_wl={}, capping to n_wl'.format(
                n_workers, n_wl))
        n_workers = n_wl

    pol = cfg['simulation'].get('polarizations',
            [[1, 0, 0], [0, 1, 0]])
    prop = cfg['simulation'].get('propagation_dirs',
            [[0, 0, 1], [0, 0, 1]])

    cfg_pickle = _strip_unpicklable(cfg)
    factory = _make_particle_factory(cfg_pickle)

    bem_kwargs = _build_bem_kwargs(cfg)

    # v1.5.1 Bug 4 follow-up: forward simulation.type to mnpbem so the
    # wavelength-split path picks the right BEM class. Previously this
    # path was hard-coded to BEMRet (dense) in mnpbem.utils.multi_gpu,
    # which OOM'd on 12k+ face Au@Ag dimer when the user requested iter.
    bem_class_name = _resolve_bem_class_name(cfg)

    # Forward iter options (hmatrix, tol, maxit, etc.) to the worker BEM
    # constructor when using an iterative solver.
    if bem_class_name in ('BEMRetIter', 'BEMRetLayerIter'):
        iter_cfg = cfg['simulation'].get('iter', dict()) or dict()
        for key in ('solver', 'tol', 'maxit', 'restart', 'precond',
                    'output', 'hmatrix', 'htol', 'cleaf', 'kmax'):
            if key in iter_cfg:
                bem_kwargs[key] = iter_cfg[key]

    print_info('multi_gpu wavelength-split: n_gpus={}, n_wl={}, n_pol={}, bem_class={}'.format(
            n_workers, n_wl, len(pol), bem_class_name or 'BEMRet'))

    t0 = time.time()
    raw = solve_spectrum_multi_gpu(
            particle_factory = factory,
            enei = enei,
            pol_dirs = pol,
            prop_dirs = prop,
            n_gpus = n_workers,
            bem_kwargs = bem_kwargs,
            bem_class = bem_class_name)
    wall_s = time.time() - t0

    ext = np.asarray(raw['ext'])
    sca = np.asarray(raw['sca'])
    abs_ = ext - sca

    n_pol = ext.shape[1]
    peak_idx = int(np.argmax(ext[:, 0]))
    peak_wl = float(enei[peak_idx])
    peak_ext_x = float(ext[peak_idx, 0])

    print_info('multi_gpu: peak ext_x = {:.3f} at {:.2f} nm'.format(
            peak_ext_x, peak_wl))
    print_info('multi_gpu: total wall = {:.2f} min'.format(wall_s / 60.0))

    return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'wall_s': wall_s,
            'warmup_s': 0.0,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_pol,
            'per_gpu_s': raw.get('per_gpu_s', []),
            'n_gpus': raw.get('n_gpus', n_workers)}


def _dispatch_vram_share(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:
    """v1.2.0: VRAM share — pool multi-GPU memory for one large LU factorisation.

    Activated when ``compute.n_gpus_per_worker > 1``. Sets the
    ``MNPBEM_VRAM_SHARE`` family of environment variables that the
    mnpbem ``lu_factor_dispatch`` (Agent β) consults at runtime, then
    falls through to the standard single-GPU dispatch path so that the
    dense LU factor in BEMStat/BEMRet is partitioned across GPUs by the
    chosen backend (cuSolverMg / Magma / NCCL).

    If the installed mnpbem port has not yet adopted the VRAM-share env
    vars (Agent β still in flight), the env vars are simply ignored and
    the run degrades to a single-GPU LU on the default device.
    """
    backend = cfg['compute'].get('vram_share_backend', 'cusolvermg')
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    print_info(
            'VRAM share dispatch: n_gpus_per_worker={}, backend={}'.format(
                    n_gpus_per_worker, backend))

    os.environ['MNPBEM_VRAM_SHARE'] = '1'
    os.environ['MNPBEM_VRAM_SHARE_GPUS'] = str(n_gpus_per_worker)
    os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = str(backend)

    # Single worker handles the full computation; the LU dispatch picks
    # up the env vars internally and partitions across the requested GPUs.
    from .single_node import _dispatch_single_gpu
    try:
        return _dispatch_single_gpu(cfg, p, epstab, enei)
    finally:
        # Best-effort cleanup so subsequent dispatches in the same process
        # don't inherit stale VRAM-share state.
        for key in ('MNPBEM_VRAM_SHARE',
                'MNPBEM_VRAM_SHARE_GPUS',
                'MNPBEM_VRAM_SHARE_BACKEND'):
            os.environ.pop(key, None)


def _make_particle_factory(cfg: Dict[str, Any]):
    import functools
    cfg_struct = cfg['structure']
    cfg_materials = cfg.get('materials', dict())

    return functools.partial(_particle_factory_top, cfg_struct, cfg_materials)


def _particle_factory_top(cfg_struct: Dict[str, Any],
        cfg_materials: Dict[str, Any]):
    from pymnpbem_simulation.structures import build_structure
    p, _, _ = build_structure(cfg_struct, cfg_materials)
    return p


def _build_bem_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bem_kwargs = dict()

    hmode = cfg['compute'].get('hmode', 'dense')

    if hmode != 'dense':
        bem_kwargs['hmode'] = hmode

    # v1.5.1 — propagate iter-only kwargs to the worker BEM class so the
    # wavelength-split path actually uses the requested H-matrix +
    # preconditioner combination (otherwise mnpbem's worker silently runs
    # plain BEMRetIter dense which still OOMs on 12k face).
    iter_cfg = cfg['compute'].get('iter', dict()) or dict()
    sim_type = cfg['simulation'].get('type', 'ret')
    is_iter_path = sim_type.endswith('_iter') or bool(cfg['compute'].get('iterative', False))

    if is_iter_path:
        if 'hmatrix' in iter_cfg:
            v = iter_cfg['hmatrix']
            if isinstance(v, str) and v.lower() == 'auto':
                bem_kwargs['hmatrix'] = True
            else:
                bem_kwargs['hmatrix'] = bool(v)
        else:
            # default ON for large mesh — matches docs/PERFORMANCE.md
            bem_kwargs['hmatrix'] = True

        if 'preconditioner' in iter_cfg:
            bem_kwargs['preconditioner'] = iter_cfg['preconditioner']
        else:
            bem_kwargs['preconditioner'] = 'auto'

        if 'schur' in iter_cfg:
            v = iter_cfg['schur']
            if isinstance(v, str) and v.lower() == 'auto':
                pass  # let mnpbem decide
            else:
                bem_kwargs['schur'] = bool(v)

        if 'htol' in iter_cfg:
            bem_kwargs['htol'] = float(iter_cfg['htol'])

        if 'tol' in iter_cfg:
            bem_kwargs['tol'] = float(iter_cfg['tol'])

        if 'maxit' in iter_cfg:
            bem_kwargs['maxit'] = int(iter_cfg['maxit'])

    return bem_kwargs


def _resolve_bem_class_name(cfg: Dict[str, Any]) -> str:
    """Return the mnpbem BEM class name implied by simulation.type.

    Mirrors the runner registry used by single-node dispatch.
    """
    sim_type = cfg.get('simulation', {}).get('type', 'ret')

    table = {
            'ret':                      'BEMRet',
            'ret_iter':                 'BEMRetIter',
            'ret_layer':                'BEMRetLayer',
            'ret_layer_iter':           'BEMRetLayerIter',
            'planewave_ret':            'BEMRet',
            'planewave_ret_iter':       'BEMRetIter',
            'planewave_ret_layer':      'BEMRetLayer',
            'planewave_ret_layer_iter': 'BEMRetLayerIter'}

    return table.get(sim_type, 'BEMRet')


def _strip_unpicklable(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    return copy.deepcopy(cfg)
