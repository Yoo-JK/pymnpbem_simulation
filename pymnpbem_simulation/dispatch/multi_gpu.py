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

    backend = cfg['compute'].get('vram_share_backend', 'cusolvermg')
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    raise NotImplementedError(
            '[error] VRAM share (n_gpus_per_worker={}) with backend <{}> '
            'is planned for M5+ release. See project_mnpbem_gpu_vram_sharing memo.'.format(
                    n_gpus_per_worker, backend))


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

    return bem_kwargs


def _resolve_bem_class_name(cfg: Dict[str, Any]) -> str:
    """Map simulation.type → BEM solver class name for multi_gpu dispatch.

    Bug 4 follow-up: solve_spectrum_multi_gpu defaults to BEMRet when
    bem_class is unset, silently ignoring simulation.type=ret_iter etc.
    Dispatch must inspect cfg and pass the correct class name.
    """
    sim_type = cfg.get('simulation', {}).get('type', 'ret')
    mapping = {
            'ret':              'BEMRet',
            'ret_iter':         'BEMRetIter',
            'ret_layer':        'BEMRetLayer',
            'ret_layer_iter':   'BEMRetLayerIter',
            'planewave_ret':       'BEMRet',
            'planewave_ret_iter':  'BEMRetIter',
            'planewave_ret_layer': 'BEMRetLayer',
            'planewave_ret_layer_iter': 'BEMRetLayerIter'}
    return mapping.get(sim_type, 'BEMRet')


def _strip_unpicklable(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    return copy.deepcopy(cfg)
