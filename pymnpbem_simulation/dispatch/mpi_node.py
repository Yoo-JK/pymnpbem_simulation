import os
import time

from typing import Any, Dict, Optional

import numpy as np

from ..util import print_info, print_error


def dispatch_mpi(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Optional[Dict[str, Any]]:

    try:
        import mpi4py  # noqa: F401
    except ImportError:
        raise RuntimeError(
                '[error] mpi4py not installed. pip install mpi4py and run with srun/mpirun.')

    from mnpbem.utils.mpi_dispatch import solve_spectrum_mpi

    pol = cfg['simulation'].get('polarizations',
            [[1, 0, 0], [0, 1, 0]])
    prop = cfg['simulation'].get('propagation_dirs',
            [[0, 0, 1], [0, 0, 1]])

    cfg_pickle = _strip_unpicklable(cfg)
    factory = _make_particle_factory(cfg_pickle)

    bem_kwargs = _build_bem_kwargs(cfg)

    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    print_info('mpi_node: n_wl={}, n_pol={}, n_gpus_per_node={}'.format(
            len(enei), len(pol), n_gpus_per_worker))

    t0 = time.time()
    raw = solve_spectrum_mpi(
            particle_factory = factory,
            enei = enei,
            pol_dirs = pol,
            prop_dirs = prop,
            n_gpus_per_node = n_gpus_per_worker if n_gpus_per_worker >= 1 else 0,
            bem_kwargs = bem_kwargs)
    wall_s = time.time() - t0

    if raw is None:
        return None

    ext = np.asarray(raw['ext'])
    sca = np.asarray(raw['sca'])
    abs_ = ext - sca

    n_pol = ext.shape[1]
    peak_idx = int(np.argmax(ext[:, 0]))
    peak_wl = float(enei[peak_idx])
    peak_ext_x = float(ext[peak_idx, 0])

    print_info('mpi_node: peak ext_x = {:.3f} at {:.2f} nm'.format(
            peak_ext_x, peak_wl))
    print_info('mpi_node: total wall = {:.2f} min, n_ranks={}'.format(
            wall_s / 60.0, raw.get('n_ranks', 1)))

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
            'per_rank_s': raw.get('per_rank_s', []),
            'n_ranks': raw.get('n_ranks', 1),
            'n_gpus_per_node': raw.get('n_gpus_per_node', 0)}


def _make_particle_factory(cfg: Dict[str, Any]):
    cfg_struct = cfg['structure']
    cfg_materials = cfg.get('materials', dict())

    def _factory():
        from ..structures import build_structure
        p, _, _ = build_structure(cfg_struct, cfg_materials)
        return p

    return _factory


def _build_bem_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bem_kwargs = dict()

    hmode = cfg['compute'].get('hmode', 'dense')

    if hmode != 'dense':
        bem_kwargs['hmode'] = hmode

    return bem_kwargs


def _strip_unpicklable(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    return copy.deepcopy(cfg)
