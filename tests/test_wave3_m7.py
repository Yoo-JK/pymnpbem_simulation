import os
import sys

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# Mirror smoke
# ============================================================================

def test_with_mirror_in_registry() -> None:
    from pymnpbem_simulation.structures import REGISTRY

    assert 'with_mirror' in REGISTRY, '[error] with_mirror missing from REGISTRY'


def test_with_mirror_builds_xy() -> None:
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'with_mirror',
            'base': {'type': 'sphere', 'diameter': 20, 'mesh_density': 60},
            'mirror': {'sym': 'xy'}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, epstab, n = build_structure(cfg, cfg_m)

    assert n > 0, '[error] mirror produced 0 half-faces'
    assert len(epstab) >= 2
    assert hasattr(p, 'pfull'), '[error] mirror particle missing pfull'
    assert p.pfull.nfaces == 4 * n, \
            '[error] xy mirror: full nfaces ({}) != 4 * half ({})'.format(p.pfull.nfaces, n)
    assert getattr(p, 'sym', None) == 'xy'


def test_with_mirror_builds_x_only() -> None:
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'with_mirror',
            'base': {'type': 'sphere', 'diameter': 20, 'mesh_density': 60},
            'mirror': {'sym': 'x'}}
    p, _, n = build_structure(cfg, {'medium': 'water', 'particle': 'gold'})
    assert p.pfull.nfaces == 2 * n


def test_with_mirror_invalid_sym_raises() -> None:
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'with_mirror',
            'base': {'type': 'sphere', 'diameter': 20, 'mesh_density': 60},
            'mirror': {'sym': 'z'}}

    with pytest.raises(ValueError):
        build_structure(cfg, {'medium': 'water', 'particle': 'gold'})


def test_planewave_ret_mirror_smoke() -> None:
    """Mirror BEM solver runs end-to-end and produces finite, positive cross sections."""
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_struct = {'type': 'with_mirror',
            'base': {'type': 'sphere', 'diameter': 20, 'mesh_density': 60},
            'mirror': {'sym': 'x'}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, eps, _ = build_structure(cfg_struct, cfg_m)

    cfg = {'structure': cfg_struct,
            'simulation': {'type': 'ret_mirror', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]]}}
    res = build_simulation(p, eps, cfg).run(np.array([550.0, 600.0]))

    assert res['ext'].shape == (2, 1)
    assert np.all(np.isfinite(res['ext']))
    assert np.all(res['ext'] > 0), '[error] mirror ext not strictly positive'


# ============================================================================
# Iterative BEM smoke (compare against dense)
# ============================================================================

def test_ret_iter_in_registry() -> None:
    from pymnpbem_simulation.simulation import REGISTRY

    keys = list(REGISTRY.keys())
    assert ('ret_iter', 'planewave') in keys, '[error] ret_iter not registered'
    assert ('stat_iter', 'planewave') in keys, '[error] stat_iter not registered'
    assert ('ret_layer_iter', 'planewave') in keys, '[error] ret_layer_iter not registered'


def test_planewave_ret_iter_matches_dense() -> None:
    """ret_iter (GMRES tol=1e-8) should match dense BEMRet to ~1e-10 precision."""
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_struct = {'type': 'sphere', 'diameter': 30, 'mesh_density': 60}
    cfg_m = {'medium': 'water', 'particle': 'gold'}
    enei = np.array([600.0, 700.0])

    p, eps, _ = build_structure(cfg_struct, cfg_m)
    cfg_dense = {'structure': cfg_struct,
            'simulation': {'type': 'ret', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]]}}
    res_dense = build_simulation(p, eps, cfg_dense).run(enei)

    p2, eps2, _ = build_structure(cfg_struct, cfg_m)
    cfg_iter = {'structure': cfg_struct,
            'simulation': {'type': 'ret_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                    'iter': {'tol': 1e-8, 'maxit': 500, 'precond': 'hmat'}}}
    res_iter = build_simulation(p2, eps2, cfg_iter).run(enei)

    rel = np.max(np.abs(res_iter['ext'] - res_dense['ext'])
            / np.maximum(np.abs(res_dense['ext']), 1e-30))
    assert rel < 1e-8, \
            '[error] ret_iter vs dense rel diff {:.3e} > 1e-8'.format(rel)


def test_planewave_stat_iter_matches_dense() -> None:
    """stat_iter (GMRES tol=1e-8) should match dense BEMStat to ~1e-10."""
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_struct = {'type': 'sphere', 'diameter': 30, 'mesh_density': 60}
    cfg_m = {'medium': 'water', 'particle': 'gold'}
    enei = np.array([550.0, 700.0])

    p, eps, _ = build_structure(cfg_struct, cfg_m)
    cfg_dense = {'structure': cfg_struct,
            'simulation': {'type': 'stat', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]]}}
    res_dense = build_simulation(p, eps, cfg_dense).run(enei)

    p2, eps2, _ = build_structure(cfg_struct, cfg_m)
    cfg_iter = {'structure': cfg_struct,
            'simulation': {'type': 'stat_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]],
                    'iter': {'tol': 1e-8, 'maxit': 500, 'precond': 'hmat'}}}
    res_iter = build_simulation(p2, eps2, cfg_iter).run(enei)

    rel = np.max(np.abs(res_iter['ext'] - res_dense['ext'])
            / np.maximum(np.abs(res_dense['ext']), 1e-30))
    assert rel < 1e-8, \
            '[error] stat_iter vs dense rel diff {:.3e} > 1e-8'.format(rel)


def test_ret_iter_dimer_smoke() -> None:
    """small dimer iter run produces finite positive ext."""
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_struct = {'type': 'dimer_sphere', 'diameter': 20, 'gap': 5, 'n_verts': 60}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, eps, _ = build_structure(cfg_struct, cfg_m)
    cfg = {'structure': cfg_struct,
            'simulation': {'type': 'ret_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                    'iter': {'tol': 1e-6, 'maxit': 300, 'precond': 'hmat'}}}
    res = build_simulation(p, eps, cfg).run(np.array([600.0, 700.0]))

    assert np.all(np.isfinite(res['ext']))
    assert np.all(res['ext'] > 0)
    assert res['solver_type'] == 'BEMRetIter'


# ============================================================================
# Nonlocal eps smoke
# ============================================================================

def test_with_nonlocal_in_registry() -> None:
    from pymnpbem_simulation.structures import REGISTRY

    assert 'with_nonlocal' in REGISTRY


def test_nonlocal_k0_matches_local() -> None:
    """nonlocal eps at k=0 must reduce to base local eps exactly (bit-identical)."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'sphere', 'diameter': 20, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'k_nm_inv': 0.0}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    _p, eps_loc, _ = build_structure(cfg_local, cfg_m)
    _p2, eps_nl, _ = build_structure(cfg_nl, cfg_m)

    e_loc, _ = eps_loc[1](600.0)
    e_nl, _ = eps_nl[1](600.0)

    assert np.isclose(e_loc, e_nl, rtol = 0, atol = 0), \
            '[error] nonlocal at k=0 differs from local: {} vs {}'.format(e_loc, e_nl)


def test_nonlocal_kpos_shifts_eps() -> None:
    """nonlocal eps at k>0 should differ from local."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'sphere', 'diameter': 20, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'k_nm_inv': 0.05}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    _p, eps_loc, _ = build_structure(cfg_local, cfg_m)
    _p2, eps_nl, _ = build_structure(cfg_nl, cfg_m)

    e_loc, _ = eps_loc[1](600.0)
    e_nl, _ = eps_nl[1](600.0)

    assert not np.isclose(e_loc, e_nl), \
            '[error] nonlocal at k=0.05 should differ from local'


def test_nonlocal_full_simulation_k0() -> None:
    """Full BEM with nonlocal(k=0) should produce ext identical to local-baseline."""
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_local = {'type': 'sphere', 'diameter': 20, 'mesh_density': 60}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, eps, _ = build_structure(cfg_local, cfg_m)
    res_loc = build_simulation(p, eps, {'structure': cfg_local,
            'simulation': {'type': 'ret', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]]}}).run(
                    np.array([600.0]))

    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'k_nm_inv': 0.0}}
    p2, eps2, _ = build_structure(cfg_nl, cfg_m)
    res_nl = build_simulation(p2, eps2, {'structure': cfg_nl,
            'simulation': {'type': 'ret', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]]}}).run(
                    np.array([600.0]))

    rel = abs(res_loc['ext'][0, 0] - res_nl['ext'][0, 0]) / abs(res_loc['ext'][0, 0])
    assert rel < 1e-12, \
            '[error] nonlocal(k=0) full sim rel diff {:.3e} != 0'.format(rel)


def test_make_hydrodynamic_drude_eps_callable() -> None:
    """Direct hydrodynamic Drude EpsFun smoke."""
    from pymnpbem_simulation.material import make_hydrodynamic_drude_eps

    eps = make_hydrodynamic_drude_eps(eps_inf = 10, wp_eV = 9.0, gamma_eV = 0.07,
            beta_m_s = 1.0e6, k_nm_inv = 0.0)
    e0, k0 = eps(600.0)
    assert np.isfinite(e0)
    assert np.isfinite(k0)

    # at k=0.05 should differ
    eps_k = make_hydrodynamic_drude_eps(eps_inf = 10, wp_eV = 9.0, gamma_eV = 0.07,
            beta_m_s = 1.0e6, k_nm_inv = 0.05)
    e_k, _ = eps_k(600.0)
    assert not np.isclose(e0, e_k)


if __name__ == '__main__':
    import pytest as _pt
    sys.exit(_pt.main([__file__, '-v']))
