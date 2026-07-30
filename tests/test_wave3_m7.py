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
# Nonlocal cover-layer (EpsNonlocal port, Wave 2)
# ============================================================================

def test_with_nonlocal_in_registry() -> None:
    from pymnpbem_simulation.structures import REGISTRY

    assert 'with_nonlocal' in REGISTRY


def test_nonlocal_epstab_three_entries() -> None:
    """epstab = [eps_embed, eps_metal_core, eps_nonlocal_shell]."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, epstab, _ = build_structure(cfg_nl, cfg_m)

    assert len(epstab) == 3, \
            '[error] expected 3 epstab entries, got {}'.format(len(epstab))
    for entry in epstab:
        assert callable(entry)


def test_nonlocal_shell_eps_is_small() -> None:
    """eps_shell is the artificial cover-layer permittivity (small magnitude)."""
    from pymnpbem_simulation.structures import build_structure
    from mnpbem.materials import EpsNonlocal

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    _p, epstab, _ = build_structure(cfg_nl, cfg_m)

    assert isinstance(epstab[2], EpsNonlocal)

    val_shell, _ = epstab[2](600.0)
    # Yu Luo cover-layer eps_t is typically O(0.1) complex at vis frequencies.
    assert abs(val_shell) > 0.0
    assert abs(val_shell) < 5.0


def test_nonlocal_attaches_refun_on_particle() -> None:
    """coverlayer.refine refun is stashed on the ComParticle so simulation
    runners can forward it to BEMStat."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, _epstab, _ = build_structure(cfg_nl, cfg_m)

    refun = getattr(p, '_mnpbem_refun', None)
    assert refun is not None, '[error] _mnpbem_refun not attached'
    assert callable(refun)


def test_nonlocal_two_subparticles_per_base() -> None:
    """Each base sub-particle yields shell + core (2x sub-particles in
    final ComParticle)."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'water', 'particle': 'gold'}

    p, _epstab, _ = build_structure(cfg_nl, cfg_m)

    assert len(p.p) == 2, \
            '[error] expected 2 sub-particles (shell + core), got {}'.format(len(p.p))


def test_nonlocal_dimer_two_pairs() -> None:
    """dimer base -> 4 sub-particles (2 shells + 2 cores)."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'dimer_sphere', 'diameter': 10, 'gap': 1.0,
            'n_verts': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'vacuum', 'particle': 'gold'}

    p, _epstab, _ = build_structure(cfg_nl, cfg_m)

    assert len(p.p) == 4, \
            '[error] dimer nonlocal: expected 4 sub-particles, got {}'.format(len(p.p))


def test_nonlocal_runs_planewave_stat_smoke() -> None:
    """Full BEMStat run on a small nonlocal sphere produces finite + positive ext."""
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'vacuum', 'particle': 'gold'}

    p, eps, _ = build_structure(cfg_nl, cfg_m)
    cfg = {'structure': cfg_nl,
            'simulation': {'type': 'stat', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]]}}
    res = build_simulation(p, eps, cfg).run(np.array([550.0, 600.0, 650.0]))

    assert res['ext'].shape == (3, 1)
    assert np.all(np.isfinite(res['ext']))
    assert np.all(res['ext'] > 0)


def test_nonlocal_dimer_nano_gap_blueshift() -> None:
    """Sub-nm Au dimer gap: nonlocal cover layer must blueshift the bonding
    plasmon peak relative to a purely local Drude dimer.

    Reference: Ciraci et al., Science 337, 1072 (2012); Luo et al., PRL 111,
    093901 (2013). The peak shift increases as the gap shrinks; for a 1 nm
    gap with 10 nm Au spheres the shift is on the order of several to tens
    of nm. We only require a strict (positive) blueshift.
    """
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_dimer_local = {'type': 'dimer_sphere', 'diameter': 10, 'gap': 1.0,
            'n_verts': 144}
    cfg_dimer_nl = {'type': 'with_nonlocal', 'base': cfg_dimer_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'vacuum', 'particle': 'gold'}

    enei = np.linspace(450.0, 750.0, 31)

    p_loc, eps_loc, _ = build_structure(cfg_dimer_local, cfg_m)
    res_loc = build_simulation(p_loc, eps_loc,
            {'structure': cfg_dimer_local,
                    'simulation': {'type': 'stat', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]]}}).run(enei)

    p_nl, eps_nl, _ = build_structure(cfg_dimer_nl, cfg_m)
    res_nl = build_simulation(p_nl, eps_nl,
            {'structure': cfg_dimer_nl,
                    'simulation': {'type': 'stat', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]]}}).run(enei)

    peak_loc = float(enei[int(np.argmax(res_loc['ext'][:, 0]))])
    peak_nl = float(enei[int(np.argmax(res_nl['ext'][:, 0]))])

    blueshift_nm = peak_loc - peak_nl
    print('[info] dimer nano-gap nonlocal blueshift: peak_loc={:.2f}nm '
            'peak_nl={:.2f}nm shift={:.2f}nm'.format(
                    peak_loc, peak_nl, blueshift_nm))

    msg = ('[error] expected blueshift > 0, got {:.2f} nm '
            '(peak_local={:.2f}, peak_nonlocal={:.2f})').format(
                    blueshift_nm, peak_loc, peak_nl)
    assert blueshift_nm > 0.0, msg


def test_nonlocal_silver_factory() -> None:
    """silver metal name routes through make_nonlocal_pair (no Fermi velocity error)."""
    from pymnpbem_simulation.structures import build_structure

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'silver', 'delta_d': 0.05}}
    cfg_m = {'medium': 'vacuum', 'particle': 'silver'}

    _p, epstab, _ = build_structure(cfg_nl, cfg_m)
    val, _ = epstab[2](500.0)
    assert np.isfinite(val) and abs(val) > 0.0


def test_nonlocal_builtin_metals_unchanged() -> None:
    """Default metal paths must stay bit-identical to mnpbem make_nonlocal_pair."""
    from pymnpbem_simulation.material.nonlocal_eps import build_nonlocal_eps
    from mnpbem.materials import make_nonlocal_pair, EpsConst

    wl = np.array([500.0, 636.0, 800.0])

    for metal in ('gold', 'silver', 'aluminum', 'au', 'ag', 'al'):
        _eb, core, shell = build_nonlocal_eps({'metal': metal})
        core_ref, shell_ref = make_nonlocal_pair(metal,
                eps_embed = EpsConst(1.0), delta_d = 0.05, beta = None)
        assert np.array_equal(np.asarray(core(wl)[0]), np.asarray(core_ref(wl)[0])), metal
        assert np.array_equal(np.asarray(shell(wl)[0]), np.asarray(shell_ref(wl)[0])), metal


def test_nonlocal_copper_preset() -> None:
    """Copper has no mnpbem EpsDrude factory; the built-in preset covers it."""
    from pymnpbem_simulation.material.nonlocal_eps import build_nonlocal_eps

    for metal in ('copper', 'cu'):
        _eb, core, shell = build_nonlocal_eps({'metal': metal})
        eps_core, _ = core(600.0)
        eps_shell, _ = shell(600.0)
        assert np.isfinite(eps_core) and eps_core.real < 0.0
        assert np.isfinite(eps_shell) and abs(eps_shell) > 0.0


def test_nonlocal_drude_params_override() -> None:
    """Flat and material-keyed drude_params forms give the same result."""
    from pymnpbem_simulation.material.nonlocal_eps import build_nonlocal_eps

    params = {'omega_p': 9.02, 'gamma': 0.071, 'eps_inf': 9.84, 'v_f': 1.39e6}

    _eb, _c, flat = build_nonlocal_eps({'metal': 'gold', 'drude_params': params})
    _eb, _c, keyed = build_nonlocal_eps({'metal': 'au',
            'drude_params': {'gold': params}})
    _eb, _c, default = build_nonlocal_eps({'metal': 'gold'})

    assert np.array_equal(np.asarray(flat(600.0)[0]), np.asarray(keyed(600.0)[0]))
    assert not np.array_equal(np.asarray(flat(600.0)[0]), np.asarray(default(600.0)[0]))

    # A key for a different metal must not hijack this build.
    _eb, _c, unmatched = build_nonlocal_eps({'metal': 'gold',
            'drude_params': {'silver': {'omega_p': 9.17}}})
    assert np.array_equal(np.asarray(unmatched(600.0)[0]), np.asarray(default(600.0)[0]))


def test_nonlocal_qcm_falls_back_to_hydrodynamic() -> None:
    """'qcm' is accepted for MATLAB-wrapper parity but is not implemented."""
    from pymnpbem_simulation.material.nonlocal_eps import build_nonlocal_eps

    _eb, _c, qcm = build_nonlocal_eps({'metal': 'gold', 'model': 'qcm'})
    _eb, _c, hydro = build_nonlocal_eps({'metal': 'gold'})

    assert np.array_equal(np.asarray(qcm(600.0)[0]), np.asarray(hydro(600.0)[0]))


def test_nonlocal_invalid_specs_raise() -> None:
    from pymnpbem_simulation.material.nonlocal_eps import build_nonlocal_eps

    for spec in ({'metal': 'gold', 'model': 'bogus'}, {'metal': 'platinum'}):
        with pytest.raises(ValueError):
            build_nonlocal_eps(spec)


def test_auto_wrap_nonlocal_forwards_landes_keys() -> None:
    """compute.nonlocal + materials.nonlocal -> structure.with_nonlocal spec."""
    from pymnpbem_simulation.config import _auto_wrap_nonlocal

    cfg = {'compute': {'nonlocal': True},
            'structure': {'type': 'dimer_cube', 'size': 47, 'gap': 0.6},
            'materials': {'particle_list': ['gold'],
                    'nonlocal': {'model': 'qcm', 'delta_d': 0.08,
                            'drude_params': {'gold': {'omega_p': 9.02}}}}}

    out = _auto_wrap_nonlocal(cfg)
    spec = out['structure']['nonlocal']

    assert out['structure']['type'] == 'with_nonlocal'
    assert out['structure']['base']['type'] == 'dimer_cube'
    assert spec['model'] == 'qcm'
    assert spec['delta_d'] == pytest.approx(0.08)
    assert spec['drude_params']['gold']['omega_p'] == pytest.approx(9.02)

    # Defaults stay hydrodynamic, and the wrap is a no-op when disabled.
    minimal = _auto_wrap_nonlocal({'compute': {'nonlocal': True},
            'structure': {'type': 'sphere', 'diameter': 10},
            'materials': {'particle_list': ['silver']}})
    assert minimal['structure']['nonlocal']['model'] == 'hydrodynamic'

    off = _auto_wrap_nonlocal({'compute': {'nonlocal': False},
            'structure': {'type': 'sphere'}})
    assert off['structure']['type'] == 'sphere'


if __name__ == '__main__':
    import pytest as _pt
    sys.exit(_pt.main([__file__, '-v']))
