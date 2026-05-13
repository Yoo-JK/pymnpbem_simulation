import sys

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _DummyParticle(object):

    def __init__(self,
            n: int = 0,
            cover_layer: bool = False) -> None:
        self.n = n
        if cover_layer:
            self._mnpbem_refun = lambda *a, **k: None


def _make_runner(cfg: Dict[str, Any],
        n_face: int = 0,
        cover_layer: bool = False) -> Any:
    from pymnpbem_simulation.simulation.base import SimulationRunner

    p = _DummyParticle(n = n_face, cover_layer = cover_layer)
    return SimulationRunner(cfg, p, epstab = [])


# ----------------------------------------------------------------------
# _resolve_preconditioner: 4 cases
# ----------------------------------------------------------------------


def test_preconditioner_auto() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({'preconditioner': 'auto'}, hmatrix_active = True)
    assert out['preconditioner'] == 'auto'
    assert 'htol_precond' not in out


def test_preconditioner_none() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({'preconditioner': 'none'}, hmatrix_active = True)
    assert out['preconditioner'] == 'none'


def test_preconditioner_hlu_dense() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({'preconditioner': 'hlu_dense'}, hmatrix_active = True)
    assert out['preconditioner'] == 'hlu_dense'


def test_preconditioner_hlu_tree() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({'preconditioner': 'hlu_tree'}, hmatrix_active = True)
    assert out['preconditioner'] == 'hlu_tree'


def test_preconditioner_htol_precond_forwarded() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner(
            {'preconditioner': 'auto', 'htol_precond': 1.0e-3},
            hmatrix_active = True)
    assert out['htol_precond'] == 1.0e-3


def test_preconditioner_missing_defaults_to_auto() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({}, hmatrix_active = True)
    assert out['preconditioner'] == 'auto'


def test_preconditioner_legacy_bool_true() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({'preconditioner': True}, hmatrix_active = True)
    assert out['preconditioner'] == 'auto'


def test_preconditioner_legacy_bool_false() -> None:
    runner = _make_runner({}, n_face = 100)
    out = runner._resolve_preconditioner({'preconditioner': False}, hmatrix_active = True)
    assert out['preconditioner'] == 'none'


# ----------------------------------------------------------------------
# _resolve_schur_iter
# ----------------------------------------------------------------------


def test_schur_iter_auto_no_cover_off() -> None:
    runner = _make_runner({}, n_face = 100, cover_layer = False)
    out = runner._resolve_schur_iter({'schur': 'auto'}, has_cover_layer = False)
    assert out == dict()


def test_schur_iter_auto_with_cover_on() -> None:
    runner = _make_runner({}, n_face = 100, cover_layer = True)
    out = runner._resolve_schur_iter({'schur': 'auto'}, has_cover_layer = True)
    assert out['schur'] is True
    assert out['schur_g_ss_solver'] == 'auto'


def test_schur_iter_explicit_true() -> None:
    runner = _make_runner({}, n_face = 100, cover_layer = False)
    out = runner._resolve_schur_iter({'schur': True}, has_cover_layer = False)
    assert out['schur'] is True


def test_schur_iter_explicit_false() -> None:
    runner = _make_runner({}, n_face = 100, cover_layer = True)
    out = runner._resolve_schur_iter({'schur': False}, has_cover_layer = True)
    assert out == dict()


def test_schur_iter_inner_knobs() -> None:
    runner = _make_runner({}, n_face = 100, cover_layer = True)
    out = runner._resolve_schur_iter(
            {'schur': True,
                    'schur_g_ss_solver': 'lu_dense',
                    'schur_inner_tol': 1.0e-10,
                    'schur_inner_maxit': 500},
            has_cover_layer = True)
    assert out['schur'] is True
    assert out['schur_g_ss_solver'] == 'lu_dense'
    assert out['schur_inner_tol'] == 1.0e-10
    assert out['schur_inner_maxit'] == 500


def test_schur_iter_missing_defaults_to_auto() -> None:
    runner_no_cover = _make_runner({}, n_face = 100, cover_layer = False)
    out = runner_no_cover._resolve_schur_iter({}, has_cover_layer = False)
    assert out == dict()

    runner_cover = _make_runner({}, n_face = 100, cover_layer = True)
    out = runner_cover._resolve_schur_iter({}, has_cover_layer = True)
    assert out.get('schur') is True


def test_has_cover_layer_detects_refun() -> None:
    runner_no = _make_runner({}, n_face = 100, cover_layer = False)
    assert runner_no._has_cover_layer() is False

    runner_yes = _make_runner({}, n_face = 100, cover_layer = True)
    assert runner_yes._has_cover_layer() is True


def test_has_cover_layer_detects_via_pfull() -> None:
    from pymnpbem_simulation.simulation.base import SimulationRunner

    class _Mirror(object):
        def __init__(self) -> None:
            self.pfull = _DummyParticle(n = 100, cover_layer = True)

    runner = SimulationRunner({}, _Mirror(), epstab = [])
    assert runner._has_cover_layer() is True


# ----------------------------------------------------------------------
# _iter_preconditioner_options + _iter_schur_options forwarding
# ----------------------------------------------------------------------


def test_iter_preconditioner_options_forwards_when_set() -> None:
    from pymnpbem_simulation.simulation.planewave_ret_iter import _iter_preconditioner_options

    cfg = {'simulation': {'iter': {
            'preconditioner': 'hlu_tree',
            'htol_precond': 1.0e-5}}}
    runner = _make_runner(cfg, n_face = 100)

    out = _iter_preconditioner_options(runner, cfg)
    assert out['preconditioner'] == 'hlu_tree'
    assert out['htol_precond'] == 1.0e-5


def test_iter_preconditioner_options_empty_when_not_set() -> None:
    from pymnpbem_simulation.simulation.planewave_ret_iter import _iter_preconditioner_options

    cfg = {'simulation': {'iter': {'tol': 1.0e-6}}}
    runner = _make_runner(cfg, n_face = 100)
    assert _iter_preconditioner_options(runner, cfg) == dict()


def test_iter_schur_options_active_with_cover() -> None:
    from pymnpbem_simulation.simulation.planewave_ret_iter import _iter_schur_options

    cfg = {'simulation': {'iter': {'schur': 'auto'}}}
    runner = _make_runner(cfg, n_face = 100, cover_layer = True)
    out = _iter_schur_options(runner, cfg)
    assert out.get('schur') is True


def test_iter_schur_options_explicit_true_no_cover() -> None:
    from pymnpbem_simulation.simulation.planewave_ret_iter import _iter_schur_options

    cfg = {'simulation': {'iter': {'schur': True}}}
    runner = _make_runner(cfg, n_face = 100, cover_layer = False)
    out = _iter_schur_options(runner, cfg)
    assert out.get('schur') is True


def test_iter_schur_options_empty_auto_no_cover() -> None:
    from pymnpbem_simulation.simulation.planewave_ret_iter import _iter_schur_options

    cfg = {'simulation': {'iter': {'schur': 'auto'}}}
    runner = _make_runner(cfg, n_face = 100, cover_layer = False)
    assert _iter_schur_options(runner, cfg) == dict()


# ----------------------------------------------------------------------
# _construct_bem fallback for v1.5.0 kwargs
# ----------------------------------------------------------------------


def test_construct_bem_strips_preconditioner_on_typeerror() -> None:
    """If BEMRetIter rejects 'preconditioner', fallback drops it (and
    htol_precond) and retries."""

    class _OldBEMIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            if 'preconditioner' in kwargs:
                raise TypeError(
                        "__init__() got an unexpected keyword argument 'preconditioner'")
            self.p = p
            self.kwargs = kwargs

    runner = _make_runner({'simulation': {'iter': {'preconditioner': 'auto'}}}, n_face = 100)
    bem = runner._construct_bem(_OldBEMIter, runner.p,
            preconditioner = 'auto', htol_precond = 1.0e-4,
            tol = 1.0e-6)

    assert 'preconditioner' not in bem.kwargs
    assert 'htol_precond' not in bem.kwargs
    assert bem.kwargs.get('tol') == 1.0e-6


def test_construct_bem_strips_schur_iter_on_typeerror() -> None:
    """If BEMRetIter rejects 'schur', fallback drops the entire schur
    family (schur, schur_g_ss_solver, schur_inner_*)."""

    class _OldBEMIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            # First reject schur_g_ss_solver, then on retry reject schur,
            # then accept (because all schur* kwargs gone).
            if 'schur_g_ss_solver' in kwargs:
                raise TypeError(
                        "__init__() got an unexpected keyword argument 'schur_g_ss_solver'")
            if 'schur' in kwargs:
                raise TypeError(
                        "__init__() got an unexpected keyword argument 'schur'")
            self.p = p
            self.kwargs = kwargs

    runner = _make_runner({}, n_face = 100)
    bem = runner._construct_bem(_OldBEMIter, runner.p,
            schur = True, schur_g_ss_solver = 'auto',
            schur_inner_tol = 1.0e-8, schur_inner_maxit = 200,
            tol = 1.0e-6)

    assert 'schur' not in bem.kwargs
    assert 'schur_g_ss_solver' not in bem.kwargs
    assert 'schur_inner_tol' not in bem.kwargs
    assert 'schur_inner_maxit' not in bem.kwargs
    assert bem.kwargs.get('tol') == 1.0e-6


def test_construct_bem_passes_v150_kwargs_when_supported() -> None:
    """Modern BEMRetIter accepts all v1.5.0 kwargs untouched."""

    class _NewBEMIter(object):
        def __init__(self,
                p: Any,
                preconditioner: str = 'auto',
                htol_precond: float = 1.0e-4,
                schur: bool = False,
                schur_g_ss_solver: str = 'auto',
                schur_inner_tol: float = 1.0e-8,
                **kwargs: Any) -> None:
            self.p = p
            self.preconditioner = preconditioner
            self.htol_precond = htol_precond
            self.schur = schur
            self.schur_g_ss_solver = schur_g_ss_solver
            self.schur_inner_tol = schur_inner_tol
            self.kwargs = kwargs

    runner = _make_runner({}, n_face = 100)
    bem = runner._construct_bem(_NewBEMIter, runner.p,
            preconditioner = 'hlu_tree', htol_precond = 1.0e-5,
            schur = True, schur_g_ss_solver = 'lu_dense',
            schur_inner_tol = 1.0e-10,
            tol = 1.0e-6)

    assert bem.preconditioner == 'hlu_tree'
    assert bem.htol_precond == 1.0e-5
    assert bem.schur is True
    assert bem.schur_g_ss_solver == 'lu_dense'
    assert bem.schur_inner_tol == 1.0e-10


def test_construct_bem_propagates_unrelated_typeerror_v150() -> None:
    class _PickyBEM(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            raise TypeError('completely unrelated init failure')

    runner = _make_runner({}, n_face = 100)
    with pytest.raises(TypeError, match = 'unrelated'):
        runner._construct_bem(_PickyBEM, runner.p, preconditioner = 'auto')


# ----------------------------------------------------------------------
# Iter runner build_solver wires v1.5.0 kwargs through
# ----------------------------------------------------------------------


def test_ret_iter_runner_passes_preconditioner(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = dict()

    class _SpyBEMRetIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            captured.update(kwargs)
            self.p = p

        def solve(self,
                exc: Any) -> Any:
            return (None, self)

    import mnpbem.bem as bem_mod
    monkeypatch.setattr(bem_mod, 'BEMRetIter', _SpyBEMRetIter)

    from pymnpbem_simulation.simulation.planewave_ret_iter import PlaneWaveRetIterRunner

    cfg = {
            'simulation': {
                    'type': 'ret_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                    'iter': {'tol': 1.0e-6, 'maxit': 200, 'precond': 'hmat',
                            'preconditioner': 'hlu_tree',
                            'htol_precond': 1.0e-5}}}

    runner = PlaneWaveRetIterRunner(cfg, _DummyParticle(n = 100), epstab = [])
    runner.build_solver()

    assert captured.get('preconditioner') == 'hlu_tree'
    assert captured.get('htol_precond') == 1.0e-5


def test_ret_iter_runner_skips_preconditioner_when_unset(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = dict()

    class _SpyBEMRetIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            captured.update(kwargs)
            self.p = p

    import mnpbem.bem as bem_mod
    monkeypatch.setattr(bem_mod, 'BEMRetIter', _SpyBEMRetIter)

    from pymnpbem_simulation.simulation.planewave_ret_iter import PlaneWaveRetIterRunner

    cfg = {
            'simulation': {
                    'type': 'ret_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                    'iter': {'hmatrix': 'auto'}}}

    runner = PlaneWaveRetIterRunner(cfg, _DummyParticle(n = 100), epstab = [])
    runner.build_solver()

    assert 'preconditioner' not in captured
    assert 'htol_precond' not in captured


def test_ret_iter_runner_passes_schur_iter_with_cover(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = dict()

    class _SpyBEMRetIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            captured.update(kwargs)
            self.p = p

    import mnpbem.bem as bem_mod
    monkeypatch.setattr(bem_mod, 'BEMRetIter', _SpyBEMRetIter)

    from pymnpbem_simulation.simulation.planewave_ret_iter import PlaneWaveRetIterRunner

    cfg = {
            'simulation': {
                    'type': 'ret_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                    'iter': {'schur': 'auto'}}}

    p = _DummyParticle(n = 100, cover_layer = True)
    runner = PlaneWaveRetIterRunner(cfg, p, epstab = [])
    runner.build_solver()

    assert captured.get('schur') is True


def test_ret_iter_runner_skips_schur_iter_no_cover_auto(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = dict()

    class _SpyBEMRetIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            captured.update(kwargs)
            self.p = p

    import mnpbem.bem as bem_mod
    monkeypatch.setattr(bem_mod, 'BEMRetIter', _SpyBEMRetIter)

    from pymnpbem_simulation.simulation.planewave_ret_iter import PlaneWaveRetIterRunner

    cfg = {
            'simulation': {
                    'type': 'ret_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                    'iter': {'schur': 'auto'}}}

    runner = PlaneWaveRetIterRunner(cfg, _DummyParticle(n = 100), epstab = [])
    runner.build_solver()

    assert 'schur' not in captured


def test_stat_iter_runner_passes_v150_options(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = dict()

    class _SpyBEMStatIter(object):
        def __init__(self,
                p: Any,
                **kwargs: Any) -> None:
            captured.update(kwargs)
            self.p = p

    import mnpbem.bem as bem_mod
    monkeypatch.setattr(bem_mod, 'BEMStatIter', _SpyBEMStatIter)

    from pymnpbem_simulation.simulation.planewave_stat_iter import PlaneWaveStatIterRunner

    cfg = {
            'simulation': {
                    'type': 'stat_iter', 'excitation': 'planewave',
                    'polarizations': [[1, 0, 0]],
                    'iter': {'preconditioner': 'hlu_dense',
                            'schur': True,
                            'schur_g_ss_solver': 'lu_dense'}}}

    runner = PlaneWaveStatIterRunner(cfg, _DummyParticle(n = 100), epstab = [])
    runner.build_solver()

    assert captured.get('preconditioner') == 'hlu_dense'
    assert captured.get('schur') is True
    assert captured.get('schur_g_ss_solver') == 'lu_dense'


# ----------------------------------------------------------------------
# Example YAML schema sanity
# ----------------------------------------------------------------------


def test_dimer_iter_precond_yaml_loads() -> None:
    import yaml

    path = REPO_ROOT / 'examples' / 'dimer_iter_precond.yaml'
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    assert cfg['simulation']['type'] == 'ret_iter'
    iter_cfg = cfg['simulation']['iter']
    assert iter_cfg.get('preconditioner') in (
            'auto', 'none', 'hlu_dense', 'hlu_tree')
    assert iter_cfg.get('hmatrix') is True


def test_nonlocal_iter_schur_yaml_loads() -> None:
    import yaml

    path = REPO_ROOT / 'examples' / 'nonlocal_iter_schur.yaml'
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    assert cfg['simulation']['type'] == 'ret_iter'
    iter_cfg = cfg['simulation']['iter']
    assert iter_cfg.get('schur') in ('auto', True, False, 'true', 'false')
    assert 'preconditioner' in iter_cfg


# ----------------------------------------------------------------------
# Physics smoke (small mesh — preconditioner ON vs OFF)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_planewave_ret_iter_preconditioner_smoke() -> None:
    import inspect

    try:
        from mnpbem.bem import BEMRetIter
    except ImportError:
        pytest.skip('mnpbem.bem.BEMRetIter unavailable')

    sig = inspect.signature(BEMRetIter.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values())
    if 'preconditioner' not in sig.parameters and not has_var_kw:
        pytest.skip(
                'BEMRetIter does not accept preconditioner= kwarg yet (Agent α not merged)')

    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_struct = {'type': 'sphere', 'diameter': 30, 'mesh_density': 60}
    cfg_m = {'medium': 'water', 'particle': 'gold'}
    enei = np.array([550.0, 600.0])

    p1, eps1, _ = build_structure(cfg_struct, cfg_m)
    res_off = build_simulation(p1, eps1,
            {'structure': cfg_struct,
                    'simulation': {'type': 'ret_iter', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                            'iter': {'tol': 1.0e-8, 'maxit': 500, 'precond': 'hmat',
                                    'hmatrix': True, 'htol': 1.0e-8,
                                    'preconditioner': 'none'}}}).run(enei)

    p2, eps2, _ = build_structure(cfg_struct, cfg_m)
    res_on = build_simulation(p2, eps2,
            {'structure': cfg_struct,
                    'simulation': {'type': 'ret_iter', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]], 'propagation_dirs': [[0, 0, 1]],
                            'iter': {'tol': 1.0e-8, 'maxit': 500, 'precond': 'hmat',
                                    'hmatrix': True, 'htol': 1.0e-8,
                                    'preconditioner': 'auto',
                                    'htol_precond': 1.0e-4}}}).run(enei)

    rel = np.max(np.abs(res_on['ext'] - res_off['ext'])
            / np.maximum(np.abs(res_off['ext']), 1e-30))
    assert rel < 1e-4, \
            '[error] preconditioner ON vs OFF rel diff {:.3e} > 1e-4'.format(rel)


@pytest.mark.slow
def test_planewave_stat_iter_schur_on_off_matches() -> None:
    """Cover-layer nonlocal sphere: schur ON vs OFF must give identical
    spectra (Schur is an exact reduction, not an approximation)."""
    import inspect

    try:
        from mnpbem.bem import BEMStatIter
    except ImportError:
        pytest.skip('mnpbem.bem.BEMStatIter unavailable')

    sig = inspect.signature(BEMStatIter.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values())
    if 'schur' not in sig.parameters and not has_var_kw:
        pytest.skip(
                'BEMStatIter does not accept schur= kwarg yet (Agent β not merged)')

    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_struct = {
            'type': 'with_nonlocal',
            'base': {'type': 'sphere', 'diameter': 10, 'mesh_density': 60},
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05,
                    'beta': None, 'eps_embed': 1.0}}
    cfg_m = {'medium': 'vacuum', 'particle': 'gold'}
    enei = np.array([550.0])

    p1, eps1, _ = build_structure(cfg_struct, cfg_m)
    res_off = build_simulation(p1, eps1,
            {'structure': cfg_struct,
                    'simulation': {'type': 'stat_iter', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]],
                            'iter': {'tol': 1.0e-8, 'maxit': 500, 'precond': 'hmat',
                                    'hmatrix': False, 'schur': False}}}).run(enei)

    p2, eps2, _ = build_structure(cfg_struct, cfg_m)
    res_on = build_simulation(p2, eps2,
            {'structure': cfg_struct,
                    'simulation': {'type': 'stat_iter', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]],
                            'iter': {'tol': 1.0e-8, 'maxit': 500, 'precond': 'hmat',
                                    'hmatrix': False, 'schur': True}}}).run(enei)

    rel = np.max(np.abs(res_on['ext'] - res_off['ext'])
            / np.maximum(np.abs(res_off['ext']), 1e-30))
    assert rel < 1e-6, \
            '[error] schur ON vs OFF rel diff {:.3e} > 1e-6'.format(rel)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
