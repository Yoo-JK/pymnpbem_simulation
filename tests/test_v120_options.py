"""v1.2.0 wrapper-level option tests (Agent γ).

Covers
------
* ``compute.schur_complement`` ON / OFF / auto resolution in
  ``SimulationRunner._bem_options``.
* ``compute.n_gpus_per_worker > 1`` activates VRAM-share dispatch and
  propagates ``MNPBEM_VRAM_SHARE_*`` env vars to the worker.
* Graceful fallback when the installed mnpbem port has not yet adopted
  the v1.2.0 BEMStat ``schur=`` kwarg (Agent α not merged) — the
  wrapper retries without it instead of raising.
"""

import os
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
    """Minimal stand-in for ComParticle with optional cover-layer flag."""

    def __init__(self, refun: Any = None) -> None:
        if refun is not None:
            self._mnpbem_refun = refun


def _make_runner(cfg: Dict[str, Any], refun: Any = None) -> Any:
    from pymnpbem_simulation.simulation.base import SimulationRunner

    p = _DummyParticle(refun = refun)
    return SimulationRunner(cfg, p, epstab = [])


# ----------------------------------------------------------------------
# Schur option resolution
# ----------------------------------------------------------------------


def test_schur_auto_no_cover_layer_omits_kwarg() -> None:
    """auto + no cover layer -> 'schur' key is omitted entirely.

    Important so that pre-v1.2.0 mnpbem builds (no schur kwarg in
    BEMStat) keep working unchanged for non-nonlocal simulations.
    """
    cfg = {'compute': {'schur_complement': 'auto'}}
    opts = _make_runner(cfg, refun = None)._bem_options()

    assert 'schur' not in opts
    assert 'refun' not in opts


def test_schur_auto_with_cover_layer_turns_on() -> None:
    """auto + nonlocal cover layer (refun present) -> schur=True."""
    refun = lambda obj, G, F: (G, F)
    cfg = {'compute': {'schur_complement': 'auto'}}
    opts = _make_runner(cfg, refun = refun)._bem_options()

    assert opts.get('schur') is True
    assert opts.get('refun') is refun


def test_schur_explicit_true_forces_on_without_cover_layer() -> None:
    """explicit true overrides auto and turns Schur on regardless."""
    cfg = {'compute': {'schur_complement': True}}
    opts = _make_runner(cfg, refun = None)._bem_options()

    assert opts.get('schur') is True


def test_schur_explicit_false_disables_even_with_cover_layer() -> None:
    """explicit false overrides auto and disables Schur even when nonlocal."""
    refun = lambda obj, G, F: (G, F)
    cfg = {'compute': {'schur_complement': False}}
    opts = _make_runner(cfg, refun = refun)._bem_options()

    assert opts.get('schur') is False
    assert opts.get('refun') is refun


def test_schur_string_true_false_parsed() -> None:
    """YAML loaders sometimes leave the value as the string 'true'/'false'."""
    refun = lambda obj, G, F: (G, F)

    cfg_t = {'compute': {'schur_complement': 'true'}}
    assert _make_runner(cfg_t, refun = None)._bem_options().get('schur') is True

    cfg_f = {'compute': {'schur_complement': 'false'}}
    assert _make_runner(cfg_f, refun = refun)._bem_options().get('schur') is False


def test_schur_default_is_auto() -> None:
    """Missing key defaults to auto."""
    refun = lambda obj, G, F: (G, F)

    cfg_no_compute: Dict[str, Any] = dict()
    opts_no_cover = _make_runner(cfg_no_compute, refun = None)._bem_options()
    assert 'schur' not in opts_no_cover

    opts_cover = _make_runner(cfg_no_compute, refun = refun)._bem_options()
    assert opts_cover.get('schur') is True


# ----------------------------------------------------------------------
# _construct_bem fallback (Agent α not yet merged)
# ----------------------------------------------------------------------


def test_construct_bem_strips_unknown_schur_on_typeerror() -> None:
    """If BEMStat raises TypeError mentioning 'schur', helper retries
    without that kwarg."""
    from pymnpbem_simulation.simulation.base import SimulationRunner

    class _OldBEM(object):

        def __init__(self, p: Any, **kwargs: Any) -> None:
            if 'schur' in kwargs:
                raise TypeError(
                        "__init__() got an unexpected keyword argument 'schur'")
            self.p = p
            self.kwargs = kwargs

    cfg = {'compute': {'schur_complement': True}}
    runner = _make_runner(cfg, refun = None)
    bem = runner._construct_bem(_OldBEM, runner.p, schur = True, foo = 'bar')

    assert bem.kwargs == {'foo': 'bar'}
    assert 'schur' not in bem.kwargs


def test_construct_bem_propagates_unrelated_typeerror() -> None:
    """Helper must NOT swallow TypeErrors that have nothing to do with v1.2.0."""
    from pymnpbem_simulation.simulation.base import SimulationRunner

    class _PickyBEM(object):

        def __init__(self, p: Any, **kwargs: Any) -> None:
            raise TypeError("unrelated failure")

    runner = _make_runner({'compute': {}}, refun = None)
    with pytest.raises(TypeError, match = 'unrelated'):
        runner._construct_bem(_PickyBEM, runner.p, schur = True)


def test_construct_bem_passes_schur_when_supported() -> None:
    """If BEMStat accepts schur=, helper does NOT strip it."""
    from pymnpbem_simulation.simulation.base import SimulationRunner

    class _NewBEM(object):

        def __init__(self, p: Any, schur: bool = False, **kwargs: Any) -> None:
            self.p = p
            self.schur = schur

    runner = _make_runner({'compute': {}}, refun = None)
    bem = runner._construct_bem(_NewBEM, runner.p, schur = True)

    assert bem.schur is True


# ----------------------------------------------------------------------
# Schur ON vs OFF on a real (small) nonlocal sphere — physics regression.
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_schur_on_off_matches_on_small_nonlocal_sphere() -> None:
    """Schur enabled vs disabled must give the same extinction (within
    numerical tolerance) on the same nonlocal cover-layer sphere.

    Skipped if Agent α has not landed (BEMStat raises on unknown
    schur=); we only assert *consistency*, not Schur correctness.
    """
    from pymnpbem_simulation.structures import build_structure
    from pymnpbem_simulation.simulation import build_simulation

    cfg_local = {'type': 'sphere', 'diameter': 10, 'mesh_density': 60}
    cfg_nl = {'type': 'with_nonlocal', 'base': cfg_local,
            'nonlocal': {'metal': 'gold', 'delta_d': 0.05}}
    cfg_m = {'medium': 'vacuum', 'particle': 'gold'}
    enei = np.array([550.0, 600.0, 650.0])

    p_off, eps_off, _ = build_structure(cfg_nl, cfg_m)
    res_off = build_simulation(p_off, eps_off,
            {'structure': cfg_nl,
                    'compute': {'schur_complement': False},
                    'simulation': {'type': 'stat', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]]}}).run(enei)

    p_on, eps_on, _ = build_structure(cfg_nl, cfg_m)
    res_on = build_simulation(p_on, eps_on,
            {'structure': cfg_nl,
                    'compute': {'schur_complement': True},
                    'simulation': {'type': 'stat', 'excitation': 'planewave',
                            'polarizations': [[1, 0, 0]]}}).run(enei)

    # If Agent α has not merged yet both runs go through the same
    # (no-schur) code path and trivially match.
    rel = np.max(np.abs(res_on['ext'] - res_off['ext'])
            / np.maximum(np.abs(res_off['ext']), 1e-30))
    assert rel < 1e-6, \
            '[error] schur ON vs OFF rel diff {:.3e} > 1e-6'.format(rel)


# ----------------------------------------------------------------------
# VRAM share env-var propagation
# ----------------------------------------------------------------------


def test_vram_share_dispatch_sets_env_vars(monkeypatch: Any) -> None:
    """``_dispatch_vram_share`` must set the MNPBEM_VRAM_SHARE_* env
    vars before delegating to ``_dispatch_single_gpu``."""
    captured: Dict[str, str] = dict()

    def _fake_single_gpu(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        captured['MNPBEM_VRAM_SHARE'] = os.environ.get('MNPBEM_VRAM_SHARE', '')
        captured['MNPBEM_VRAM_SHARE_GPUS'] = os.environ.get(
                'MNPBEM_VRAM_SHARE_GPUS', '')
        captured['MNPBEM_VRAM_SHARE_BACKEND'] = os.environ.get(
                'MNPBEM_VRAM_SHARE_BACKEND', '')
        return {'ok': True}

    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.single_node._dispatch_single_gpu',
            _fake_single_gpu)

    # also clear any pre-existing state
    for k in ('MNPBEM_VRAM_SHARE',
            'MNPBEM_VRAM_SHARE_GPUS',
            'MNPBEM_VRAM_SHARE_BACKEND'):
        monkeypatch.delenv(k, raising = False)

    from pymnpbem_simulation.dispatch.multi_gpu import _dispatch_vram_share

    cfg = {'compute': {'n_gpus_per_worker': 4,
            'vram_share_backend': 'magma'}}
    out = _dispatch_vram_share(cfg, p = None, epstab = None,
            enei = np.array([600.0]))

    assert out == {'ok': True}
    assert captured['MNPBEM_VRAM_SHARE'] == '1'
    assert captured['MNPBEM_VRAM_SHARE_GPUS'] == '4'
    assert captured['MNPBEM_VRAM_SHARE_BACKEND'] == 'magma'


def test_vram_share_dispatch_cleans_env_vars(monkeypatch: Any) -> None:
    """After ``_dispatch_vram_share`` returns, the env vars must be
    cleared so subsequent dispatches in the same process see a clean
    slate."""

    def _fake_single_gpu(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        return {'ok': True}

    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.single_node._dispatch_single_gpu',
            _fake_single_gpu)
    for k in ('MNPBEM_VRAM_SHARE',
            'MNPBEM_VRAM_SHARE_GPUS',
            'MNPBEM_VRAM_SHARE_BACKEND'):
        monkeypatch.delenv(k, raising = False)

    from pymnpbem_simulation.dispatch.multi_gpu import _dispatch_vram_share

    cfg = {'compute': {'n_gpus_per_worker': 2}}
    _dispatch_vram_share(cfg, None, None, np.array([550.0]))

    assert 'MNPBEM_VRAM_SHARE' not in os.environ
    assert 'MNPBEM_VRAM_SHARE_GPUS' not in os.environ
    assert 'MNPBEM_VRAM_SHARE_BACKEND' not in os.environ


def test_vram_share_default_backend_cusolvermg(monkeypatch: Any) -> None:
    """Default backend is cusolvermg when not specified in cfg."""
    captured: Dict[str, str] = dict()

    def _fake_single_gpu(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        captured['backend'] = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', '')
        return {'ok': True}

    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.single_node._dispatch_single_gpu',
            _fake_single_gpu)

    from pymnpbem_simulation.dispatch.multi_gpu import _dispatch_vram_share

    cfg = {'compute': {'n_gpus_per_worker': 2}}
    _dispatch_vram_share(cfg, None, None, np.array([550.0]))

    assert captured['backend'] == 'cusolvermg'


def test_multi_gpu_routes_n_gpus_gt_one_to_vram_share(monkeypatch: Any) -> None:
    """``dispatch_multi_gpu`` with n_gpus_per_worker > 1 must enter the
    VRAM-share path, NOT wavelength-split."""
    called: Dict[str, bool] = {'vram': False, 'split': False}

    def _fake_vram(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        called['vram'] = True
        return {'kind': 'vram_share'}

    def _fake_split(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        called['split'] = True
        return {'kind': 'split'}

    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.multi_gpu._dispatch_vram_share',
            _fake_vram)
    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.multi_gpu._dispatch_wavelength_split',
            _fake_split)

    from pymnpbem_simulation.dispatch.multi_gpu import dispatch_multi_gpu

    cfg = {'compute': {'n_workers': 1, 'n_gpus_per_worker': 4}}
    out = dispatch_multi_gpu(cfg, None, None, np.array([600.0]))

    assert called['vram'] is True
    assert called['split'] is False
    assert out['kind'] == 'vram_share'


def test_multi_gpu_routes_n_gpus_one_to_split(monkeypatch: Any) -> None:
    """n_gpus_per_worker == 1 with multiple workers stays on
    wavelength-split (the legacy multi-GPU path)."""
    called: Dict[str, bool] = {'vram': False, 'split': False}

    def _fake_vram(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        called['vram'] = True
        return {'kind': 'vram_share'}

    def _fake_split(cfg: Dict[str, Any],
            p: Any,
            epstab: Any,
            enei: np.ndarray) -> Dict[str, Any]:
        called['split'] = True
        return {'kind': 'split'}

    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.multi_gpu._dispatch_vram_share',
            _fake_vram)
    monkeypatch.setattr(
            'pymnpbem_simulation.dispatch.multi_gpu._dispatch_wavelength_split',
            _fake_split)

    from pymnpbem_simulation.dispatch.multi_gpu import dispatch_multi_gpu

    cfg = {'compute': {'n_workers': 4, 'n_gpus_per_worker': 1}}
    out = dispatch_multi_gpu(cfg, None, None, np.array([600.0]))

    assert called['split'] is True
    assert called['vram'] is False
    assert out['kind'] == 'split'


# ----------------------------------------------------------------------
# Example YAML schema sanity
# ----------------------------------------------------------------------


def test_dimer_nonlocal_schur_yaml_loads() -> None:
    """The new Schur example parses and exposes compute.schur_complement."""
    import yaml

    path = REPO_ROOT / 'examples' / 'dimer_nonlocal_schur.yaml'
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    assert cfg['compute']['schur_complement'] in ('auto', True, False, 'true', 'false')
    assert cfg['structure']['type'] == 'with_nonlocal'


def test_large_mesh_vram_share_yaml_loads() -> None:
    """The new VRAM-share example parses and triggers the VRAM-share path."""
    import yaml

    path = REPO_ROOT / 'examples' / 'large_mesh_vram_share.yaml'
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    assert int(cfg['compute']['n_gpus_per_worker']) > 1
    assert cfg['compute']['vram_share_backend'] in (
            'cusolvermg', 'magma', 'nccl')


def test_dimer_vram_share_yaml_loads() -> None:
    """The reused M2 example still parses and now triggers VRAM share."""
    import yaml

    path = REPO_ROOT / 'examples' / 'dimer_vram_share.yaml'
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    assert int(cfg['compute']['n_gpus_per_worker']) > 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
