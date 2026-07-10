from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
import runpy
import sys
import types

import pytest

from pymnpbem_simulation.config import _resolve_substrate_eps
from pymnpbem_simulation.config import merge_str_sim_args
from pymnpbem_simulation.material.material_descriptor import resolve_refractive_index_paths
from pymnpbem_simulation.structures import build_structure
from pymnpbem_simulation.structures.sphere import _build_eps_particle


def _has_real_mnpbem() -> bool:
    return importlib.util.find_spec('mnpbem.materials') is not None \
        and importlib.util.find_spec('mnpbem.geometry') is not None


def test_resolve_refractive_index_paths_constant_and_table() -> None:
    ri_paths = {
        'agcl': {'type': 'constant', 'epsilon': '2.02'},
        'ito': {'type': 'table', 'file': '/tmp/ito.dat'},
        'legacy': {'file': '/tmp/legacy.dat'},
        'plain': 1.77,
    }

    out = resolve_refractive_index_paths(ri_paths)

    assert out['agcl'] == {'type': 'constant', 'epsilon': 2.02}
    assert out['ito'] == '/tmp/ito.dat'
    assert out['legacy'] == {'file': '/tmp/legacy.dat'}
    assert out['plain'] == 1.77


def test_resolve_refractive_index_paths_python_module(tmp_path) -> None:
    module_path = tmp_path / 'custom_eps.py'
    module_path.write_text(
        'def generate_eps_func():\n'
        '    def _eps(enei):\n'
        '        return 1.0 + 0j\n'
        '    return _eps\n',
        encoding = 'utf-8')

    out = resolve_refractive_index_paths({
        'user_mat': {
            'type': 'python_module',
            'module_path': str(module_path),
            'factory': 'generate_eps_func'}})

    assert callable(out['user_mat'])
    assert out['user_mat'](2.0) == (1.0 + 0j)


def test_resolve_substrate_eps_descriptor_and_shorthand() -> None:
    ri_paths = {
        'glass': {'type': 'constant', 'epsilon': 2.25},
        'silicon': {'type': 'table', 'file': 'si.dat'},
        'water': 1.77,
        'air': '1.0',
        'vacuum': 'vacuum'}

    assert _resolve_substrate_eps('glass', ri_paths) == 2.25
    assert _resolve_substrate_eps('silicon', ri_paths) == 'si.dat'
    assert _resolve_substrate_eps('water', ri_paths) == 1.77
    assert _resolve_substrate_eps('air', ri_paths) == 1.0
    assert _resolve_substrate_eps('vacuum', ri_paths) == 'vacuum'
    assert _resolve_substrate_eps('unknown', ri_paths) == 'unknown'


def test_resolve_substrate_eps_rejects_callable() -> None:
    with pytest.raises(ValueError):
        _resolve_substrate_eps('glass', {'glass': lambda x: x})


def test_build_eps_particle_supports_resolved_runtime_values(monkeypatch,
    tmp_path) -> None:
    fake_pkg = types.ModuleType('mnpbem')
    fake_materials = types.ModuleType('mnpbem.materials')

    class EpsConst:
        def __init__(self, value):
            self.value = float(value)

    class EpsTable:
        def __init__(self, path):
            self.path = str(path)

    class EpsDrude:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, enei):
            return 1.0 + 0j

    fake_materials.EpsConst = EpsConst
    fake_materials.EpsTable = EpsTable
    fake_materials.EpsDrude = EpsDrude

    monkeypatch.setitem(sys.modules, 'mnpbem', fake_pkg)
    monkeypatch.setitem(sys.modules, 'mnpbem.materials', fake_materials)

    eps_const = _build_eps_particle('agcl', {'agcl': {'type': 'constant', 'epsilon': 2.02}})
    assert isinstance(eps_const, EpsConst)
    assert eps_const.value == 2.02

    eps_table = _build_eps_particle('ito', {'ito': 'ito.dat'})
    assert isinstance(eps_table, EpsTable)
    assert eps_table.path == 'ito.dat'

    eps_scalar = _build_eps_particle('al2o3', {'al2o3': 1.77})
    assert isinstance(eps_scalar, EpsConst)
    assert eps_scalar.value == 1.77

    fn = lambda enei: 1.0 + 0j
    eps_fn = _build_eps_particle('user_mat', {'user_mat': fn})
    assert eps_fn is fn

    module_path = tmp_path / 'user_material.py'
    module_path.write_text(
        'def generate_eps_func():\n'
        '    def _eps(enei):\n'
        '        return (enei * 0.0) + (2.5 + 0j)\n'
        '    return _eps\n',
        encoding = 'utf-8')
    resolved = resolve_refractive_index_paths({
        'from_file': {
            'type': 'python_module',
            'module_path': str(module_path),
            'factory': 'generate_eps_func'}})
    eps_from_file = _build_eps_particle('from_file', resolved)
    assert callable(eps_from_file)
    assert eps_from_file(2.0) == (2.5 + 0j)


def test_demo_sphere_structure_with_python_descriptor(monkeypatch) -> None:
    fake_pkg = types.ModuleType('mnpbem')
    fake_materials = types.ModuleType('mnpbem.materials')
    fake_geometry = types.ModuleType('mnpbem.geometry')

    class EpsConst:
        def __init__(self, value):
            self.value = float(value)

    class EpsTable:
        def __init__(self, path):
            self.path = str(path)

    class EpsDrude:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, enei):
            return 1.0 + 0j

    def trisphere(n, diameter):
        return {'n': int(n), 'diameter': float(diameter)}

    class ComParticle:
        def __init__(self, epstab, meshes, inout, interp = 'curv', refine = 2):
            self.eps = list(epstab)
            self.meshes = list(meshes)
            self.inout = list(inout)
            self.pfull = types.SimpleNamespace(nfaces = 123)

    fake_materials.EpsConst = EpsConst
    fake_materials.EpsTable = EpsTable
    fake_materials.EpsDrude = EpsDrude

    fake_geometry.trisphere = trisphere
    fake_geometry.ComParticle = ComParticle

    monkeypatch.setitem(sys.modules, 'mnpbem', fake_pkg)
    monkeypatch.setitem(sys.modules, 'mnpbem.materials', fake_materials)
    monkeypatch.setitem(sys.modules, 'mnpbem.geometry', fake_geometry)

    ex_path = Path(__file__).resolve().parents[1] / 'examples' / 'sphere_str_descriptor.py'
    str_args = runpy.run_path(str(ex_path))['args']

    sim_args = {
        'simulation_type': 'ret',
        'excitation_type': 'planewave',
        'wavelength_range': [500.0, 600.0, 3],
        'output_dir': './_tmp_test_out'}

    cfg = merge_str_sim_args(str_args, sim_args)
    p, epstab, nfaces = build_structure(cfg['structure'], cfg.get('materials', dict()))

    assert isinstance(epstab[0], EpsConst)
    assert isinstance(epstab[1], EpsDrude)
    assert callable(epstab[1])
    assert epstab[1](2.0) != 0.0
    assert nfaces == 123
    assert len(p.meshes) == 1


@pytest.mark.skipif(not _has_real_mnpbem(),
        reason = 'requires real mnpbem installation')
def test_demo_sphere_structure_with_python_descriptor_real_mnpbem() -> None:
    EpsDrude = getattr(importlib.import_module('mnpbem.materials'), 'EpsDrude')

    ex_path = Path(__file__).resolve().parents[1] / 'examples' / 'sphere_str_descriptor.py'
    str_args = runpy.run_path(str(ex_path))['args']

    sim_args = {
        'simulation_type': 'ret',
        'excitation_type': 'planewave',
        'wavelength_range': [500.0, 600.0, 3],
        'output_dir': './_tmp_test_out'}

    cfg = merge_str_sim_args(str_args, sim_args)
    p, epstab, nfaces = build_structure(cfg['structure'], cfg.get('materials', dict()))

    assert p is not None
    assert len(epstab) == 2
    assert isinstance(epstab[1], EpsDrude)
    assert callable(epstab[1])
    val = epstab[1](200.0)
    assert val is not None
    assert nfaces > 0
