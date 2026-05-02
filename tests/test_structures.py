import os
import sys
import glob

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _smoke_cfg(stype: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_m = {'medium': 'water', 'particle': 'gold', 'core': 'gold', 'shell': 'silver'}

    table: Dict[str, Dict[str, Any]] = {
        'sphere':                   {'type': 'sphere', 'diameter': 30, 'mesh_density': 60},
        'cube':                     {'type': 'cube', 'size': 20, 'n_per_edge': 6, 'e': 0.25},
        'rod':                      {'type': 'rod', 'diameter': 10, 'height': 30, 'mesh_density': 6.0},
        'ellipsoid':                {'type': 'ellipsoid', 'axes': [10, 12, 15], 'n_verts': 60},
        'triangle':                 {'type': 'triangle', 'side_length': 30, 'thickness': 5, 'nz': 5},
        'dimer_sphere':             {'type': 'dimer_sphere', 'diameter': 30, 'gap': 5, 'n_verts': 60},
        'dimer_cube':               {'type': 'dimer_cube', 'edge': 20, 'gap': 5, 'n_per_edge': 6, 'e': 0.25},
        'core_shell_sphere':        {'type': 'core_shell_sphere', 'core_diameter': 20, 'shell_thickness': 5, 'mesh_density': 60},
        'core_shell_cube':          {'type': 'core_shell_cube', 'core_size': 20, 'shell_thickness': 5, 'n_per_edge': 6, 'e': 0.25},
        'core_shell_rod':           {'type': 'core_shell_rod', 'core_diameter': 10, 'shell_thickness': 3, 'height': 40, 'mesh_density': 6.0},
        'dimer_core_shell_cube':    {'type': 'dimer_core_shell_cube', 'core_size': 20, 'shell_thickness': 5, 'gap': 3, 'n_per_edge': 6},
        'advanced_monomer_cube':    {'type': 'advanced_monomer_cube', 'core_size': 20, 'shell_layers': [5],
                'materials': ['gold', 'silver'], 'n_per_edge': 6},
        'advanced_dimer_cube':      {'type': 'advanced_dimer_cube', 'core_size': 20, 'shell_layers': [5],
                'materials': ['gold', 'silver'], 'gap': 3, 'tilt_angle': 10, 'rotation_angle': 5, 'n_per_edge': 6},
        'connected_dimer_cube':     {'type': 'connected_dimer_cube', 'core_size': 15, 'gap': 0.0, 'n_per_edge': 5, 'e': 0.25},
        'sphere_cluster_aggregate': {'type': 'sphere_cluster_aggregate', 'n_spheres': 3, 'diameter': 30, 'gap': -0.1, 'n_verts': 60}}

    if stype not in table:
        raise KeyError('[error] unknown smoke cfg <{}>'.format(stype))

    return table[stype], cfg_m


@pytest.mark.parametrize('stype', [
    'sphere',
    'cube',
    'rod',
    'ellipsoid',
    'triangle',
    'dimer_sphere',
    'dimer_cube',
    'core_shell_sphere',
    'core_shell_cube',
    'core_shell_rod',
    'dimer_core_shell_cube',
    'advanced_monomer_cube',
    'advanced_dimer_cube',
    'connected_dimer_cube',
    'sphere_cluster_aggregate'])
def test_structure_builds(stype: str) -> None:
    from pymnpbem_simulation.structures import build_structure, REGISTRY

    assert stype in REGISTRY, '[error] missing <{}> in REGISTRY'.format(stype)

    cfg_struct, cfg_mat = _smoke_cfg(stype)

    p, epstab, nfaces = build_structure(cfg_struct, cfg_mat)

    assert nfaces > 0, '[error] <{}> built with 0 faces'.format(stype)
    assert isinstance(epstab, list), '[error] <{}> epstab not list'.format(stype)
    assert len(epstab) >= 2, '[error] <{}> epstab too short'.format(stype)


def test_sphere_cluster_all_n() -> None:
    from pymnpbem_simulation.structures import SphereClusterBuilder

    cfg_m = {'medium': 'water', 'particle': 'gold'}
    for ns in range(1, 8):
        cfg = {'n_spheres': ns, 'diameter': 30, 'gap': -0.1, 'n_verts': 60}
        p, _, n = SphereClusterBuilder(cfg, cfg_m).build()
        assert n > 0, '[error] cluster N={} produced 0 faces'.format(ns)


def test_from_shape_inline_arrays() -> None:
    from mnpbem.geometry import trisphere
    from pymnpbem_simulation.structures import FromShapeBuilder

    cfg_m = {'medium': 'water', 'particle': 'gold'}
    spc = trisphere(60, 30.0)
    cfg_struct = {'vertices': spc.verts.tolist(), 'faces': spc.faces.tolist()}

    p, _, n = FromShapeBuilder(cfg_struct, cfg_m).build()
    assert n > 0


def test_from_shape_npz_roundtrip(tmp_path) -> None:
    from mnpbem.geometry import trisphere
    from pymnpbem_simulation.structures import FromShapeBuilder

    cfg_m = {'medium': 'water', 'particle': 'gold'}
    spc = trisphere(60, 30.0)

    npz_path = tmp_path / 'mesh.npz'
    np.savez(str(npz_path), verts = spc.verts, faces = spc.faces)

    cfg_struct = {'mesh_file': str(npz_path)}
    p, _, n = FromShapeBuilder(cfg_struct, cfg_m).build()
    assert n > 0


def test_registry_keys() -> None:
    from pymnpbem_simulation.structures import REGISTRY

    expected = {
        'sphere', 'cube', 'rod', 'ellipsoid', 'triangle',
        'dimer_sphere', 'dimer_cube',
        'core_shell_sphere', 'core_shell_cube', 'core_shell_rod',
        'dimer_core_shell_cube',
        'advanced_monomer_cube', 'advanced_dimer_cube',
        'connected_dimer_cube',
        'sphere_cluster_aggregate',
        'from_shape'}

    missing = expected - set(REGISTRY.keys())
    assert not missing, '[error] missing keys: {}'.format(missing)


def test_unknown_structure_raises() -> None:
    from pymnpbem_simulation.structures import build_structure

    with pytest.raises(ValueError):
        build_structure({'type': 'no_such_structure'}, {'medium': 'water', 'particle': 'gold'})


@pytest.mark.parametrize('yaml_file', sorted(glob.glob(str(REPO_ROOT / 'examples' / '*.yaml'))))
def test_example_yaml_loads(yaml_file: str) -> None:
    from pymnpbem_simulation.config import load_yaml
    from pymnpbem_simulation.structures import REGISTRY

    cfg = load_yaml(yaml_file)
    cfg_struct = cfg.get('structure', {})

    if not cfg_struct:
        return

    stype = cfg_struct.get('type', '')
    assert stype in REGISTRY, '[error] <{}> not in REGISTRY ({})'.format(stype, yaml_file)
