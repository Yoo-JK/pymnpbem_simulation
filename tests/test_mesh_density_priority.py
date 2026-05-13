"""mesh_density 우선순위 (v1.6.0 D).

pymnpbem builder 의 mesh_density 가 n_per_edge 보다 우선 적용되는지 검증.
의미체계는 mnpbem_simulation MATLAB wrapper 와 동일:
    n_per_edge = round(core_size / mesh_density)

기준은 core size (shell 포함 X). multilayer 의 경우 모든 layer 가 동일한
n_per_edge 를 공유한다.

cube 류 builder 만 영향. sphere 류 (n_verts), rod 류 (D, H, n) 는 별개.
"""

import os
import sys
import warnings

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation.structures.advanced_monomer_cube import (
        _layer_sizes,
        _n_per_edge_from_density,
        _resolve_n_per_edge)


def test_n_per_edge_from_density_basic():
    assert _n_per_edge_from_density(47.0, 2.0) == 24
    assert _n_per_edge_from_density(100.0, 4.0) == 25
    assert _n_per_edge_from_density(30.0, 3.0) == 10


def test_n_per_edge_from_density_round_to_nearest():
    # 47 / 2.5 = 18.8 -> 19
    assert _n_per_edge_from_density(47.0, 2.5) == 19
    # 50 / 3 = 16.67 -> 17
    assert _n_per_edge_from_density(50.0, 3.0) == 17


def test_n_per_edge_from_density_min_clamp():
    # 너무 큰 mesh_density 는 최소 2 로 clamp
    assert _n_per_edge_from_density(10.0, 100.0) == 2


def test_resolve_n_per_edge_only_n_per_edge():
    cfg = {'n_per_edge': 8, 'core_size': 30.0}
    out = _resolve_n_per_edge(cfg, 1)
    assert out == [8]


def test_resolve_n_per_edge_only_mesh_density():
    cfg = {'mesh_density': 2.0, 'core_size': 47.0}
    out = _resolve_n_per_edge(cfg, 1)
    assert out == [24]


def test_resolve_n_per_edge_only_mesh_density_explicit_edge():
    cfg = {'mesh_density': 4.0}
    out = _resolve_n_per_edge(cfg, 1, edge_override = 100.0)
    assert out == [25]


def test_resolve_n_per_edge_default_when_neither():
    cfg = {'core_size': 30.0}
    out = _resolve_n_per_edge(cfg, 1)
    assert out == [16]


def test_resolve_n_per_edge_both_consistent_no_warning():
    cfg = {'mesh_density': 2.0, 'n_per_edge': 24, 'core_size': 47.0}
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter('always')
        out = _resolve_n_per_edge(cfg, 1)
    assert out == [24]
    assert len(w) == 0, '[error] 일치하는 경우 warning 이 발생함'


def test_resolve_n_per_edge_both_inconsistent_warning():
    cfg = {'mesh_density': 2.0, 'n_per_edge': 16, 'core_size': 47.0}
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter('always')
        out = _resolve_n_per_edge(cfg, 1)
    assert out == [24], '[error] mesh_density 가 우선되어야 함'
    assert len(w) == 1, '[error] inconsistent 경우 warning 이 발생해야 함'
    assert 'mesh_density' in str(w[0].message)
    assert 'priority' in str(w[0].message)


def test_resolve_n_per_edge_n_per_edges_overrides():
    cfg = {'n_per_edges': [8, 12], 'mesh_density': 2.0, 'core_size': 30.0}
    out = _resolve_n_per_edge(cfg, 2)
    # n_per_edges 명시된 경우 그대로 사용 (가장 우선)
    assert out == [8, 12]


def test_resolve_n_per_edge_multilayer_core_size():
    # core 30 nm 기준. 30 / 2 = 15. 모든 layer 동일 n_per_edge.
    cfg = {'core_size': 30.0,
            'shell_layers': [5.0, 5.0],
            'mesh_density': 2.0}
    out = _resolve_n_per_edge(cfg, 3)
    assert out == [15, 15, 15]


def test_resolve_n_per_edge_multilayer_47nm_core_4nm_shell():
    # 사용자 case: 47 nm core + 4 nm shell -> mesh_density=2 -> n=24 (core 기준)
    cfg = {'core_size': 47.0,
            'shell_layers': [4.0],
            'mesh_density': 2.0}
    out = _resolve_n_per_edge(cfg, 2)
    assert out == [24, 24]


def test_layer_sizes_simple():
    cfg = {'core_size': 30.0, 'shell_layers': [3.0, 2.0]}
    sizes = _layer_sizes(cfg, 3)
    assert sizes == [30.0, 36.0, 40.0]


def test_layer_sizes_dict_form():
    cfg = {'core_size': 30.0,
            'shell_layers': [{'thickness': 3.0}, {'thickness': 2.0}]}
    sizes = _layer_sizes(cfg, 3)
    assert sizes == [30.0, 36.0, 40.0]


# 사용자 case 회귀 (47 nm + mesh_density=2 -> n=24, 100 nm + mesh_density=4 -> n=25)


def test_user_case_47nm_md2():
    cfg = {'core_size': 47.0, 'mesh_density': 2.0}
    assert _resolve_n_per_edge(cfg, 1) == [24]


def test_user_case_100nm_md4():
    cfg = {'edge': 100.0, 'mesh_density': 4.0}
    assert _resolve_n_per_edge(cfg, 1) == [25]


# 빌더 통합 테스트 (mnpbem 의존)


_MNPBEM_AVAILABLE = True
try:
    import mnpbem  # noqa: F401
except Exception:
    _MNPBEM_AVAILABLE = False


@pytest.mark.skipif(not _MNPBEM_AVAILABLE, reason = 'mnpbem not installed')
def test_cube_builder_mesh_density_matches_n_per_edge():
    from pymnpbem_simulation.structures.cube import CubeBuilder

    cfg_m = {'medium': 'water', 'particle': 'gold'}

    cfg_md = {'size': 30.0, 'mesh_density': 5.0, 'e': 0.25, 'refine': 1}
    p1, _, n1 = CubeBuilder(cfg_md, cfg_m).build()

    cfg_n = {'size': 30.0, 'n_per_edge': 6, 'e': 0.25, 'refine': 1}
    p2, _, n2 = CubeBuilder(cfg_n, cfg_m).build()

    assert n1 == n2, '[error] mesh_density=5 (-> 6) 가 n_per_edge=6 과 동일해야 함, got {} vs {}'.format(n1, n2)


@pytest.mark.skipif(not _MNPBEM_AVAILABLE, reason = 'mnpbem not installed')
def test_dimer_cube_builder_user_case_47nm_md2():
    from pymnpbem_simulation.structures.dimer_cube import DimerCubeBuilder

    cfg_m = {'medium': 'water', 'particle': 'gold'}

    cfg_md = {'edge': 47.0, 'gap': 0.6, 'mesh_density': 2.0,
            'e': 0.2, 'refine': 1}
    p1, _, n1 = DimerCubeBuilder(cfg_md, cfg_m).build()

    cfg_n = {'edge': 47.0, 'gap': 0.6, 'n_per_edge': 24,
            'e': 0.2, 'refine': 1}
    p2, _, n2 = DimerCubeBuilder(cfg_n, cfg_m).build()

    assert n1 == n2, '[error] 47 nm + mesh_density=2 와 n_per_edge=24 가 동일해야 함, got {} vs {}'.format(n1, n2)


@pytest.mark.skipif(not _MNPBEM_AVAILABLE, reason = 'mnpbem not installed')
def test_dimer_cube_builder_mesh_density_overrides_n_per_edge():
    from pymnpbem_simulation.structures.dimer_cube import DimerCubeBuilder

    cfg_m = {'medium': 'water', 'particle': 'gold'}

    # mesh_density=2 -> n_per_edge=24, but explicit n_per_edge=12
    cfg_conflict = {'edge': 47.0, 'gap': 0.6, 'mesh_density': 2.0,
            'n_per_edge': 12, 'e': 0.2, 'refine': 1}
    cfg_priority = {'edge': 47.0, 'gap': 0.6, 'n_per_edge': 24,
            'e': 0.2, 'refine': 1}

    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter('always')
        _, _, n_conflict = DimerCubeBuilder(cfg_conflict, cfg_m).build()
    _, _, n_priority = DimerCubeBuilder(cfg_priority, cfg_m).build()

    assert n_conflict == n_priority, \
        '[error] mesh_density 가 우선되어야 함, got {} vs {}'.format(n_conflict, n_priority)
    assert any('mesh_density' in str(wi.message) for wi in w), \
        '[error] mesh_density override warning 이 발생해야 함'
