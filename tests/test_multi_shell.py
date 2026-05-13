"""Multi-shell core_shell builders (Issue 2 v1.5+).

Verifies the new ``shells: [...]`` list config supports N-layer core_shell
construction (1, 2, 3 shells) and that the legacy single-shell schema
(``shell_thickness`` + ``materials.shell``) still works unchanged.
"""

import os
import sys

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation.structures.core_shell_sphere import (
        _build_inout_table, _normalize_shells)


def test_inout_table_single_shell():
    inout = _build_inout_table(1)
    assert inout == [[2, 3], [3, 1]]


def test_inout_table_two_shells():
    inout = _build_inout_table(2)
    # epstab = [medium=1, core=2, shell1=3, shell2=4]
    # particles = [p_core, p_shell1, p_shell2]
    assert inout == [[2, 3], [3, 4], [4, 1]]


def test_inout_table_three_shells():
    inout = _build_inout_table(3)
    # epstab = [medium=1, core=2, shell1=3, shell2=4, shell3=5]
    assert inout == [[2, 3], [3, 4], [4, 5], [5, 1]]


def test_inout_table_five_shells():
    inout = _build_inout_table(5)
    # epstab = [medium=1, core=2, shell1=3, shell2=4, shell3=5, shell4=6, shell5=7]
    # particles = [p_core, p_shell1, ..., p_shell5]
    expected = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1]]
    assert inout == expected


def test_normalize_shells_legacy_single():
    cfg = {'shell_thickness': 5.0, 'n_shell': 100}
    cfg_m = {'shell': 'silver'}
    out = _normalize_shells(cfg, cfg_m, default_n = 60)
    assert len(out) == 1
    assert out[0]['thickness'] == 5.0
    assert out[0]['material'] == 'silver'
    assert out[0]['n'] == 100


def test_normalize_shells_legacy_no_n_shell():
    cfg = {'shell_thickness': 3.0}
    cfg_m = {'shell': 'silver'}
    out = _normalize_shells(cfg, cfg_m, default_n = 60)
    assert out[0]['n'] == 60


def test_normalize_shells_new_list_three():
    cfg = {'shells': [
        {'thickness': 3.0, 'material': 'silver'},
        {'thickness': 2.0, 'material': 'silica'},
        {'thickness': 1.0, 'material': 'gold'}]}
    cfg_m = {}
    out = _normalize_shells(cfg, cfg_m, default_n = 60)
    assert len(out) == 3
    assert [sh['material'] for sh in out] == ['silver', 'silica', 'gold']
    assert [sh['thickness'] for sh in out] == [3.0, 2.0, 1.0]


def test_normalize_shells_empty_returns_empty():
    cfg = {}
    cfg_m = {}
    out = _normalize_shells(cfg, cfg_m, default_n = 60)
    assert out == []


# --- end-to-end builder tests ----------------------------------------------


def test_core_shell_sphere_legacy_single_shell():
    """Legacy single-shell config (shell_thickness) still works."""
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'core_shell_sphere',
            'core_diameter': 20.0,
            'shell_thickness': 5.0,
            'mesh_density': 60}
    cfg_m = {'medium': 'water', 'core': 'gold', 'shell': 'silver'}
    p, epstab, nfaces = build_structure(cfg, cfg_m)
    assert nfaces > 0
    assert len(epstab) == 3  # medium + core + 1 shell


def test_core_shell_sphere_two_shells():
    """New `shells: [...]` config builds N=2 layers."""
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'core_shell_sphere',
            'core_diameter': 20.0,
            'mesh_density': 60,
            'shells': [
                {'thickness': 3.0, 'material': 'silver'},
                {'thickness': 2.0, 'material': 'gold'}]}
    cfg_m = {'medium': 'water', 'core': 'gold'}
    p, epstab, nfaces = build_structure(cfg, cfg_m)
    assert nfaces > 0
    assert len(epstab) == 4  # medium + core + 2 shells


def test_core_shell_sphere_three_shells():
    """N=3 layers."""
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'core_shell_sphere',
            'core_diameter': 15.0,
            'mesh_density': 60,
            'shells': [
                {'thickness': 2.0, 'material': 'silver'},
                {'thickness': 2.0, 'material': 'gold'},
                {'thickness': 1.0, 'material': 'silver'}]}
    cfg_m = {'medium': 'water', 'core': 'gold'}
    p, epstab, nfaces = build_structure(cfg, cfg_m)
    assert nfaces > 0
    assert len(epstab) == 5  # medium + core + 3 shells


def test_core_shell_cube_two_shells():
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'core_shell_cube',
            'core_size': 20.0,
            'n_per_edge': 6,
            'e': 0.25,
            'shells': [
                {'thickness': 3.0, 'material': 'silver'},
                {'thickness': 2.0, 'material': 'gold'}]}
    cfg_m = {'medium': 'water', 'core': 'gold'}
    p, epstab, nfaces = build_structure(cfg, cfg_m)
    assert nfaces > 0
    assert len(epstab) == 4


def test_core_shell_rod_two_shells():
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'core_shell_rod',
            'core_diameter': 10.0,
            'height': 40.0,
            'mesh_density': 6.0,
            'shells': [
                {'thickness': 2.0, 'material': 'silver'},
                {'thickness': 1.0, 'material': 'gold'}]}
    cfg_m = {'medium': 'water', 'core': 'gold'}
    p, epstab, nfaces = build_structure(cfg, cfg_m)
    assert nfaces > 0
    assert len(epstab) == 4


def test_core_shell_no_shells_raises():
    """Builder must error if neither legacy nor new config gives a shell."""
    from pymnpbem_simulation.structures import build_structure

    cfg = {'type': 'core_shell_sphere',
            'core_diameter': 20.0,
            'mesh_density': 60}
    cfg_m = {'medium': 'water', 'core': 'gold'}
    with pytest.raises(ValueError, match = 'no shells'):
        build_structure(cfg, cfg_m)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
