"""Migration: field-only auto-redirect (Issue 4).

Source ``.py`` configs that set ``calculate_cross_sections=False`` and
``calculate_fields=True`` (or supply ``field_region``) should produce
YAML that drives FieldCalculator instead of the spectrum runner.
"""

import os
import sys
import tempfile

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation.migration.py_to_yaml import (
        _post_process, _redirect_field_only_simulation,
        _grid_from_field_region, upgrade_yaml)


def _spectrum_cfg(
        calculate_cross_sections = True,
        calculate_fields = False,
        with_region = False) -> dict:
    sim = {
        'type': 'ret',
        'excitation': 'planewave',
        'calculate_cross_sections': calculate_cross_sections,
        'calculate_fields': calculate_fields,
        'enei_min': 500,
        'enei_max': 1000,
        'n_wavelengths': 100}
    if with_region:
        sim['field_region'] = {
            'x_range': [-80, 80, 161],
            'y_range': [0, 0, 1],
            'z_range': [-80, 80, 161]}
        sim['field_mindist'] = 0.5
        sim['field_nmax'] = 2000
    return {'simulation': sim}


def test_redirect_skipped_when_cross_sections_true():
    cfg = _spectrum_cfg(True, False)
    out = _redirect_field_only_simulation(cfg)
    assert out['simulation']['type'] == 'ret'


def test_redirect_skipped_when_no_field_signals():
    """Cross sections off but no fields requested -> leave type alone."""
    cfg = _spectrum_cfg(False, False)
    out = _redirect_field_only_simulation(cfg)
    assert out['simulation']['type'] == 'ret'


def test_redirect_triggered_by_calculate_fields():
    cfg = _spectrum_cfg(False, True)
    out = _redirect_field_only_simulation(cfg)
    assert out['simulation']['type'] == 'field'
    assert out['simulation']['original_type'] == 'ret'


def test_redirect_triggered_by_field_region():
    """field_region present but calculate_fields not set -> still redirect."""
    cfg = _spectrum_cfg(False, False, with_region = True)
    out = _redirect_field_only_simulation(cfg)
    assert out['simulation']['type'] == 'field'
    assert 'grid' in out['simulation']


def test_grid_from_field_region_basic():
    region = {
        'x_range': [-80, 80, 161],
        'y_range': [0, 0, 1],
        'z_range': [-80, 80, 161]}
    grid = _grid_from_field_region(region)
    assert grid['type'] == 'rectangular'
    assert grid['x_range'] == [-80.0, 80.0]
    assert grid['n_points'] == [161, 1, 161]


def test_redirect_preserves_field_mindist_nmax():
    cfg = _spectrum_cfg(False, True, with_region = True)
    out = _redirect_field_only_simulation(cfg)
    sim = out['simulation']
    assert sim.get('mindist') == 0.5
    assert sim.get('nmax') == 2000


def test_post_process_runs_redirect():
    """Top-level post_process must invoke the redirect path."""
    cfg = {
        'simulation': {
            'type': 'ret',
            'wavelength_range': [500, 1000, 100],
            'calculate_cross_sections': False,
            'calculate_fields': True,
            'field_region': {
                'x_range': [-50, 50, 51],
                'y_range': [0, 0, 1],
                'z_range': [-50, 50, 51]}}}
    out = _post_process(cfg)
    assert out['simulation']['type'] == 'field'
    # wavelength_range expansion still ran
    assert out['simulation']['enei_min'] == 500


def test_upgrade_yaml_inplace(tmp_path):
    """End-to-end: take a v1.4 spectrum-converted yaml and upgrade it."""
    cfg = {
        'output': {'name': 'au_r0.2_g0.0', 'dir': './results'},
        'structure': {'type': 'connected_dimer_cube',
                'core_size': 47, 'gap': -0.1, 'rounding': 0.2,
                'mesh_density': 2, 'n_per_edge': 24},
        'materials': {'medium': 'water', 'particle': 'gold'},
        'compute': {'use_parallel': True, 'n_workers': 4},
        'simulation': {
            'type': 'ret',
            'excitation': 'planewave',
            'wavelength_range': [500, 1000, 100],
            'calculate_cross_sections': False,
            'calculate_fields': True,
            'field_region': {
                'x_range': [-80, 80, 161],
                'y_range': [0, 0, 1],
                'z_range': [-80, 80, 161]},
            'field_mindist': 0.5,
            'field_nmax': 2000,
            'enei_min': 500,
            'enei_max': 1000,
            'n_wavelengths': 100}}

    in_path = tmp_path / 'input.yaml'
    with open(str(in_path), 'w') as f:
        yaml.safe_dump(cfg, f)

    upgrade_yaml(str(in_path))

    with open(str(in_path), 'r') as f:
        new_cfg = yaml.safe_load(f)

    sim = new_cfg['simulation']
    assert sim['type'] == 'field'
    assert sim['original_type'] == 'ret'
    assert sim['grid']['type'] == 'rectangular'
    assert sim['grid']['x_range'] == [-80.0, 80.0]
    assert sim['grid']['n_points'] == [161, 1, 161]
    assert sim['mindist'] == 0.5


def test_upgrade_yaml_keeps_spectrum_when_cross_sections_true(tmp_path):
    cfg = {
        'simulation': {
            'type': 'ret',
            'calculate_cross_sections': True,
            'calculate_fields': False}}

    p = tmp_path / 'input.yaml'
    with open(str(p), 'w') as f:
        yaml.safe_dump(cfg, f)

    upgrade_yaml(str(p))

    with open(str(p), 'r') as f:
        new_cfg = yaml.safe_load(f)

    assert new_cfg['simulation']['type'] == 'ret'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
