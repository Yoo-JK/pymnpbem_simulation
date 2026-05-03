"""--str-conf / --sim-conf CLI tests.

Validates the new mnpbem_simulation-compatible CLI pattern: two .py
config files (structure + simulation) merged into a single internal
cfg dict equivalent to the legacy --config <yaml> mode.
"""

import os
import sys
import argparse

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation.config import (
        load_py_config, merge_str_sim_args, apply_defaults, validate_config)
from pymnpbem_simulation.cli import (
        _build_parser, _has_required_inputs, _build_overrides)
from pymnpbem_simulation.migration.yaml_to_str_sim import (
        yaml_to_flat_args, split_flat_to_str_sim, convert_yaml_to_str_sim)


def _write_py(path, args_dict):
    with open(str(path), 'w') as f:
        f.write('args = {}\n'.format(repr(args_dict)))


def test_load_py_config_roundtrip(tmp_path):
    p = tmp_path / 'cfg.py'
    _write_py(p, {'foo': 1, 'bar': 'baz', 'list': [1, 2, 3]})
    loaded = load_py_config(str(p))
    assert loaded == {'foo': 1, 'bar': 'baz', 'list': [1, 2, 3]}


def test_load_py_config_missing_args(tmp_path):
    p = tmp_path / 'cfg.py'
    with open(str(p), 'w') as f:
        f.write('x = 5\n')
    with pytest.raises(ValueError, match = 'args'):
        load_py_config(str(p))


def test_load_py_config_not_dict(tmp_path):
    p = tmp_path / 'cfg.py'
    with open(str(p), 'w') as f:
        f.write('args = [1, 2, 3]\n')
    with pytest.raises(ValueError, match = 'is not a dict'):
        load_py_config(str(p))


def test_load_py_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_py_config('/nonexistent/path/cfg.py')


def test_merge_str_sim_basic(tmp_path):
    str_args = {
        'structure': 'sphere',
        'diameter': 50,
        'mesh_density': 5,
        'materials': ['gold'],
        'medium': 'water'}
    sim_args = {
        'simulation_type': 'stat',
        'excitation_type': 'planewave',
        'wavelength_range': [400, 800, 50],
        'polarizations': [[1, 0, 0]],
        'propagation_dirs': [[0, 0, 1]]}

    cfg = merge_str_sim_args(str_args, sim_args)

    assert cfg['structure']['type'] == 'sphere'
    assert cfg['structure']['diameter'] == 50
    assert cfg['simulation']['type'] == 'stat'
    assert cfg['simulation']['excitation'] == 'planewave'
    assert cfg['materials']['particle_list'] == ['gold']
    assert cfg['materials']['medium'] == 'water'


def test_merge_str_sim_compute_block(tmp_path):
    str_args = {'structure': 'sphere', 'diameter': 50, 'materials': ['gold']}
    sim_args = {
        'simulation_type': 'ret',
        'excitation_type': 'planewave',
        'wavelength_range': [400, 800, 20],
        'compute': {
            'n_workers': 4,
            'n_threads': 2,
            'n_gpus_per_worker': 1,
            'multi_node': False,
            'hmode': 'dense'}}

    cfg = merge_str_sim_args(str_args, sim_args)

    assert cfg['compute']['n_workers'] == 4
    assert cfg['compute']['n_threads'] == 2
    assert cfg['compute']['n_gpus_per_worker'] == 1


def test_merge_str_sim_output_block(tmp_path):
    str_args = {'structure': 'sphere', 'diameter': 50, 'materials': ['gold']}
    sim_args = {
        'simulation_type': 'stat',
        'excitation_type': 'planewave',
        'wavelength_range': [400, 800, 20],
        'output': {
            'dir': '/tmp/run',
            'name': 'sphere_test',
            'save_plots': True}}

    cfg = merge_str_sim_args(str_args, sim_args)

    assert cfg['output']['dir'] == '/tmp/run'
    assert cfg['output']['name'] == 'sphere_test'
    assert cfg['output']['save_plots'] is True


def test_merge_str_sim_iter_block(tmp_path):
    str_args = {'structure': 'sphere', 'diameter': 50, 'materials': ['gold']}
    sim_args = {
        'simulation_type': 'ret',
        'excitation_type': 'planewave',
        'wavelength_range': [400, 800, 20],
        'iter': {
            'tol': 1e-6,
            'maxiter': 200},
        'compute': {'n_workers': 1}}

    cfg = merge_str_sim_args(str_args, sim_args)

    assert cfg['compute']['iterative'] is True
    assert cfg['compute']['iter_options']['tol'] == 1e-6
    assert cfg['compute']['iter_options']['maxiter'] == 200


def test_merge_str_sim_validates(tmp_path):
    str_args = {'structure': 'sphere', 'diameter': 50, 'materials': ['gold']}
    sim_args = {
        'simulation_type': 'stat',
        'excitation_type': 'planewave',
        'wavelength_range': [400, 800, 20],
        'output': {'dir': '/tmp/run', 'name': 'sphere'}}

    cfg = merge_str_sim_args(str_args, sim_args)
    cfg = apply_defaults(cfg)
    validate_config(cfg)


def test_parser_str_sim_args():
    parser = _build_parser()
    args = parser.parse_args([
        '--str-conf', 'a.py', '--sim-conf', 'b.py', '--verbose'])
    assert args.str_conf == 'a.py'
    assert args.sim_conf == 'b.py'
    assert args.verbose is True
    assert args.config is None


def test_parser_legacy_yaml():
    parser = _build_parser()
    args = parser.parse_args(['--config', 'foo.yaml'])
    assert args.config == 'foo.yaml'
    assert args.str_conf is None
    assert args.sim_conf is None


def test_has_required_inputs_str_sim():
    args = argparse.Namespace(str_conf = 'a.py', sim_conf = 'b.py', config = None)
    assert _has_required_inputs(args) is True


def test_has_required_inputs_legacy():
    args = argparse.Namespace(str_conf = None, sim_conf = None, config = 'x.yaml')
    assert _has_required_inputs(args) is True


def test_has_required_inputs_neither():
    args = argparse.Namespace(str_conf = None, sim_conf = None, config = None)
    assert _has_required_inputs(args) is False


def test_cli_overrides_priority():
    parser = _build_parser()
    args = parser.parse_args([
        '--str-conf', 'a.py', '--sim-conf', 'b.py',
        '--n-workers', '8',
        '--n-threads', '2',
        '--n-gpus-per-worker', '4',
        '--simulation-name', 'override_name',
        '--n-wavelengths', '10'])

    overrides = _build_overrides(args)
    assert overrides['compute']['n_workers'] == 8
    assert overrides['compute']['n_threads'] == 2
    assert overrides['compute']['n_gpus_per_worker'] == 4
    assert overrides['output']['name'] == 'override_name'
    assert args.n_wavelengths == 10


def test_cli_vram_share_backend_override():
    parser = _build_parser()
    args = parser.parse_args([
        '--str-conf', 'a.py', '--sim-conf', 'b.py',
        '--vram-share-backend', 'cusolvermg'])
    overrides = _build_overrides(args)
    assert overrides['compute']['vram_share_backend'] == 'cusolvermg'


def test_yaml_to_str_sim_split(tmp_path):
    cfg = {
        'output': {
            'name': 'auag_test',
            'dir': './results',
            'save_plots': True},
        'structure': {
            'type': 'advanced_dimer_cube',
            'core_size': 47,
            'shell_layers': [4],
            'roundings': [0.2, 0.2],
            'mesh_density': 2,
            'gap': 0.6,
            'refine': 3},
        'materials': {
            'particle_list': ['gold', 'silver'],
            'medium': 'water'},
        'compute': {
            'n_workers': 5,
            'n_threads': 1,
            'n_gpus_per_worker': 0},
        'simulation': {
            'type': 'ret',
            'excitation': 'planewave',
            'wavelength_range': [300, 1000, 140],
            'polarizations': [[1, 0, 0], [0, 1, 0]],
            'propagation_dirs': [[0, 0, 1], [0, 0, 1]],
            'interp': 'curv'}}

    yaml_path = tmp_path / 'in.yaml'
    with open(str(yaml_path), 'w') as f:
        yaml.safe_dump(cfg, f)

    str_path = tmp_path / 'out_str.py'
    sim_path = tmp_path / 'out_sim.py'

    str_args, sim_args = convert_yaml_to_str_sim(
            str(yaml_path), str(str_path), str(sim_path))

    assert str_args.get('structure') == 'advanced_dimer_cube'
    assert str_args.get('core_size') == 47
    assert str_args.get('materials') == ['gold', 'silver']
    assert str_args.get('medium') == 'water'

    assert sim_args.get('simulation_type') == 'ret'
    assert sim_args.get('excitation_type') == 'planewave'
    assert sim_args.get('wavelength_range') == [300, 1000, 140]
    assert sim_args.get('output_dir') == './results'

    # Round-trip: re-load .py and merge back
    loaded_str = load_py_config(str(str_path))
    loaded_sim = load_py_config(str(sim_path))
    cfg_merged = merge_str_sim_args(loaded_str, loaded_sim)

    assert cfg_merged['structure']['type'] == 'advanced_dimer_cube'
    assert cfg_merged['simulation']['type'] == 'ret'


def test_examples_auag_dimer_loadable():
    str_path = REPO_ROOT / 'examples' / 'auag_dimer_str.py'
    sim_path = REPO_ROOT / 'examples' / 'auag_dimer_sim.py'
    assert str_path.exists()
    assert sim_path.exists()

    str_args = load_py_config(str(str_path))
    sim_args = load_py_config(str(sim_path))

    cfg = merge_str_sim_args(str_args, sim_args)
    cfg = apply_defaults(cfg)
    validate_config(cfg)

    assert cfg['structure']['type'] == 'advanced_dimer_cube'
    assert cfg['simulation']['type'] in {'ret', 'ret_iter'}
    assert cfg['compute']['n_workers'] == 5
    assert cfg['output']['name'] == 'auag_r0.2_g0.6'


def test_examples_sphere_loadable():
    str_path = REPO_ROOT / 'examples' / 'sphere_str.py'
    sim_path = REPO_ROOT / 'examples' / 'sphere_sim.py'

    str_args = load_py_config(str(str_path))
    sim_args = load_py_config(str(sim_path))

    cfg = merge_str_sim_args(str_args, sim_args)
    cfg = apply_defaults(cfg)
    validate_config(cfg)

    assert cfg['structure']['type'] == 'sphere'
    assert cfg['simulation']['type'] == 'stat'


def test_yaml_legacy_still_loadable():
    """Backward-compat: existing YAML examples must still parse."""
    from pymnpbem_simulation.config import load_yaml

    yaml_path = REPO_ROOT / 'examples' / 'dimer_baseline.yaml'
    cfg = load_yaml(str(yaml_path))
    cfg = apply_defaults(cfg)
    validate_config(cfg)
    assert cfg['structure']['type'] == 'dimer_cube'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
