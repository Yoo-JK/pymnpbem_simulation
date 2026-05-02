import os
import sys
import copy

from typing import Any, Dict, List, Tuple, Optional

import yaml


REQUIRED_KEYS = [
    ('structure', 'type'),
    ('simulation', 'type'),
    ('simulation', 'excitation'),
    ('output', 'dir')]


DEFAULT_COMPUTE = {
    'n_workers': 1,
    'n_threads': 1,
    'n_gpus_per_worker': 0,
    'multi_node': False,
    'hmode': 'dense'}


DEFAULT_OUTPUT_FORMATS = ['npz', 'json', 'png']


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError('[error] YAML root must be a mapping in <{}>'.format(path))

    return cfg


def merge_overrides(cfg: Dict[str, Any],
        overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    for section, kvs in overrides.items():

        if kvs is None:
            continue

        if section not in out:
            out[section] = dict()

        for k, v in kvs.items():

            if v is None:
                continue

            out[section][k] = v

    return out


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    if 'compute' not in out:
        out['compute'] = dict()

    for k, v in DEFAULT_COMPUTE.items():

        if k not in out['compute']:
            out['compute'][k] = v

    if 'output' not in out:
        raise ValueError('[error] Missing <output> section!')

    if 'formats' not in out['output']:
        out['output']['formats'] = list(DEFAULT_OUTPUT_FORMATS)

    if 'name' not in out['output']:
        out['output']['name'] = 'simulation'

    if 'simulation' in out and 'wavelength_range' in out['simulation'] \
            and 'enei_min' not in out['simulation']:
        wr = out['simulation']['wavelength_range']
        out['simulation']['enei_min'] = wr[0]
        out['simulation']['enei_max'] = wr[1]
        out['simulation']['n_wavelengths'] = wr[2]

    return out


def validate_config(cfg: Dict[str, Any]) -> None:
    for section, key in REQUIRED_KEYS:

        if section not in cfg:
            raise ValueError('[error] Missing config section <{}>!'.format(section))

        if key not in cfg[section]:
            raise ValueError(
                '[error] Missing required key <{}.{}>!'.format(section, key))

    sim_type = cfg['simulation']['type']

    if sim_type not in {'ret', 'stat', 'ret_layer', 'stat_layer',
            'ret_iter', 'stat_iter', 'ret_layer_iter'}:
        raise ValueError('[error] Invalid simulation.type <{}>!'.format(sim_type))

    exc = cfg['simulation']['excitation']

    if exc not in {'planewave', 'dipole', 'eels'}:
        raise ValueError('[error] Invalid simulation.excitation <{}>!'.format(exc))


def save_yaml(path: str,
        cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys = False, default_flow_style = None)
