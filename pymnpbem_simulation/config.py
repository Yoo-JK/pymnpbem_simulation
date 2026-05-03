import os
import sys
import copy
import codecs

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
            'ret_iter', 'stat_iter', 'ret_layer_iter',
            'ret_mirror', 'stat_mirror'}:
        raise ValueError('[error] Invalid simulation.type <{}>!'.format(sim_type))

    exc = cfg['simulation']['excitation']

    if exc not in {'planewave', 'dipole', 'eels'}:
        raise ValueError('[error] Invalid simulation.excitation <{}>!'.format(exc))


def save_yaml(path: str,
        cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys = False, default_flow_style = None)


def load_py_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            '[error] Python config file not found: <{}>!'.format(path))

    with codecs.open(path, 'r', encoding = 'utf-8') as f:
        src = f.read()

    namespace = dict()
    exec(src, namespace)

    if 'args' not in namespace:
        raise ValueError(
            "[error] Config file <{}> must define 'args' dict!".format(path))

    if not isinstance(namespace['args'], dict):
        raise ValueError(
            "[error] 'args' in <{}> is not a dict!".format(path))

    return namespace['args']


_NESTED_PASSTHROUGH_KEYS = ('compute', 'output', 'iter', 'postprocess')


def merge_str_sim_args(str_args: Dict[str, Any],
        sim_args: Dict[str, Any]) -> Dict[str, Any]:
    from .migration.py_to_yaml import convert_args_to_yaml, merge_args

    str_flat = {k: v for k, v in str_args.items()
            if k not in _NESTED_PASSTHROUGH_KEYS}
    sim_flat = {k: v for k, v in sim_args.items()
            if k not in _NESTED_PASSTHROUGH_KEYS}

    merged_flat = merge_args(str_flat, sim_flat)
    cfg = convert_args_to_yaml(merged_flat)

    cfg = _ensure_compute_block(cfg, sim_args)
    cfg = _ensure_output_block(cfg, sim_args)
    cfg = _ensure_postprocess_block(cfg, sim_args)
    cfg = _drop_empty_extras(cfg)

    return cfg


def _ensure_compute_block(cfg: Dict[str, Any],
        sim_args: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    nested_compute = sim_args.get('compute', None)

    if isinstance(nested_compute, dict):

        if 'compute' not in out:
            out['compute'] = dict()

        for k, v in nested_compute.items():
            out['compute'][k] = v

    nested_iter = sim_args.get('iter', None)

    if isinstance(nested_iter, dict):

        if 'compute' not in out:
            out['compute'] = dict()

        out['compute']['iterative'] = True

        if 'iter_options' not in out['compute']:
            out['compute']['iter_options'] = dict()

        for k, v in nested_iter.items():
            out['compute']['iter_options'][k] = v

    return out


def _ensure_output_block(cfg: Dict[str, Any],
        sim_args: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    nested_output = sim_args.get('output', None)

    if isinstance(nested_output, dict):

        if 'output' not in out:
            out['output'] = dict()

        for k, v in nested_output.items():
            out['output'][k] = v

    return out


def _ensure_postprocess_block(cfg: Dict[str, Any],
        sim_args: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    nested_pp = sim_args.get('postprocess', None)

    if isinstance(nested_pp, dict):

        if 'postprocess' not in out:
            out['postprocess'] = dict()

        for k, v in nested_pp.items():
            out['postprocess'][k] = v

    return out


def _drop_empty_extras(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    extras = out.get('extras', None)

    if isinstance(extras, dict) and len(extras) == 0:
        del out['extras']

    return out
