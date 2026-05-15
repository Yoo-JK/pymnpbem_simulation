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

    # Forward simulation.interp -> structure.interp when the user wrote interp
    # only under the simulation block (MATLAB-style op.interp convention).
    # Builders read structure.interp only, so without this forward the
    # user-specified interp is silently ignored.
    if ('structure' in out and 'simulation' in out
            and 'interp' in out['simulation']
            and 'interp' not in out['structure']):
        out['structure']['interp'] = out['simulation']['interp']

    out = _auto_promote_iter(out)
    out = _auto_wrap_substrate(out)
    out = _auto_convert_field_region(out)

    return out


_SIM_ITER_PROMOTE = {
    'ret': 'ret_iter',
    'stat': 'stat_iter'}


_SIM_LAYER_PROMOTE = {
    'ret': 'ret_layer',
    'stat': 'stat_layer',
    'ret_iter': 'ret_layer_iter'}


def _auto_convert_field_region(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy <simulation.field_region> (with [min, max, npts]
    triples) into the modern <simulation.grid> rectangular block expected
    by FieldCalculator.

    ONLY triggers when this is actually a field-evaluation pass — i.e.:
      * <calculate_fields> is True, OR
      * <calculate_spectrum> is False (field-only mode), OR
      * <type> is already 'field'

    If a yaml carries field_region metadata for a future field run but
    the current pass is spectrum-only (calculate_fields=false, default),
    the conversion is skipped so dispatch correctly routes to the
    spectrum runner instead of FieldCalculator.

    Also maps <field_mindist> -> <mindist> and <field_nmax> -> <nmax>.
    No-op when <grid> already exists or <field_region> is missing.
    """
    sim = cfg.get('simulation', None)

    if not isinstance(sim, dict):
        return cfg

    if 'grid' in sim:
        return cfg

    region = sim.get('field_region', None)
    if not isinstance(region, dict):
        return cfg

    is_field_pass = (
            sim.get('calculate_fields') is True
            or sim.get('calculate_spectrum') is False
            or sim.get('type') in ('field', 'field_ret', 'field_stat'))

    if not is_field_pass:
        return cfg

    grid: Dict[str, Any] = {'type': 'rectangular'}
    n_points: List[int] = []

    for axis in ('x', 'y', 'z'):
        rng = region.get('{}_range'.format(axis), None)
        if isinstance(rng, (list, tuple)) and len(rng) >= 2:
            grid['{}_range'.format(axis)] = [float(rng[0]), float(rng[1])]
            n_points.append(int(rng[2]) if len(rng) >= 3 else 1)

    if len(n_points) == 3:
        grid['n_points'] = n_points

    sim['grid'] = grid

    if 'field_mindist' in sim and 'mindist' not in sim:
        sim['mindist'] = sim['field_mindist']
    if 'field_nmax' in sim and 'nmax' not in sim:
        sim['nmax'] = sim['field_nmax']

    return cfg


def _auto_promote_iter(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Promote <simulation.type> from <ret>/<stat> to the <_iter> variant when
    <compute.iterative>=True. Runs before _auto_wrap_substrate so the chain
    <ret> -> <ret_iter> -> <ret_layer_iter> resolves in one pass when both
    <iterative> and <use_substrate> are True.
    """

    cmp_ = cfg.get('compute', None)

    if not isinstance(cmp_, dict):
        return cfg

    if cmp_.get('iterative') is not True:
        return cfg

    sim = cfg.get('simulation', None)

    if not isinstance(sim, dict):
        return cfg

    sim_type = str(sim.get('type', '')).lower()

    if sim_type in _SIM_ITER_PROMOTE:
        promoted = _SIM_ITER_PROMOTE[sim_type]
        sim['type'] = promoted
        print('[info] auto-promoted simulation.type <{}> -> <{}> '
                '(compute.iterative=true)'.format(sim_type, promoted))

    return cfg


def _auto_wrap_substrate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-wrap MATLAB-style <materials.use_substrate=True> into the canonical
    <structure.type=with_substrate> form.

    Triggered when the YAML uses <materials.use_substrate>=True but
    <structure.type> is a plain particle type (not already with_substrate). The
    base structure config is preserved in <structure.base>, the substrate spec
    is normalized into <structure.substrate>, and <simulation.type> is promoted
    from <ret>/<stat>/<ret_iter> to the corresponding _layer variant.

    Substrate eps is resolved via:
      1. <materials.refractive_index_paths[<material>]> with type='constant'
         (epsilon -> EpsConst).
      2. <materials.refractive_index_paths[<material>]> with type='table'
         (file -> EpsTable spec, passed through as a string).
      3. Built-in substrate name (handled by WithSubstrateBuilder presets).
    """

    out = cfg
    materials = out.get('materials', None)

    if not isinstance(materials, dict):
        return out

    if not bool(materials.get('use_substrate', False)):
        return out

    cfg_struct = out.get('structure', None)

    if not isinstance(cfg_struct, dict):
        return out

    base_type = str(cfg_struct.get('type', '')).lower()

    if base_type == 'with_substrate':
        return out

    sub_spec = materials.get('substrate', dict())
    ri_paths = materials.get('refractive_index_paths', dict())
    material_name = sub_spec.get('material', 'glass') if isinstance(sub_spec, dict) else 'glass'

    if isinstance(sub_spec, dict) and ('position' in sub_spec or 'z_shift' in sub_spec):
        print('[warn] substrate spec 에 <position>/<z_shift> 발견 — 무시됨. '
                '<gap> 만 지원.')

    gap = float(sub_spec.get('gap', 0.001)) if isinstance(sub_spec, dict) else 0.001

    eps_resolved = _resolve_substrate_eps(material_name, ri_paths)

    out = copy.deepcopy(out)

    base_cfg = dict(out['structure'])
    out['structure'] = {
            'type': 'with_substrate',
            'base': base_cfg,
            'substrate': {
                    'eps': eps_resolved,
                    'gap': gap}}

    sim_cfg = out.get('simulation', None)

    if isinstance(sim_cfg, dict):
        sim_type = str(sim_cfg.get('type', 'ret')).lower()

        if sim_type in _SIM_LAYER_PROMOTE:
            sim_cfg['type'] = _SIM_LAYER_PROMOTE[sim_type]

    print('[info] auto-wrapped <structure.type={}> with substrate '
            '(material={}, eps={}, gap={}, sim.type -> {})'.format(
                    base_type, material_name, eps_resolved, gap,
                    out.get('simulation', dict()).get('type', '?')))

    return out


def _resolve_substrate_eps(name: Any, ri_paths: Any) -> Any:
    """Resolve substrate eps from MATLAB-style refractive_index_paths or fall
    back to a built-in preset name.

    Returns a value compatible with WithSubstrateBuilder._build_eps_substrate:
      - float: treated as eps directly via EpsConst.
      - str: preset name (glass/silicon/etc.) or *.dat path.
    """

    if not isinstance(name, str):
        return name

    if isinstance(ri_paths, dict):
        spec = ri_paths.get(name, None)

        if isinstance(spec, dict):
            stype = str(spec.get('type', '')).lower()

            if stype == 'constant':
                return float(spec['epsilon'])

            if stype == 'table':
                return str(spec.get('file', name))

    return name


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
