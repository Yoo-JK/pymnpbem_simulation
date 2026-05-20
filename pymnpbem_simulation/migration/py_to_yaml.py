import os
import sys
import argparse
import codecs

from typing import Any, Dict, List, Optional

import yaml

from ..util import print_info, print_error, ensure_dir


_KEY_TO_SECTION = {
    'structure': ('structure', 'type'),
    'structure_name': ('output', 'name'),
    'simulation_name': ('output', 'name'),
    'simulation_type': ('simulation', 'type'),
    'excitation_type': ('simulation', 'excitation'),
    'wavelength_range': ('simulation', 'wavelength_range'),
    'polarizations': ('simulation', 'polarizations'),
    'propagation_dirs': ('simulation', 'propagation_dirs'),
    'dipole_position': ('simulation', 'dipole_position'),
    'dipole_moment': ('simulation', 'dipole_moment'),
    'impact_parameter': ('simulation', 'impact_parameter'),
    'beam_energy': ('simulation', 'beam_energy'),
    'beam_width': ('simulation', 'beam_width'),
    'interp': ('simulation', 'interp'),
    'refine': ('structure', 'refine'),
    'relcutoff': ('simulation', 'relcutoff'),
    'waitbar': ('simulation', 'waitbar'),
    'use_parallel': ('compute', 'use_parallel'),
    'num_workers': ('compute', 'n_workers'),
    'max_comp_threads': ('compute', 'n_threads'),
    'wavelength_chunk_size': ('compute', 'wavelength_chunk_size'),
    'use_mirror_symmetry': ('compute', 'mirror'),
    'use_iterative_solver': ('compute', 'iterative'),
    'use_nonlocality': ('compute', 'nonlocal'),
    'use_h2_compression': ('compute', 'hmode'),
    'gpu_precision': ('compute', 'gpu_precision'),
    'medium': ('materials', 'medium'),
    'materials': ('materials', 'particle_list'),
    'substrate': ('materials', 'substrate'),
    'use_substrate': ('materials', 'use_substrate'),
    'refractive_index_paths': ('materials', 'refractive_index_paths'),
    'diameter': ('structure', 'diameter'),
    'size': ('structure', 'size'),
    'gap': ('structure', 'gap'),
    'rounding': ('structure', 'rounding'),
    'roundings': ('structure', 'roundings'),
    'mesh_density': ('structure', 'mesh_density'),
    'core_size': ('structure', 'core_size'),
    'shell_layers': ('structure', 'shell_layers'),
    'offset': ('structure', 'offset'),
    'tilt_angle': ('structure', 'tilt_angle'),
    'tilt_axis': ('structure', 'tilt_axis'),
    'rotation_angle': ('structure', 'rotation_angle'),
    'n_spheres': ('structure', 'n_spheres'),
    'shape_file': ('structure', 'shape_file'),
    'voxel_size': ('structure', 'voxel_size'),
    'voxel_method': ('structure', 'voxel_method'),
    'output_dir': ('output', 'dir'),
    'output_formats': ('output', 'formats'),
    'save_plots': ('output', 'save_plots'),
    'plot_format': ('output', 'plot_format'),
    'plot_dpi': ('output', 'plot_dpi'),
    'spectrum_xaxis': ('postprocess', 'spectrum_xaxis'),
    'calculate_cross_sections': ('simulation', 'calculate_cross_sections'),
    'calculate_fields': ('simulation', 'calculate_fields'),
    'field_region': ('simulation', 'field_region'),
    'field_mindist': ('simulation', 'field_mindist'),
    'field_nmax': ('simulation', 'field_nmax'),
    'field_wavelength_idx': ('simulation', 'field_wavelength_idx'),
    'export_field_arrays': ('simulation', 'export_field_arrays'),
    'field_hotspot_count': ('simulation', 'field_hotspot_count'),
    'field_hotspot_min_distance': ('simulation', 'field_hotspot_min_distance'),
    'run_eigenmode_analysis': ('postprocess', 'run_eigenmode_analysis'),
    'eigenmode_n': ('postprocess', 'eigenmode_n'),
    'eigenmode_top_k': ('postprocess', 'eigenmode_top_k'),
    'retarded_eigen_wavelength': ('postprocess', 'retarded_eigen_wavelength'),
    'fano_target_wavelengths': ('postprocess', 'fano_target_wavelengths'),
    'svd_rank_threshold': ('postprocess', 'svd_rank_threshold')}


_DROP_KEYS = {
    'mnpbem_path',
    'matlab_executable',
    'matlab_options'}


def load_py_args(path: str) -> Dict[str, Any]:
    with codecs.open(path, 'r', encoding = 'utf-8') as f:
        src = f.read()

    namespace = dict()
    exec(src, namespace)

    if 'args' not in namespace:
        raise ValueError("[error] <{}> does not define 'args' dict!".format(path))

    if not isinstance(namespace['args'], dict):
        raise ValueError("[error] 'args' in <{}> is not a dict!".format(path))

    return namespace['args']


def merge_args(args_str: Dict[str, Any],
        args_sim: Dict[str, Any]) -> Dict[str, Any]:

    out = dict(args_str)
    out.update(args_sim)

    return out


def convert_args_to_yaml(args: Dict[str, Any]) -> Dict[str, Any]:
    out = dict()
    unmapped = []

    for k, v in args.items():

        if k in _DROP_KEYS:
            continue

        if k not in _KEY_TO_SECTION:
            unmapped.append(k)
            continue

        section, sub_key = _KEY_TO_SECTION[k]

        if section not in out:
            out[section] = dict()

        out[section][sub_key] = _normalize_value(k, v)

    if unmapped:
        if 'extras' not in out:
            out['extras'] = dict()
        for k in unmapped:
            out['extras'][k] = args[k]
        print_info('merged {} unmapped keys into <extras>: {}'.format(
            len(unmapped), unmapped))

    out = _post_process(out)

    return out


def _normalize_value(key: str,
        v: Any) -> Any:

    if key == 'use_h2_compression':

        if isinstance(v, bool):
            return 'aca-gpu' if v else 'dense'

        return v

    if key == 'num_workers':

        if v == 'auto':
            return -1

        if v == 'env':
            return 'env'

        return int(v) if isinstance(v, (int, float)) else v

    if key == 'max_comp_threads':

        if v == 'auto':
            return -1

        if v == 'max':
            return -1

        return int(v) if isinstance(v, (int, float)) else v

    return v


def _post_process(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)

    if 'simulation' in out and 'wavelength_range' in out['simulation']:
        wr = out['simulation']['wavelength_range']
        if isinstance(wr, (list, tuple)) and len(wr) == 3:
            out['simulation']['enei_min'] = wr[0]
            out['simulation']['enei_max'] = wr[1]
            out['simulation']['n_wavelengths'] = wr[2]

    out = _redirect_field_only_simulation(out)
    out = _redirect_iterative_to_iter_type(out)

    return out


# Issue A (v1.5.1) — translate compute.iterative=true into the matching
# _iter simulation.type so legacy py configs convert to a YAML that
# routes to BEMRetIter / BEMStatIter without further user intervention.
_ITER_TYPE_MAP = {
        'ret': 'ret_iter',
        'stat': 'stat_iter',
        'ret_layer': 'ret_layer_iter'}


def _redirect_iterative_to_iter_type(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of dispatch.single_node._redirect_iterative_to_iter_type
    applied at YAML migration time. Keeps the on-disk YAML self-consistent
    (simulation.type=ret_iter) so users see what will actually run."""
    if not isinstance(cfg, dict):
        return cfg

    compute = cfg.get('compute', dict())
    iterative = bool(compute.get('iterative', False))

    if not iterative:
        return cfg

    sim = cfg.get('simulation', dict())
    sim_type = sim.get('type', 'ret')

    if not isinstance(sim_type, str):
        return cfg

    if sim_type.endswith('_iter'):
        return cfg

    new_type = _ITER_TYPE_MAP.get(sim_type, None)
    if new_type is None:
        return cfg

    print_info(
            'migration iterative redirect (Issue A): simulation.type <{}> -> <{}>'.format(
                    sim_type, new_type))

    sim = dict(sim)
    sim['type'] = new_type
    out = dict(cfg)
    out['simulation'] = sim
    return out


def _redirect_field_only_simulation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """If the source config is field-only (no spectrum, fields requested),
    redirect ``simulation.type`` to ``field`` and synthesise the ``grid``
    block expected by FieldCalculator from the legacy ``field_region``.

    Triggered when:
      - ``simulation.calculate_cross_sections`` is False, AND
      - ``simulation.calculate_fields`` is True OR ``field_region`` is set.

    Without this redirect the YAML keeps ``simulation.type == 'ret'`` (or
    similar) so the spectrum runner is invoked and never produces field
    output (Issue 4 — jk-config dimer_au_r0.2/au_r0.2_g*.yaml family).
    """
    if 'simulation' not in cfg:
        return cfg

    sim = dict(cfg['simulation'])

    cross = sim.get('calculate_cross_sections', True)
    fields = sim.get('calculate_fields', False)
    has_region = 'field_region' in sim

    if cross is True or cross is None:
        return cfg
    if not (fields is True or has_region):
        return cfg

    # Stash the spectrum sim type (informational; postprocess may still
    # want to know the original variant). Switch active type to 'field'.
    original_type = sim.get('type', 'ret')
    sim['original_type'] = original_type
    sim['type'] = 'field'

    # Build grid block from field_region (legacy schema:
    # x_range = [xmin, xmax, npts], same for y_range, z_range).
    region = sim.pop('field_region', None)
    grid = sim.get('grid', None)
    if grid is None and region is not None:
        grid = _grid_from_field_region(region)
    if grid is not None:
        sim['grid'] = grid

    # Map legacy field_* keys onto FieldCalculator inputs.
    if 'field_mindist' in sim and 'mindist' not in sim:
        sim['mindist'] = sim['field_mindist']
    if 'field_nmax' in sim and 'nmax' not in sim:
        sim['nmax'] = sim['field_nmax']

    out = dict(cfg)
    out['simulation'] = sim
    return out


def _grid_from_field_region(region: Dict[str, Any]) -> Dict[str, Any]:
    grid = {'type': 'rectangular'}

    x_range = region.get('x_range', None)
    y_range = region.get('y_range', None)
    z_range = region.get('z_range', None)

    n_points = []
    if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
        grid['x_range'] = [float(x_range[0]), float(x_range[1])]
        n_points.append(int(x_range[2]) if len(x_range) >= 3 else 1)
    if isinstance(y_range, (list, tuple)) and len(y_range) >= 2:
        grid['y_range'] = [float(y_range[0]), float(y_range[1])]
        n_points.append(int(y_range[2]) if len(y_range) >= 3 else 1)
    if isinstance(z_range, (list, tuple)) and len(z_range) >= 2:
        grid['z_range'] = [float(z_range[0]), float(z_range[1])]
        n_points.append(int(z_range[2]) if len(z_range) >= 3 else 1)

    if len(n_points) == 3:
        grid['n_points'] = n_points

    return grid


def convert_py_to_yaml(path_str: Optional[str],
        path_sim: str,
        path_out: str) -> Dict[str, Any]:

    if path_str is None or path_str == '':
        args = load_py_args(path_sim)
    else:
        args_str = load_py_args(path_str)
        args_sim = load_py_args(path_sim)
        args = merge_args(args_str, args_sim)

    cfg = convert_args_to_yaml(args)

    ensure_dir(os.path.dirname(os.path.abspath(path_out)))
    with open(path_out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys = False, default_flow_style = None)

    print_info('saved YAML <{}>'.format(path_out))

    return cfg


def upgrade_yaml(path_in: str, path_out: Optional[str] = None) -> Dict[str, Any]:
    """Re-run the migration post-processors on an already-converted YAML.

    Useful for upgrading v1.4 YAMLs (e.g. jk-config branch) so the new
    field-only redirect (Issue 4) takes effect without re-running py->yaml.
    """
    with codecs.open(path_in, 'r', encoding = 'utf-8') as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("[error] <{}> is not a YAML mapping!".format(path_in))

    cfg = _post_process(cfg)

    if path_out is None:
        path_out = path_in

    ensure_dir(os.path.dirname(os.path.abspath(path_out)))
    with open(path_out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys = False, default_flow_style = None)

    print_info('upgraded YAML <{}> -> <{}>'.format(path_in, path_out))
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description = 'Convert legacy mnpbem_simulation .py config to YAML.')
    parser.add_argument('input_str', type = str, nargs = '?', default = '',
            help = 'Path to config_str_*.py (structure config). Empty for sim-only.')
    parser.add_argument('input_sim', type = str,
            help = 'Path to config_sim_*.py (simulation config).')
    parser.add_argument('output_yaml', type = str,
            help = 'Output YAML path.')
    parser.add_argument('--upgrade-yaml', action = 'store_true',
            help = 'Treat <input_sim> as an existing YAML and re-run post-'
                    'processors only (in-place if no <output_yaml>).')

    args = parser.parse_args(argv)

    try:
        if args.upgrade_yaml:
            out_path = args.output_yaml if args.output_yaml else args.input_sim
            upgrade_yaml(args.input_sim, out_path)
            return 0
        path_str = args.input_str if args.input_str != '' else None
        convert_py_to_yaml(path_str, args.input_sim, args.output_yaml)
        return 0
    except Exception as e:
        print_error('migration failed: {}'.format(e))
        return 1


if __name__ == '__main__':
    sys.exit(main())
