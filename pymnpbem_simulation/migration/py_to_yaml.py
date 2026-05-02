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

    return out


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description = 'Convert legacy mnpbem_simulation .py config to YAML.')
    parser.add_argument('input_str', type = str, nargs = '?', default = '',
            help = 'Path to config_str_*.py (structure config). Empty for sim-only.')
    parser.add_argument('input_sim', type = str,
            help = 'Path to config_sim_*.py (simulation config).')
    parser.add_argument('output_yaml', type = str,
            help = 'Output YAML path.')

    args = parser.parse_args(argv)

    try:
        path_str = args.input_str if args.input_str != '' else None
        convert_py_to_yaml(path_str, args.input_sim, args.output_yaml)
        return 0
    except Exception as e:
        print_error('migration failed: {}'.format(e))
        return 1


if __name__ == '__main__':
    sys.exit(main())
