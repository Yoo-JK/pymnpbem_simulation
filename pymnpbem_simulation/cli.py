import os
import sys
import argparse
import time

from typing import Any, Dict, List, Optional

import numpy as np


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from .util import print_info, print_error, ensure_dir, save_json, now_str
    from .config import load_yaml, merge_overrides, apply_defaults, validate_config, save_yaml
    from .auto_detect import detect_n_gpus, detect_n_cpus, auto_compute_plan, detect_multi_node, detect_mpi_rank
    from .env_setup import setup_env, assert_pre_import

    try:
        cfg = load_yaml(args.config)
    except Exception as e:
        print_error('failed to load config <{}>: {}'.format(args.config, e))
        return 1

    overrides = _build_overrides(args)
    cfg = merge_overrides(cfg, overrides)

    if args.auto:
        n_w, n_t, n_g = auto_compute_plan()
        is_multi_node = detect_multi_node()
        print_info('auto: n_workers={}, n_threads={}, n_gpus_per_worker={}, multi_node={}'.format(
            n_w, n_t, n_g, is_multi_node))
        if cfg.get('compute', {}).get('n_workers', 1) == 1:
            cfg.setdefault('compute', {})
            cfg['compute']['n_workers'] = n_w
            cfg['compute']['n_threads'] = n_t
            cfg['compute']['n_gpus_per_worker'] = n_g

        if is_multi_node and not cfg.get('compute', {}).get('multi_node', False):
            cfg.setdefault('compute', {})
            cfg['compute']['multi_node'] = True

    cfg = apply_defaults(cfg)

    try:
        validate_config(cfg)
    except ValueError as e:
        print_error('config validation failed: {}'.format(e))
        return 2

    n_threads = int(cfg['compute']['n_threads'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    assert_pre_import()
    setup_env(n_threads, n_gpus_per_worker)
    print_info('env: n_threads={}, n_gpus_per_worker={}'.format(
        n_threads, n_gpus_per_worker))

    out_root = cfg['output']['dir']
    out_name = cfg['output']['name']
    out_dir = os.path.join(out_root, out_name)
    ensure_dir(out_dir)

    snapshot_path = os.path.join(out_dir, 'config.yaml')
    save_yaml(snapshot_path, cfg)
    print_info('saved config snapshot <{}>'.format(snapshot_path))

    enei = _build_enei(cfg, args.n_wavelengths)
    print_info('wavelengths: {} points from {:.1f} to {:.1f} nm'.format(
        len(enei), float(enei[0]), float(enei[-1])))

    from .structures import build_structure
    from .dispatch import dispatch_single_node
    from .io import save_spectrum, save_field, save_run_metadata
    from .postprocess import analyze_spectrum, plot_spectrum

    cfg_struct = cfg['structure']
    cfg_materials = cfg.get('materials', dict())

    p, epstab, nfaces = build_structure(cfg_struct, cfg_materials)

    save_run_metadata(out_dir, cfg, nfaces, timestamp = now_str())

    if args.reanalyze:
        print_info('--reanalyze: skipping simulation, postprocess only')
        return _reanalyze(out_dir)

    t0 = time.time()
    result = dispatch_single_node(cfg, p, epstab, enei)
    total_s = time.time() - t0

    if result is None:
        rank = detect_mpi_rank()
        print_info('mpi rank {}: non-zero rank, exiting cleanly'.format(rank))
        return 0

    print_info('dispatch finished in {:.2f} min'.format(total_s / 60.0))

    if result.get('kind', None) == 'field':
        save_field(out_dir, result)
        print_info('done (field calculation). results in <{}>'.format(out_dir))
        return 0

    save_spectrum(out_dir, result)

    analysis = analyze_spectrum(result)
    save_json(os.path.join(out_dir, 'spectrum_analysis.json'), analysis)

    if cfg['output'].get('save_plots', True):
        _create_spectrum_plots(out_dir, result, cfg, out_name)

    sc = result.get('surface_charge', None)
    if sc is not None and cfg['output'].get('save_plots', True):
        from .postprocess import plot_all_surface_charge

        plot_format = cfg['output'].get('plot_format', ['png'])
        dpi = int(cfg['output'].get('plot_dpi', 200))
        pol_labels = _make_polarization_labels(cfg)

        files = plot_all_surface_charge(out_dir, sc,
                plot_format = plot_format, dpi = dpi,
                polarization_labels = pol_labels, verbose = args.verbose)
        print_info('surface_charge: saved {} plot file(s)'.format(len(files)))

    print_info('done. results in <{}>'.format(out_dir))

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = 'pymnpbem_simulation',
        description = 'Run MNPBEM (Python port) electromagnetic simulation from YAML config.')

    parser.add_argument('--config', type = str, required = True,
            help = 'Path to YAML config file.')

    parser.add_argument('--n-workers', type = int, default = None,
            help = 'Number of worker processes (overrides YAML).')
    parser.add_argument('--n-threads', type = int, default = None,
            help = 'BLAS/OMP threads per worker.')
    parser.add_argument('--n-gpus-per-worker', type = int, default = None,
            help = 'GPUs per worker (0 = CPU, 1 = single GPU, 2+ = VRAM pool).')
    parser.add_argument('--multi-node', action = 'store_true',
            help = 'Enable mpi4py multi-node dispatch (requires mpi4py + srun/mpirun).')
    parser.add_argument('--vram-share-backend',
            choices = ['cusolvermg', 'magma', 'nccl'],
            default = None,
            help = 'Backend for VRAM share (n-gpus-per-worker > 1). Planned for M5+ release.')
    parser.add_argument('--auto', action = 'store_true',
            help = 'Auto-detect compute plan from SLURM/PBS environment.')

    parser.add_argument('--output-dir', type = str, default = None,
            help = 'Override output directory (YAML override).')
    parser.add_argument('--simulation-name', type = str, default = None,
            help = 'Override simulation name (folder).')
    parser.add_argument('--n-wavelengths', type = int, default = None,
            help = 'Sub-sample wavelength count (for debugging).')
    parser.add_argument('--reanalyze', action = 'store_true',
            help = 'Skip simulation, only run postprocess on existing results.')
    parser.add_argument('--verbose', action = 'store_true',
            help = 'Verbose logging.')

    return parser


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'compute': {
            'n_workers': args.n_workers,
            'n_threads': args.n_threads,
            'n_gpus_per_worker': args.n_gpus_per_worker,
            'multi_node': args.multi_node if args.multi_node else None,
            'vram_share_backend': args.vram_share_backend},
        'output': {
            'dir': args.output_dir,
            'name': args.simulation_name}}


def _build_enei(cfg: Dict[str, Any],
        override_n: Optional[int]) -> np.ndarray:

    sim = cfg['simulation']

    if 'wavelength_range' in sim:
        wr = sim['wavelength_range']
        e_min, e_max, n_wl = wr[0], wr[1], wr[2]
    else:
        e_min = sim['enei_min']
        e_max = sim['enei_max']
        n_wl = sim['n_wavelengths']

    if override_n is not None:
        n_wl = int(override_n)

    return np.linspace(float(e_min), float(e_max), int(n_wl))


def _reanalyze(out_dir: str) -> int:
    from .util import print_info, print_error, save_json
    from .config import load_yaml
    from .postprocess import (analyze_spectrum,
            plot_all_surface_charge, load_surface_charge_from_npz)

    npz_path = os.path.join(out_dir, 'spectrum.npz')

    if not os.path.exists(npz_path):
        print_error('reanalyze: <{}> not found'.format(npz_path))
        return 4

    npz = np.load(npz_path)
    result = {
        'wavelength': npz['wavelength'],
        'ext': npz['ext'],
        'sca': npz['sca'],
        'abs': npz['abs'],
        'wall_s': 0.0,
        'warmup_s': 0.0,
        'peak_idx': int(np.argmax(npz['ext'][:, 0])),
        'peak_wl_nm': float(npz['wavelength'][int(np.argmax(npz['ext'][:, 0]))]),
        'peak_ext_x': float(npz['ext'][:, 0].max()),
        'n_pol': int(npz['ext'].shape[1])}

    analysis = analyze_spectrum(result)
    save_json(os.path.join(out_dir, 'spectrum_analysis.json'), analysis)

    # Try to load config snapshot for richer plots (xaxis, polarization labels, unpolarized).
    cfg_path = os.path.join(out_dir, 'config.yaml')
    cfg = load_yaml(cfg_path) if os.path.exists(cfg_path) else dict()
    cfg.setdefault('output', dict())

    _create_spectrum_plots(out_dir, result, cfg, os.path.basename(out_dir))

    sc = load_surface_charge_from_npz(npz)

    if sc is not None:
        files = plot_all_surface_charge(out_dir, sc, plot_format = ['png'],
                dpi = 200, verbose = True)
        print_info('reanalyze surface_charge: saved {} plot file(s)'.format(len(files)))

    return 0


def _create_spectrum_plots(out_dir: str,
        result: Dict[str, Any],
        cfg: Dict[str, Any],
        out_name: str) -> None:
    """Generate the full spectrum plot set (basic + polarization + unpolarized).

    Driven by the YAML config so users can opt-in via:
        output.spectrum_xaxis: 'wavelength' | 'energy'
        output.plot_format: ['png', 'pdf', ...]
        output.plot_dpi: 150
    Unpolarized plots are auto-emitted when polarizations satisfy orthogonality.
    """
    from .postprocess import (plot_spectrum, plot_polarization_comparison,
            plot_unpolarized_spectrum, plot_polarization_vs_unpolarized,
            check_unpolarized_conditions, calculate_unpolarized_spectrum,
            export_spectrum_txt)

    output = cfg.get('output', dict())
    sim = cfg.get('simulation', dict())

    xaxis = output.get('spectrum_xaxis', 'wavelength')
    plot_format = output.get('plot_format', ['png'])
    if isinstance(plot_format, str):
        plot_format = [plot_format]
    dpi = int(output.get('plot_dpi', 150))

    pol_labels = _make_polarization_labels(cfg)

    # Basic spectrum plot (saved as spectrum.{fmt}).
    plot_spectrum(out_dir, result, title = out_name,
            xaxis = xaxis, polarization_labels = pol_labels,
            plot_format = plot_format, dpi = dpi)

    # Polarization comparison (only when n_pol > 1).
    n_pol = int(np.asarray(result['ext']).shape[1])
    if n_pol > 1:
        plot_polarization_comparison(out_dir, result, title = out_name,
                xaxis = xaxis, polarization_labels = pol_labels,
                plot_format = plot_format, dpi = dpi)

    # Unpolarized spectrum (only when conditions satisfied). Support both
    # canonical key (simulation.excitation) and legacy alias (excitation_type).
    excitation_type = sim.get('excitation', sim.get('excitation_type', 'planewave'))
    polarizations = sim.get('polarizations', None)

    info = check_unpolarized_conditions(polarizations, excitation_type, n_pol)
    unpol = None
    if info.get('can_calculate', False):
        from .util import print_info
        print_info('unpolarized: {} ({})'.format(info['method'], info['reason']))
        unpol = calculate_unpolarized_spectrum(result, info)
        plot_unpolarized_spectrum(out_dir, result, unpol, title = out_name,
                xaxis = xaxis, plot_format = plot_format, dpi = dpi)
        plot_polarization_vs_unpolarized(out_dir, result, unpol, title = out_name,
                xaxis = xaxis, polarization_labels = pol_labels,
                plot_format = plot_format, dpi = dpi)

    # Optional .txt spectrum export when configured.
    formats = output.get('formats', [])
    if isinstance(formats, str):
        formats = [formats]
    if 'txt' in [str(f).lower() for f in formats]:
        export_spectrum_txt(out_dir, result,
                polarization_labels = pol_labels,
                unpolarized = unpol,
                title = out_name)


def _make_polarization_labels(cfg: Dict[str, Any]) -> List[str]:

    pols = cfg.get('simulation', dict()).get('polarizations',
            [[1, 0, 0], [0, 1, 0]])

    labels = []

    for pol in pols:
        arr = np.asarray(pol).flatten()

        if arr.size >= 3:
            labels.append('[{:.0f} {:.0f} {:.0f}]'.format(
                    float(arr[0]), float(arr[1]), float(arr[2])))
        else:
            labels.append(str(arr.tolist()))

    return labels


if __name__ == '__main__':
    sys.exit(main())
