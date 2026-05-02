import os
import sys
import argparse

from typing import Any, Dict, List, Optional

import numpy as np


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from pymnpbem_simulation.util import (
            print_info, print_error, ensure_dir, save_json)
    from pymnpbem_simulation.postprocess import (
            analyze_spectrum, plot_spectrum,
            fano_fit, multi_fano_fit,
            qs_eigenmodes,
            multipole_decomposition, dipole_quadrupole_ratio, dominant_l,
            export_npz, export_h5, export_csv, export_json)

    if not os.path.exists(args.result):
        print_error('result file not found: <{}>'.format(args.result))
        return 1

    if args.output is None:
        out_dir = os.path.join(os.path.dirname(args.result),
                'postprocess_' + os.path.splitext(os.path.basename(args.result))[0])
    else:
        out_dir = args.output

    ensure_dir(out_dir)

    print_info('loading result <{}>'.format(args.result))
    result = _load_result(args.result)

    analyzers = {a.strip() for a in args.analyzers.split(',') if a.strip()}
    print_info('analyzers: {}'.format(sorted(analyzers)))

    summary = dict()

    if 'spectrum' in analyzers:
        analysis = analyze_spectrum(result)
        summary['spectrum'] = analysis
        save_json(os.path.join(out_dir, 'spectrum_analysis.json'), analysis)
        plot_spectrum(out_dir, result,
                title = os.path.basename(args.result))

    if 'fano' in analyzers:
        fano_res = _run_fano(result, n_peaks = args.fano_peaks)
        summary['fano'] = fano_res

    if 'eigenmode' in analyzers:
        eig_res = _run_eigenmode(result, args, out_dir)
        if eig_res is not None:
            summary['eigenmode'] = eig_res

    if 'multipole' in analyzers:
        mp_res = _run_multipole(result, args, out_dir)
        if mp_res is not None:
            summary['multipole'] = mp_res

    save_json(os.path.join(out_dir, 'postprocess_summary.json'), summary)

    if args.export_formats:
        formats = {f.strip().lower() for f in args.export_formats.split(',') if f.strip()}
        export_data = _build_export_data(result, summary)
        if 'npz' in formats:
            export_npz(export_data, os.path.join(out_dir, 'export.npz'))
        if 'h5' in formats:
            export_h5(export_data, os.path.join(out_dir, 'export.h5'))
        if 'csv' in formats:
            export_csv(export_data, os.path.join(out_dir, 'export.csv'))
        if 'json' in formats:
            export_json(export_data, os.path.join(out_dir, 'export.json'))

    print_info('postprocess done. results in <{}>'.format(out_dir))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            prog = 'pymnpbem_postprocess',
            description = 'Run postprocess analyzers on existing simulation result (.npz).')

    parser.add_argument('--result', type = str, required = True,
            help = 'Path to result .npz file (with wavelength/ext/sca/abs).')
    parser.add_argument('--analyzers', type = str,
            default = 'spectrum',
            help = 'Comma-separated: spectrum,fano,eigenmode,multipole.')
    parser.add_argument('--output', type = str, default = None,
            help = 'Output directory (default: alongside result).')

    parser.add_argument('--fano-peaks', type = int, default = 1,
            help = 'Number of Fano peaks (>=2 uses multi_fano_fit).')

    parser.add_argument('--config', type = str, default = None,
            help = 'YAML config (needed for eigenmode/multipole — rebuilds particle).')
    parser.add_argument('--n-modes', type = int, default = 10,
            help = 'Number of eigenmodes (qs_eigenmodes).')
    parser.add_argument('--max-l', type = int, default = 4,
            help = 'Multipole expansion order.')

    parser.add_argument('--export-formats', type = str, default = None,
            help = 'Comma-separated formats: npz,h5,csv,json.')

    return parser


def _load_result(path: str) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle = True)

    out = dict()
    for k in npz.files:
        out[k] = npz[k]

    if 'ext' in out:
        ext = np.asarray(out['ext'])
        out['n_pol'] = int(ext.shape[1]) if ext.ndim >= 2 else 1
        out['peak_idx'] = int(np.argmax(ext[:, 0])) if ext.ndim >= 2 else int(np.argmax(ext))
        out['peak_wl_nm'] = float(out['wavelength'][out['peak_idx']])
        out['peak_ext_x'] = float(ext[out['peak_idx'], 0]) if ext.ndim >= 2 else float(ext[out['peak_idx']])

    out.setdefault('wall_s', 0.0)
    out.setdefault('warmup_s', 0.0)

    return out


def _run_fano(result: Dict[str, Any],
        n_peaks: int = 1) -> Dict[str, Any]:

    from pymnpbem_simulation.postprocess import fano_fit, multi_fano_fit

    enei = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])

    spectrum = ext[:, 0] if ext.ndim >= 2 else ext

    if n_peaks <= 1:
        fit = fano_fit(enei, spectrum)
    else:
        fit = multi_fano_fit(enei, spectrum, n_peaks = n_peaks)

    out = dict(fit)
    out['fit_curve'] = np.asarray(out['fit_curve']).tolist()
    return out


def _run_eigenmode(result: Dict[str, Any],
        args: argparse.Namespace,
        out_dir: str) -> Optional[Dict[str, Any]]:

    from pymnpbem_simulation.util import print_info, print_error
    from pymnpbem_simulation.postprocess import qs_eigenmodes

    p = _try_rebuild_particle(args)
    if p is None:
        print_error('eigenmode: cannot rebuild particle without --config')
        return None

    eig = qs_eigenmodes(p, n_modes = args.n_modes)

    out = {
        'n_modes': int(eig['n_modes']),
        'eigenvalues_real': np.real(eig['eigenvalues']).tolist(),
        'eigenvalues_imag': np.imag(eig['eigenvalues']).tolist(),
        'inv_eps_p_real': np.real(eig['inv_eps_p']).tolist()}

    np.savez_compressed(
            os.path.join(out_dir, 'eigenmodes.npz'),
            eigenvalues = eig['eigenvalues'],
            eigenvectors_r = eig['eigenvectors_r'],
            eigenvectors_l = eig['eigenvectors_l'])
    print_info('saved <{}>'.format(os.path.join(out_dir, 'eigenmodes.npz')))

    return out


def _run_multipole(result: Dict[str, Any],
        args: argparse.Namespace,
        out_dir: str) -> Optional[Dict[str, Any]]:

    from pymnpbem_simulation.util import print_info, print_error
    from pymnpbem_simulation.postprocess import (
            multipole_decomposition, dipole_quadrupole_ratio, dominant_l)

    if 'sig_peak' not in result and 'sig' not in result:
        print_error('multipole: need <sig_peak> or <sig> in result')
        return None

    p = _try_rebuild_particle(args)
    if p is None:
        print_error('multipole: cannot rebuild particle without --config')
        return None

    sig_arr = np.asarray(result.get('sig_peak', result.get('sig')))
    if sig_arr.ndim > 1:
        sig_arr = sig_arr[:, 0]

    mp = multipole_decomposition(sig_arr, p, max_l = args.max_l)

    out = {
        'l': mp['l'].tolist(),
        'm': mp['m'].tolist(),
        'amplitudes_abs': np.abs(mp['amplitudes']).tolist(),
        'power_l': mp['power_l'].tolist(),
        'dipole_quadrupole_ratio': float(dipole_quadrupole_ratio(mp)),
        'dominant_l': int(dominant_l(mp))}

    return out


def _try_rebuild_particle(args: argparse.Namespace) -> Optional[Any]:
    if args.config is None:
        return None

    from pymnpbem_simulation.config import load_yaml, apply_defaults
    from pymnpbem_simulation.structures import build_structure

    cfg = load_yaml(args.config)
    cfg = apply_defaults(cfg)

    p, _epstab, _nfaces = build_structure(cfg['structure'], cfg.get('materials', dict()))
    return p


def _build_export_data(result: Dict[str, Any],
        summary: Dict[str, Any]) -> Dict[str, Any]:

    out = dict()

    for k in ('wavelength', 'ext', 'sca', 'abs'):
        if k in result:
            out[k] = np.asarray(result[k])

    if 'fano' in summary and summary['fano'] is not None:
        bf = summary['fano'].get('best_fit', {})
        for key, val in bf.items():
            out['fano_{}'.format(key)] = val
        if 'fit_curve' in summary['fano']:
            out['fano_fit_curve'] = np.asarray(summary['fano']['fit_curve'])

    return out


if __name__ == '__main__':
    sys.exit(main())
