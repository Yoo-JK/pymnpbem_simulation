import os
import sys
import argparse

from typing import Any, Dict, List, Optional

import numpy as np


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_anal_conf(args)   # merge --anal-conf under CLI flags (CLI wins)

    from pymnpbem_simulation.util import (
            print_info, print_error, ensure_dir, save_json)
    from pymnpbem_simulation.postprocess import (
            analyze_spectrum, plot_spectrum,
            fano_fit, multi_fano_fit,
            qs_eigenmodes,
            multipole_decomposition, dipole_quadrupole_ratio, dominant_l,
            plot_multipole_bar, plot_fano_fit,
            plot_mode_patterns, plot_eigenvalue_spectrum,
            export_npz, export_h5, export_csv, export_json,
            export_spectrum_txt,
            check_unpolarized_conditions, calculate_unpolarized_spectrum,
            plot_polarization_comparison, plot_unpolarized_spectrum,
            plot_polarization_vs_unpolarized)

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

        title = os.path.basename(args.result)
        plot_spectrum(out_dir, result, title = title, xaxis = args.xaxis)

        n_pol = int(np.asarray(result['ext']).shape[1])
        if n_pol > 1:
            plot_polarization_comparison(out_dir, result, title = title, xaxis = args.xaxis)

        # Unpolarized — only if user passes --polarizations
        if args.polarizations is not None:
            try:
                pols = json.loads(args.polarizations)
            except Exception as e:
                print_error('failed to parse --polarizations as JSON: {}'.format(e))
                pols = None
            if pols is not None:
                info = check_unpolarized_conditions(pols, args.excitation, n_pol)
                if info.get('can_calculate', False):
                    unpol = calculate_unpolarized_spectrum(result, info)
                    plot_unpolarized_spectrum(out_dir, result, unpol,
                            title = title, xaxis = args.xaxis)
                    plot_polarization_vs_unpolarized(out_dir, result, unpol,
                            title = title, xaxis = args.xaxis)

    if 'fano' in analyzers:
        fano_res = _run_fano(result, n_peaks = args.fano_peaks)
        summary['fano'] = fano_res
        if fano_res is not None and args.fano_peaks <= 1:
            try:
                enei = np.asarray(result['wavelength'])
                ext = np.asarray(result['ext'])
                spectrum = ext[:, 0] if ext.ndim >= 2 else ext
                plot_fano_fit(out_dir, enei, spectrum, fano_res,
                        title = 'Fano fit (pol 0)')
            except Exception as e:
                print_error('plot_fano_fit failed: {}'.format(e))

    if 'eigenmode' in analyzers:
        eig_res = _run_eigenmode(result, args, out_dir)
        if eig_res is not None:
            summary['eigenmode'] = eig_res

    if 'fano-analysis' in analyzers:
        fa_res = _run_fano_analysis(args, out_dir)
        if fa_res is not None:
            summary['fano-analysis'] = fa_res

    if 'multipole' in analyzers:
        mp_res = _run_multipole(result, args, out_dir)
        if mp_res is not None:
            summary['multipole'] = mp_res
            try:
                # Reconstruct power_l + max_l for plot helper.
                plot_multipole_bar(out_dir, {
                        'power_l': np.asarray(mp_res['power_l']),
                        'max_l': args.max_l},
                        title = 'Multipole decomposition')
            except Exception as e:
                print_error('plot_multipole_bar failed: {}'.format(e))

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
        if 'txt' in formats:
            # spectrum txt export (per-pol + combined)
            try:
                export_spectrum_txt(out_dir, result,
                        title = os.path.basename(args.result))
            except Exception as e:
                print_error('export_spectrum_txt failed: {}'.format(e))

    print_info('postprocess done. results in <{}>'.format(out_dir))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            prog = 'pymnpbem_postprocess',
            description = 'Run postprocess analyzers on existing simulation result (.npz).')

    # NOTE: analyzer hyperparameters default to None so we can tell "not given
    # on the CLI" apart from an explicit value.  _apply_anal_conf() then fills
    # each None from --anal-conf (if provided) or the built-in default.
    # Precedence: explicit CLI flag > --anal-conf > built-in default.
    # (분석 하이퍼파라미터를 --anal-conf 로도 줄 수 있게; CLI 플래그가 최우선.)
    parser.add_argument('--result', type = str, default = None,
            help = 'Path to result .npz file (with wavelength/ext/sca/abs). '
                   'Required unless provided via --anal-conf (result=).')
    parser.add_argument('--anal-conf', type = str, default = None,
            help = 'Analysis config .py that defines args = {...} with analyzer '
                   'hyperparameters (same style as --str-conf/--sim-conf). '
                   'Explicit CLI flags override it. '
                   '(분석 하이퍼파라미터 config; --str-conf/--sim-conf 와 동일 스타일.)')
    parser.add_argument('--analyzers', type = str, default = None,
            help = 'Comma-separated: spectrum,fano,eigenmode,multipole,fano-analysis. '
                   '(default: spectrum)')
    parser.add_argument('--output', type = str, default = None,
            help = 'Output directory (default: alongside result).')

    parser.add_argument('--fano-peaks', type = int, default = None,
            help = 'Number of Fano peaks (>=2 uses multi_fano_fit). (default: 1)')

    parser.add_argument('--config', type = str, default = None,
            help = 'YAML config (needed for eigenmode/multipole — rebuilds particle).')
    parser.add_argument('--n-modes', type = int, default = None,
            help = 'Number of eigenmodes (qs_eigenmodes). (default: 10)')
    parser.add_argument('--max-l', type = int, default = None,
            help = 'Multipole expansion order. (default: 4)')

    parser.add_argument('--export-formats', type = str, default = None,
            help = 'Comma-separated formats: npz,h5,csv,json,txt.')

    # fano-analysis (bright/dark eigenmode + multi-Lorentzian Fano fit).
    parser.add_argument('--case-dir', type = str, default = None,
            help = 'Case directory (config.yaml + sigma/) for fano-analysis. '
                   'Defaults to the directory containing --result.')
    parser.add_argument('--fano-features', type = str, default = None,
            help = 'Comma-separated Fano dip energies in eV (e.g. 1.43,1.79,1.91).')
    parser.add_argument('--fano-pol', type = int, default = None,
            help = 'Polarization index for fano-analysis sigma. (default: 0)')
    parser.add_argument('--eig-cache', type = str, default = None,
            help = 'Path to the quasistatic full-eig .npz cache (keys: ene,vr,dvec). '
                   'Loaded if present, else computed and saved here (HEAVY).')

    parser.add_argument('--xaxis', type = str,
            choices = ['wavelength', 'energy'],
            default = None,
            help = 'Spectrum x-axis: wavelength(nm) (default) or energy(eV).')
    parser.add_argument('--polarizations', type = str, default = None,
            help = 'JSON list of polarization vectors '
                   '(e.g. \'[[1,0,0],[0,1,0]]\') for unpolarized comparison.')
    parser.add_argument('--excitation', type = str,
            choices = ['planewave', 'dipole', 'eels'],
            default = None,
            help = 'Excitation type for unpolarized check. (default: planewave)')

    return parser


# Analyzer hyperparameters that --anal-conf can set, with their built-in
# defaults.  Keys match the argparse dest names (underscored).
_ANAL_CONF_DEFAULTS = {
    'result': None, 'analyzers': 'spectrum', 'output': None,
    'fano_peaks': 1, 'config': None, 'n_modes': 10, 'max_l': 4,
    'export_formats': None, 'case_dir': None, 'fano_features': None,
    'fano_pol': 0, 'eig_cache': None, 'xaxis': 'wavelength',
    'polarizations': None, 'excitation': 'planewave',
}


def _apply_anal_conf(args: argparse.Namespace) -> None:
    """Merge --anal-conf (.py with args={...}) under the CLI flags.

    Precedence: explicit CLI flag > --anal-conf value > built-in default.
    A None on the namespace means the flag was not given on the CLI, so the
    config (or the built-in default) fills it in — this keeps the analyzer
    hyperparameters config-driven and reproducible, mirroring the
    --str-conf/--sim-conf pattern of run_simulation.py.
    (분석 하이퍼파라미터를 config 로 받아 재현 가능하게. 우선순위: CLI > anal-conf > 기본값.)
    """
    conf = {}
    if getattr(args, 'anal_conf', None):
        from pymnpbem_simulation.config import load_py_config
        conf = load_py_config(args.anal_conf)
        print('[info] loading anal-conf <{}> ({} keys)'.format(args.anal_conf, len(conf)))
        unknown = set(conf) - set(_ANAL_CONF_DEFAULTS)
        if unknown:
            print('[warn] anal-conf: unknown keys ignored: {}'.format(sorted(unknown)))
    for key, hard in _ANAL_CONF_DEFAULTS.items():
        if getattr(args, key, None) is not None:
            continue                      # explicit CLI flag wins
        setattr(args, key, conf.get(key, hard))
    # Allow natural Python types in the config: lists for the comma-string
    # options, and a nested list for polarizations (JSON string downstream).
    for key in ('analyzers', 'export_formats', 'fano_features'):
        val = getattr(args, key, None)
        if isinstance(val, (list, tuple)):
            setattr(args, key, ','.join(str(x) for x in val))
    if isinstance(getattr(args, 'polarizations', None), (list, tuple)):
        import json as _json
        args.polarizations = _json.dumps(args.polarizations)
    if not args.result:
        raise SystemExit('[error] --result (or result= in --anal-conf) is required.')


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

    # Auto-render eigenvalue spectrum + mode patterns (top n_modes).
    try:
        from pymnpbem_simulation.postprocess import (
                plot_eigenvalue_spectrum, plot_mode_patterns)
        plot_eigenvalue_spectrum(out_dir, eig, title = 'QS eigenmodes')
        plot_mode_patterns(out_dir, eig, p,
                n_modes = min(args.n_modes, 5),
                title = 'QS mode patterns')
    except Exception as e:
        print_error('eigenmode plots failed: {}'.format(e))

    return out


def _run_fano_analysis(args: argparse.Namespace,
        out_dir: str) -> Optional[Dict[str, Any]]:

    from pymnpbem_simulation.util import print_info, print_error
    from pymnpbem_simulation.postprocess import analyze_fano

    case_dir = args.case_dir
    if case_dir is None:
        case_dir = os.path.dirname(os.path.abspath(args.result))

    if not os.path.exists(os.path.join(case_dir, 'config.yaml')):
        print_error('fano-analysis: config.yaml not found in case dir <{}>'.format(case_dir))
        return None
    if not os.path.exists(os.path.join(case_dir, 'sigma', 'manifest.json')):
        print_error('fano-analysis: sigma cache (sigma/manifest.json) not found in <{}>'.format(case_dir))
        return None

    if args.fano_features is None:
        print_error('fano-analysis: --fano-features required (e.g. 1.43,1.79,1.91)')
        return None

    features = [float(x.strip()) for x in args.fano_features.split(',') if x.strip()]

    print_info('fano-analysis: case <{}>, features {} eV'.format(case_dir, features))

    summary = analyze_fano(case_dir, features, out_dir,
            pol = args.fano_pol, eig_cache_path = args.eig_cache)

    return summary.to_dict()


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
