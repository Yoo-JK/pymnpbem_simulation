import os
import json

from typing import Any, Dict, List, Optional

import numpy as np

from ..util import ensure_dir, print_info


def export_npz(result: Dict[str, Any],
        output_path: str) -> str:

    ensure_dir(os.path.dirname(output_path) or '.')

    cleaned = dict()
    for k, v in result.items():
        if isinstance(v, (list, tuple)):
            cleaned[k] = np.asarray(v)
        elif isinstance(v, np.ndarray):
            cleaned[k] = v
        elif isinstance(v, (int, float, complex, np.number)):
            cleaned[k] = np.asarray(v)
        else:
            # fallback: store dict / nested — npz cannot hold; skip
            continue

    np.savez_compressed(output_path, **cleaned)
    print_info('export_npz: <{}>'.format(output_path))
    return output_path


def export_h5(result: Dict[str, Any],
        output_path: str) -> str:

    try:
        import h5py
    except ImportError:
        raise ImportError('[error] h5py not installed; pip install h5py')

    ensure_dir(os.path.dirname(output_path) or '.')

    with h5py.File(output_path, 'w') as f:
        _h5_write_recursive(f, result)

    print_info('export_h5: <{}>'.format(output_path))
    return output_path


def _h5_write_recursive(group: Any,
        d: Dict[str, Any]) -> None:

    for k, v in d.items():
        if isinstance(v, dict):
            sub = group.create_group(k)
            _h5_write_recursive(sub, v)
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v)
            group.create_dataset(k, data = arr)
        elif isinstance(v, np.ndarray):
            group.create_dataset(k, data = v)
        elif isinstance(v, (int, float, complex, np.number)):
            group.create_dataset(k, data = np.asarray(v))
        elif isinstance(v, str):
            group.attrs[k] = v
        else:
            group.attrs[k] = str(v)


def export_csv(result: Dict[str, Any],
        output_path: str) -> str:

    ensure_dir(os.path.dirname(output_path) or '.')

    columns = []
    headers = []

    for k, v in result.items():
        if isinstance(v, (list, tuple)):
            v = np.asarray(v)
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                columns.append(v)
                headers.append(k)
            elif v.ndim == 2:
                for j in range(v.shape[1]):
                    columns.append(v[:, j])
                    headers.append('{}_{}'.format(k, j))

    if not columns:
        with open(output_path, 'w') as f:
            f.write('# no 1D arrays in result\n')
        return output_path

    n_rows = max(len(c) for c in columns)
    padded = []
    for c in columns:
        if len(c) < n_rows:
            c2 = np.full(n_rows, np.nan, dtype = float)
            c2[:len(c)] = c
            padded.append(c2)
        else:
            padded.append(c[:n_rows])

    matrix = np.column_stack([np.asarray(c, dtype = float) for c in padded])

    header_line = ','.join(headers)
    np.savetxt(output_path, matrix, delimiter = ',',
            header = header_line, comments = '')

    print_info('export_csv: <{}> (rows={}, cols={})'.format(
            output_path, matrix.shape[0], matrix.shape[1]))
    return output_path


def export_spectrum_txt(out_dir: str,
        result: Dict[str, Any],
        polarization_labels: Optional[List[str]] = None,
        unpolarized: Optional[Dict[str, Any]] = None,
        title: str = '') -> List[str]:
    """Export per-polarization, unpolarized, and combined .txt spectrum files.

    Files written:
        spectra_pol1.txt, spectra_pol2.txt, ...     (per polarization)
        spectra_unpolarized.txt                      (if unpolarized provided)
        spectra_all.txt                              (combined columns)
    """
    ensure_dir(out_dir)

    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])
    abs_ = np.asarray(result['abs'])

    n_pol = ext.shape[1]

    if polarization_labels is None:
        polarization_labels = ['pol{}'.format(i + 1) for i in range(n_pol)]

    saved_files = []

    # Per-polarization files
    for ipol in range(n_pol):
        pol_label = polarization_labels[ipol] if ipol < len(polarization_labels) \
                else 'pol{}'.format(ipol + 1)

        path = os.path.join(out_dir, 'spectra_pol{}.txt'.format(ipol + 1))
        header = _format_spectrum_header(
                'Polarization {}: {}'.format(ipol + 1, pol_label),
                ['wavelength(nm)', 'scattering(nm^2)', 'extinction(nm^2)', 'absorption(nm^2)'],
                title = title)

        data_array = np.column_stack([
                wavelength,
                sca[:, ipol],
                ext[:, ipol],
                abs_[:, ipol]])

        _write_txt(path, header, data_array)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    # Unpolarized file
    if unpolarized is not None:
        path = os.path.join(out_dir, 'spectra_unpolarized.txt')

        method = unpolarized.get('method', 'unknown')
        n_avg = unpolarized.get('n_averaged', '?')
        header = _format_spectrum_header(
                'Unpolarized (averaged from {} polarizations, method: {})'.format(n_avg, method),
                ['wavelength(nm)', 'scattering(nm^2)', 'extinction(nm^2)', 'absorption(nm^2)'],
                title = title)

        data_array = np.column_stack([
                np.asarray(unpolarized['wavelength']),
                np.asarray(unpolarized['scattering']),
                np.asarray(unpolarized['extinction']),
                np.asarray(unpolarized['absorption'])])

        _write_txt(path, header, data_array)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    # Combined file
    columns = ['wavelength(nm)']
    cols_data = [wavelength]

    for ipol in range(n_pol):
        short = polarization_labels[ipol] if ipol < len(polarization_labels) \
                else 'pol{}'.format(ipol + 1)
        # sanitize short label for column header (no spaces)
        short_san = short.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_')
        columns.extend([
                'sca_{}(nm^2)'.format(short_san),
                'ext_{}(nm^2)'.format(short_san),
                'abs_{}(nm^2)'.format(short_san)])
        cols_data.extend([sca[:, ipol], ext[:, ipol], abs_[:, ipol]])

    if unpolarized is not None:
        columns.extend(['sca_unpol(nm^2)', 'ext_unpol(nm^2)', 'abs_unpol(nm^2)'])
        cols_data.extend([
                np.asarray(unpolarized['scattering']),
                np.asarray(unpolarized['extinction']),
                np.asarray(unpolarized['absorption'])])

    combined_path = os.path.join(out_dir, 'spectra_all.txt')
    combined_header = _format_spectrum_header(
            'Combined spectrum ({} polarizations{})'.format(
                    n_pol,
                    ', including unpolarized' if unpolarized is not None else ''),
            columns,
            title = title)

    combined_array = np.column_stack(cols_data)
    _write_txt(combined_path, combined_header, combined_array)
    saved_files.append(combined_path)
    print_info('saved <{}>'.format(combined_path))

    return saved_files


def _format_spectrum_header(description: str,
        columns: List[str],
        title: str = '') -> str:
    lines = []

    if title != '':
        lines.append('# {}'.format(title))

    lines.append('# {}'.format(description))
    lines.append('# Columns: ' + ' | '.join(columns))
    lines.append('# ' + ' '.join(['{:>16}'.format(c) for c in columns]))

    return '\n'.join(lines)


def _write_txt(path: str,
        header: str,
        data_array: np.ndarray,
        fmt: str = '%.6e') -> None:
    np.savetxt(path, data_array, header = header, comments = '', fmt = fmt)


def export_json(result: Dict[str, Any],
        output_path: str) -> str:

    ensure_dir(os.path.dirname(output_path) or '.')

    serializable = _to_serializable(result)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent = 2)

    print_info('export_json: <{}>'.format(output_path))
    return output_path


def _to_serializable(obj: Any) -> Any:

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {'__complex_array__': True,
                    'real': obj.real.tolist(),
                    'imag': obj.imag.tolist(),
                    'shape': list(obj.shape)}
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.complexfloating):
        return {'real': float(obj.real), 'imag': float(obj.imag)}

    if isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}

    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    return str(obj)
