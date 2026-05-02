import os
import json

from typing import Any, Dict

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
