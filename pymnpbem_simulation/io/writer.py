import os

from typing import Any, Dict

import numpy as np

from ..util import ensure_dir, save_json, now_str, print_info


def save_spectrum(out_dir: str,
        result: Dict[str, Any]) -> Dict[str, str]:

    ensure_dir(out_dir)

    npz_path = os.path.join(out_dir, 'spectrum.npz')
    json_path = os.path.join(out_dir, 'spectrum.json')

    np.savez_compressed(
        npz_path,
        wavelength = result['wavelength'],
        ext = result['ext'],
        sca = result['sca'],
        abs = result['abs'])

    summary = {
        'n_wavelengths': int(result['wavelength'].size),
        'n_pol': int(result['n_pol']),
        'wall_s': float(result['wall_s']),
        'wall_min': float(result['wall_s']) / 60.0,
        'warmup_s': float(result['warmup_s']),
        'peak_idx': int(result['peak_idx']),
        'peak_wl_nm': float(result['peak_wl_nm']),
        'peak_ext_x': float(result['peak_ext_x']),
        'wavelengths': result['wavelength'].tolist(),
        'ext_x': result['ext'][:, 0].tolist(),
        'ext_y': result['ext'][:, 1].tolist() if result['ext'].shape[1] > 1 else None,
        'sca_x': result['sca'][:, 0].tolist(),
        'sca_y': result['sca'][:, 1].tolist() if result['sca'].shape[1] > 1 else None}

    save_json(json_path, summary)

    print_info('saved <{}>'.format(npz_path))
    print_info('saved <{}>'.format(json_path))

    return {'npz': npz_path, 'json': json_path}


def save_run_metadata(out_dir: str,
        cfg: Dict[str, Any],
        nfaces: int,
        timestamp: str = '') -> str:

    ensure_dir(out_dir)

    if timestamp == '':
        timestamp = now_str()

    meta = {
        'timestamp': timestamp,
        'nfaces': nfaces,
        'config': cfg}

    path = os.path.join(out_dir, 'run_metadata.json')
    save_json(path, meta)

    return path
