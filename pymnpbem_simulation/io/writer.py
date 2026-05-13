import os

from typing import Any, Dict

import numpy as np

from ..util import ensure_dir, save_json, now_str, print_info


def save_spectrum(out_dir: str,
        result: Dict[str, Any]) -> Dict[str, str]:

    ensure_dir(out_dir)

    npz_path = os.path.join(out_dir, 'spectrum.npz')
    json_path = os.path.join(out_dir, 'spectrum.json')

    save_kwargs = {
            'wavelength': result['wavelength'],
            'ext': result['ext'],
            'sca': result['sca'],
            'abs': result['abs']}

    sc = result.get('surface_charge', None)

    if sc is not None:
        save_kwargs['surface_charge_wavelengths'] = np.asarray(sc['wavelengths'])
        save_kwargs['surface_charge_wl_indices'] = np.asarray(sc['wl_indices'])
        save_kwargs['surface_charge_sig2'] = np.asarray(sc['sig2'])
        save_kwargs['surface_charge_sig1'] = np.asarray(sc['sig1'])
        save_kwargs['surface_charge_verts'] = np.asarray(sc['verts'])
        save_kwargs['surface_charge_faces'] = np.asarray(sc['faces'])
        save_kwargs['surface_charge_centroids'] = np.asarray(sc['centroids'])
        save_kwargs['surface_charge_normals'] = np.asarray(sc['normals'])
        save_kwargs['surface_charge_areas'] = np.asarray(sc['areas'])
        save_kwargs['surface_charge_polarizations'] = np.asarray(sc['polarizations'])
        print_info('surface_charge: saving sigma at {} wavelengths'.format(
                len(sc['wavelengths'])))

    np.savez_compressed(npz_path, **save_kwargs)

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


def save_field(out_dir: str,
        result: Dict[str, Any]) -> Dict[str, str]:

    ensure_dir(out_dir)

    npz_path = os.path.join(out_dir, 'field.npz')
    json_path = os.path.join(out_dir, 'field.json')

    e = np.asarray(result['e'])
    h = result.get('h', None)
    pos = np.asarray(result['pos'])
    wl = np.asarray(result['wavelength'])

    save_kwargs = {
            'wavelength': wl,
            'pos': pos,
            'e_field': e,
            'grid_shape': np.asarray(result.get('grid_shape', e.shape[:2]))}

    if h is not None:
        save_kwargs['h_field'] = np.asarray(h)

    np.savez_compressed(npz_path, **save_kwargs)

    e_abs = np.abs(e)
    finite_mask = np.isfinite(e_abs)
    if finite_mask.any():
        e_max = float(np.nanmax(e_abs))
    else:
        e_max = float('nan')

    finite_frac = float(finite_mask.mean())

    summary = {
            'kind': 'field',
            'n_wavelengths': int(wl.size),
            'n_points': int(pos.shape[0]),
            'n_pol': int(result.get('n_pol', 1)),
            'inout': int(result.get('inout', 2)),
            'wall_s': float(result.get('wall_s', 0.0)),
            'wall_min': float(result.get('wall_s', 0.0)) / 60.0,
            'warmup_s': float(result.get('warmup_s', 0.0)),
            'wavelengths': wl.tolist(),
            'e_max_abs': e_max,
            'e_finite_fraction': finite_frac,
            'has_h_field': h is not None,
            'grid_shape': list(result.get('grid_shape', e.shape[:2]))}

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
