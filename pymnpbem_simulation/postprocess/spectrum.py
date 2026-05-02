from typing import Any, Dict

import numpy as np


def analyze_spectrum(result: Dict[str, Any]) -> Dict[str, Any]:
    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])
    abs_ = np.asarray(result['abs'])

    n_pol = ext.shape[1]
    out = {
        'n_wavelengths': int(len(wavelength)),
        'n_pol': int(n_pol),
        'per_pol': dict()}

    for i in range(n_pol):
        ext_i = ext[:, i]
        sca_i = sca[:, i]
        abs_i = abs_[:, i]

        peak_idx = int(np.argmax(ext_i))
        peak_wl = float(wavelength[peak_idx])
        peak_val = float(ext_i[peak_idx])

        fwhm = _compute_fwhm(wavelength, ext_i)

        out['per_pol'][str(i)] = {
            'peak_wl_nm': peak_wl,
            'peak_ext': peak_val,
            'peak_sca': float(sca_i[peak_idx]),
            'peak_abs': float(abs_i[peak_idx]),
            'fwhm_nm': fwhm}

    return out


def _compute_fwhm(wavelength: np.ndarray,
        spectrum: np.ndarray) -> float:

    peak_idx = int(np.argmax(spectrum))
    half_max = spectrum[peak_idx] / 2.0

    left = peak_idx
    while left > 0 and spectrum[left] > half_max:
        left -= 1

    right = peak_idx
    while right < len(spectrum) - 1 and spectrum[right] > half_max:
        right += 1

    if left == 0 or right == len(spectrum) - 1:
        return float('nan')

    return float(wavelength[right] - wavelength[left])
