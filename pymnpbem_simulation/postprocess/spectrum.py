from typing import Any, Dict, List, Optional, Tuple

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

        # Multi-peak detection (returns sorted list of {wl_nm, ext, prominence})
        peaks = find_spectrum_peaks(wavelength, ext_i)

        out['per_pol'][str(i)] = {
            'peak_wl_nm': peak_wl,
            'peak_ext': peak_val,
            'peak_sca': float(sca_i[peak_idx]),
            'peak_abs': float(abs_i[peak_idx]),
            'fwhm_nm': fwhm,
            'peaks': peaks}

    # avg / max cross sections (across wavelength axis)
    out['avg_extinction'] = np.mean(ext, axis = 0).tolist()
    out['avg_scattering'] = np.mean(sca, axis = 0).tolist()
    out['avg_absorption'] = np.mean(abs_, axis = 0).tolist()
    out['max_extinction'] = np.max(ext, axis = 0).tolist()
    out['max_scattering'] = np.max(sca, axis = 0).tolist()
    out['max_absorption'] = np.max(abs_, axis = 0).tolist()

    # enhancement factors (only meaningful for n_pol >= 2)
    if n_pol >= 2:
        out['enhancement_factors'] = compute_enhancement_factors(ext)

    return out


def find_spectrum_peaks(wavelength: np.ndarray,
        spectrum: np.ndarray,
        prominence_ratio: float = 0.1,
        max_peaks: int = 5) -> List[Dict[str, float]]:
    """Find multiple peaks via scipy.signal.find_peaks.

    Returns sorted list (by amplitude desc) of dicts:
        {'wl_nm': float, 'value': float, 'prominence': float}
    """
    try:
        from scipy import signal
    except ImportError:
        # fallback to argmax only
        idx = int(np.argmax(spectrum))
        return [{
                'wl_nm': float(wavelength[idx]),
                'value': float(spectrum[idx]),
                'prominence': float('nan')}]

    sp = np.asarray(spectrum)
    if np.iscomplexobj(sp):
        sp = np.abs(sp)

    sp_max = float(np.max(sp)) if sp.size > 0 else 0.0
    if sp_max <= 0:
        return []

    peaks, props = signal.find_peaks(sp,
            prominence = prominence_ratio * sp_max)

    if len(peaks) == 0:
        idx = int(np.argmax(sp))
        return [{
                'wl_nm': float(wavelength[idx]),
                'value': float(sp[idx]),
                'prominence': float('nan')}]

    proms = props.get('prominences', np.zeros(len(peaks)))
    order = np.argsort(sp[peaks])[::-1][:max_peaks]

    out = []
    for k in order:
        idx = int(peaks[k])
        out.append({
                'wl_nm': float(wavelength[idx]),
                'value': float(sp[idx]),
                'prominence': float(proms[k])})
    return out


def compute_enhancement_factors(ext: np.ndarray) -> Dict[str, float]:
    """Compute pairwise enhancement ratios between polarizations.

    For 2 polarizations: returns {'pol0_vs_pol1': max_ext0 / max_ext1}.
    For 3+ polarizations: returns all pairwise ratios.
    """
    n_pol = ext.shape[1]
    max_vals = np.max(ext, axis = 0)

    out = dict()
    for i in range(n_pol):
        for j in range(i + 1, n_pol):
            denom = float(max_vals[j]) if max_vals[j] > 0 else 1e-30
            out['pol{}_vs_pol{}'.format(i, j)] = float(max_vals[i] / denom)
    return out


def check_unpolarized_conditions(polarizations: Optional[List[Any]],
        excitation_type: str,
        n_polarizations: int,
        tolerance: float = 1e-6) -> Dict[str, Any]:
    """Check whether FDTD-style incoherent unpolarized averaging is possible.

    Required:
        - planewave: exactly 2 orthogonal polarizations
        - dipole: exactly 3 mutually orthogonal directions
        - eels: not supported (returns can_calculate=False)
    """
    out = {
            'can_calculate': False,
            'method': None,
            'reason': None}

    if excitation_type == 'eels':
        out['reason'] = 'EELS does not support unpolarized averaging'
        return out

    if polarizations is None or len(polarizations) == 0:
        out['reason'] = 'No polarization data'
        return out

    pols = np.asarray(polarizations, dtype = float)
    if pols.ndim < 2 or pols.shape[1] < 3:
        out['reason'] = 'Polarizations must be Nx3 vectors'
        return out

    if excitation_type == 'planewave':
        if n_polarizations != 2:
            out['reason'] = 'planewave unpolarized requires 2 pols (got {})'.format(n_polarizations)
            return out

        if _are_orthogonal(pols[0], pols[1], tolerance = tolerance):
            out['can_calculate'] = True
            out['method'] = 'orthogonal_2pol_average'
            out['reason'] = 'Two orthogonal polarizations detected'
        else:
            out['reason'] = 'Polarizations are not orthogonal'
        return out

    if excitation_type == 'dipole':
        if n_polarizations != 3:
            out['reason'] = 'dipole unpolarized requires 3 directions (got {})'.format(n_polarizations)
            return out

        if (_are_orthogonal(pols[0], pols[1], tolerance = tolerance)
                and _are_orthogonal(pols[1], pols[2], tolerance = tolerance)
                and _are_orthogonal(pols[0], pols[2], tolerance = tolerance)):
            out['can_calculate'] = True
            out['method'] = 'orthogonal_3dir_average'
            out['reason'] = 'Three orthogonal directions detected'
        else:
            out['reason'] = 'Directions are not mutually orthogonal'
        return out

    out['reason'] = 'Unknown excitation_type <{}>'.format(excitation_type)
    return out


def calculate_unpolarized_spectrum(result: Dict[str, Any],
        unpol_info: Dict[str, Any]) -> Dict[str, Any]:
    """Compute incoherent average across polarizations.

    For 2 pols: σ_unpol = (σ_p0 + σ_p1) / 2
    For 3 pols: σ_unpol = (σ_p0 + σ_p1 + σ_p2) / 3
    """
    if not unpol_info.get('can_calculate', False):
        raise ValueError('[error] cannot calculate unpolarized: {}'.format(
                unpol_info.get('reason', 'unknown')))

    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])
    abs_ = np.asarray(result['abs'])

    n_pol = ext.shape[1]

    unpol_ext = np.mean(ext, axis = 1)
    unpol_sca = np.mean(sca, axis = 1)
    unpol_abs = np.mean(abs_, axis = 1)

    peak_idx = int(np.argmax(unpol_abs))

    return {
        'wavelength': wavelength,
        'extinction': unpol_ext,
        'scattering': unpol_sca,
        'absorption': unpol_abs,
        'peak_wavelength': float(wavelength[peak_idx]),
        'peak_idx': peak_idx,
        'peak_extinction': float(unpol_ext[peak_idx]),
        'peak_scattering': float(unpol_sca[peak_idx]),
        'peak_absorption': float(unpol_abs[peak_idx]),
        'method': unpol_info['method'],
        'n_averaged': int(n_pol),
        'fwhm_nm': _compute_fwhm(wavelength, unpol_ext)}


def analyze_spectrum_unpolarized(result: Dict[str, Any],
        polarizations: Optional[List[Any]] = None,
        excitation_type: str = 'planewave') -> Dict[str, Any]:
    """High-level wrapper: check + calculate unpolarized in one call.

    Returns dict with:
        - 'unpolarized': info dict from check_unpolarized_conditions
        - 'spectrum': unpolarized cross section dict (only if can_calculate=True)
    """
    n_pol = int(np.asarray(result['ext']).shape[1])

    info = check_unpolarized_conditions(polarizations, excitation_type, n_pol)
    out = {'unpolarized': info}

    if info.get('can_calculate', False):
        out['spectrum'] = calculate_unpolarized_spectrum(result, info)
    return out


def _are_orthogonal(v1: np.ndarray,
        v2: np.ndarray,
        tolerance: float = 1e-6) -> bool:

    v1 = np.asarray(v1, dtype = float)
    v2 = np.asarray(v2, dtype = float)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 < tolerance or n2 < tolerance:
        return False

    cos = float(np.abs(np.dot(v1 / n1, v2 / n2)))
    return cos < tolerance


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
