from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from box import Box

from ..util import print_info


def fano_lineshape(x: np.ndarray,
        amp: float,
        x0: float,
        gamma: float,
        q: float,
        c: float) -> np.ndarray:

    x = np.asarray(x)
    half_g = gamma / 2.0
    eps = (x - x0) / (half_g if half_g != 0 else 1e-12)
    return amp * (q + eps) ** 2 / (1.0 + eps ** 2) + c


def fano_fit(enei: np.ndarray,
        spectrum: np.ndarray,
        initial: Optional[Dict[str, float]] = None) -> Box:

    try:
        from lmfit import Model
    except ImportError:
        return _fano_fit_scipy(enei, spectrum, initial)

    enei = np.asarray(enei, dtype = float)
    spectrum = np.asarray(spectrum, dtype = float)

    model = Model(fano_lineshape)
    p0 = _fano_initial_guess(enei, spectrum)
    if initial is not None:
        p0.update(initial)

    params = model.make_params(**p0)

    result = model.fit(spectrum, params, x = enei)

    fit_curve = result.best_fit
    best = dict(result.best_values)

    print_info('fano_fit: x0={:.2f}, gamma={:.2f}, q={:.3f}, redchi={:.3e}'.format(
            best.get('x0', float('nan')),
            best.get('gamma', float('nan')),
            best.get('q', float('nan')),
            float(result.redchi)))

    return Box({
        'best_fit': best,
        'chisqr': float(result.chisqr),
        'redchi': float(result.redchi),
        'fit_curve': np.asarray(fit_curve),
        'success': bool(result.success)})


def multi_fano_fit(enei: np.ndarray,
        spectrum: np.ndarray,
        n_peaks: int = 2,
        initial_list: Optional[List[Dict[str, float]]] = None) -> Box:

    try:
        from lmfit import Model
    except ImportError:
        return _multi_fano_fit_scipy(enei, spectrum, n_peaks, initial_list)

    enei = np.asarray(enei, dtype = float)
    spectrum = np.asarray(spectrum, dtype = float)

    composite = None
    params_total = None

    for k in range(n_peaks):
        prefix = 'p{}_'.format(k)
        m = Model(fano_lineshape, prefix = prefix)
        composite = m if composite is None else (composite + m)

    p0_list = _multi_fano_initial_guess(enei, spectrum, n_peaks)
    if initial_list is not None:
        for k in range(min(n_peaks, len(initial_list))):
            p0_list[k].update(initial_list[k])

    params_total = composite.make_params()
    for k in range(n_peaks):
        prefix = 'p{}_'.format(k)
        for key, val in p0_list[k].items():
            params_total[prefix + key].set(value = val)

    result = composite.fit(spectrum, params_total, x = enei)

    best = dict(result.best_values)

    peaks = []
    for k in range(n_peaks):
        prefix = 'p{}_'.format(k)
        peaks.append({
            'amp': best.get(prefix + 'amp', float('nan')),
            'x0': best.get(prefix + 'x0', float('nan')),
            'gamma': best.get(prefix + 'gamma', float('nan')),
            'q': best.get(prefix + 'q', float('nan')),
            'c': best.get(prefix + 'c', float('nan'))})

    print_info('multi_fano_fit: n_peaks={}, redchi={:.3e}'.format(
            n_peaks, float(result.redchi)))

    return Box({
        'peaks': peaks,
        'best_fit': best,
        'chisqr': float(result.chisqr),
        'redchi': float(result.redchi),
        'fit_curve': np.asarray(result.best_fit),
        'success': bool(result.success)})


def q_to_delta_phi(q: float) -> float:

    # Relation between the Fano asymmetry parameter q and the inter-mode phase
    # difference: q = cot(delta/2)  =>  Delta_phi = 2 * arctan(1 / q).
    # Limits: q -> inf gives Delta_phi -> 0 (symmetric Lorentzian),
    #         q  -> 0  gives Delta_phi -> pi (anti-resonance / dark mode).
    # Ported from mnpbem_simulation/.../fano_fitter.py:q_to_delta_phi (line 165).
    eps_safe = 1e-15
    if np.isinf(q):
        return 0.0
    if np.abs(q) < eps_safe:
        return float(np.pi)
    return float(2.0 * np.arctan(1.0 / q))


def validate_consistency(q_fit: float,
        delta_phi_measured: np.ndarray,
        wavelength_at_center: float,
        wavelengths: np.ndarray,
        consistency_threshold: float = np.pi / 4) -> Box:

    # Cross-check the Fano-fit asymmetry q against the directly measured
    # inter-mode phase difference Delta-phi at the resonance center.
    # Ported from mnpbem_simulation/.../fano_fitter.py:validate_consistency
    # (lines 175-192).
    wavelengths = np.asarray(wavelengths, dtype = float)
    delta_phi_measured = np.asarray(delta_phi_measured, dtype = float)

    if wavelengths.shape != delta_phi_measured.shape:
        raise ValueError(
                '[error] <wavelengths> and <delta_phi_measured> shape mismatch')

    idx = int(np.argmin(np.abs(wavelengths - wavelength_at_center)))
    measured_here = float(delta_phi_measured[idx])

    delta_phi_theory = q_to_delta_phi(q_fit)

    deviation = float(np.abs(np.angle(np.exp(1j * (measured_here - delta_phi_theory)))))
    is_consistent = bool(deviation < consistency_threshold)

    print_info('validate_consistency: q={:.3f}, dphi_theory={:.3f}, dphi_measured={:.3f}, dev={:.3f} rad, ok={}'.format(
            q_fit, delta_phi_theory, measured_here, deviation, is_consistent))

    return Box({
        'q_value': float(q_fit),
        'delta_phi_from_q': delta_phi_theory,
        'delta_phi_measured': measured_here,
        'deviation_rad': deviation,
        'is_consistent': is_consistent,
        'wavelength_at_center': float(wavelength_at_center),
        'center_index': idx})


def _fano_initial_guess(enei: np.ndarray,
        spectrum: np.ndarray) -> Dict[str, float]:

    peak_idx = int(np.argmax(spectrum))
    x0 = float(enei[peak_idx])
    bg = float(np.median(spectrum))
    amp = float(spectrum[peak_idx] - bg)
    if amp <= 0:
        amp = float(max(spectrum) - min(spectrum))
    span = float(enei[-1] - enei[0])
    gamma = max(span / 20.0, 1e-6)

    return {
        'amp': amp,
        'x0': x0,
        'gamma': gamma,
        'q': 1.0,
        'c': bg}


def _multi_fano_initial_guess(enei: np.ndarray,
        spectrum: np.ndarray,
        n_peaks: int) -> List[Dict[str, float]]:

    span = float(enei[-1] - enei[0])
    bg = float(np.median(spectrum))
    amp = float(max(spectrum) - bg)
    gamma = max(span / 20.0, 1e-6)

    out = []
    for k in range(n_peaks):
        frac = (k + 1.0) / (n_peaks + 1.0)
        out.append({
            'amp': amp / max(n_peaks, 1),
            'x0': float(enei[0] + frac * span),
            'gamma': gamma,
            'q': 1.0,
            'c': bg if k == 0 else 0.0})
    return out


def _fano_fit_scipy(enei: np.ndarray,
        spectrum: np.ndarray,
        initial: Optional[Dict[str, float]]) -> Box:

    from scipy.optimize import curve_fit

    enei = np.asarray(enei, dtype = float)
    spectrum = np.asarray(spectrum, dtype = float)

    p0 = _fano_initial_guess(enei, spectrum)
    if initial is not None:
        p0.update(initial)

    p_init = [p0['amp'], p0['x0'], p0['gamma'], p0['q'], p0['c']]

    try:
        popt, _ = curve_fit(fano_lineshape, enei, spectrum,
                p0 = p_init, maxfev = 10000)
        success = True
    except Exception as e:
        popt = p_init
        success = False
        print_info('fano_fit_scipy fallback failed: {}'.format(e))

    fit_curve = fano_lineshape(enei, *popt)
    residual = spectrum - fit_curve
    chisqr = float(np.sum(residual ** 2))
    dof = max(len(enei) - 5, 1)
    redchi = chisqr / dof

    return Box({
        'best_fit': {
            'amp': float(popt[0]),
            'x0': float(popt[1]),
            'gamma': float(popt[2]),
            'q': float(popt[3]),
            'c': float(popt[4])},
        'chisqr': chisqr,
        'redchi': redchi,
        'fit_curve': fit_curve,
        'success': success})


def _multi_fano_fit_scipy(enei: np.ndarray,
        spectrum: np.ndarray,
        n_peaks: int,
        initial_list: Optional[List[Dict[str, float]]]) -> Box:

    from scipy.optimize import curve_fit

    enei = np.asarray(enei, dtype = float)
    spectrum = np.asarray(spectrum, dtype = float)

    def multi_fano(x, *params):
        out = np.zeros_like(x)
        for k in range(n_peaks):
            offset = k * 5
            out = out + fano_lineshape(x, *params[offset:offset + 5])
        return out

    p0_list = _multi_fano_initial_guess(enei, spectrum, n_peaks)
    if initial_list is not None:
        for k in range(min(n_peaks, len(initial_list))):
            p0_list[k].update(initial_list[k])

    p_init = []
    for d in p0_list:
        p_init.extend([d['amp'], d['x0'], d['gamma'], d['q'], d['c']])

    try:
        popt, _ = curve_fit(multi_fano, enei, spectrum,
                p0 = p_init, maxfev = 20000)
        success = True
    except Exception:
        popt = p_init
        success = False

    fit_curve = multi_fano(enei, *popt)
    residual = spectrum - fit_curve
    chisqr = float(np.sum(residual ** 2))
    dof = max(len(enei) - n_peaks * 5, 1)
    redchi = chisqr / dof

    peaks = []
    for k in range(n_peaks):
        offset = k * 5
        peaks.append({
            'amp': float(popt[offset]),
            'x0': float(popt[offset + 1]),
            'gamma': float(popt[offset + 2]),
            'q': float(popt[offset + 3]),
            'c': float(popt[offset + 4])})

    return Box({
        'peaks': peaks,
        'best_fit': {'p{}_{}'.format(k, key): peaks[k][key]
                for k in range(n_peaks)
                for key in ('amp', 'x0', 'gamma', 'q', 'c')},
        'chisqr': chisqr,
        'redchi': redchi,
        'fit_curve': fit_curve,
        'success': success})
