from typing import Any, Dict, Optional, Tuple

import numpy as np
from box import Box

from ..util import print_info


def multipole_decomposition(sig: np.ndarray,
        p: Any,
        max_l: int = 4,
        center: Optional[np.ndarray] = None) -> Box:

    sig = np.asarray(sig).reshape(-1)

    pos = np.asarray(p.pos)
    area = np.asarray(p.area).reshape(-1)

    if center is None:
        # area-weighted centroid
        center_arr = (pos * area[:, None]).sum(axis = 0) / area.sum()
    else:
        center_arr = np.asarray(center)

    rel = pos - center_arr[None, :]
    r = np.linalg.norm(rel, axis = 1)
    safe_r = np.where(r > 1e-12, r, 1e-12)

    cos_theta = np.clip(rel[:, 2] / safe_r, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = np.arctan2(rel[:, 1], rel[:, 0])

    # Build (l, m) list
    n_lm = sum(2 * l + 1 for l in range(max_l + 1))

    l_arr = np.empty(n_lm, dtype = int)
    m_arr = np.empty(n_lm, dtype = int)
    amp_arr = np.empty(n_lm, dtype = complex)

    weighted_sig = sig * area

    idx = 0
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            ylm = _real_sph_harm(l, m, theta, phi)
            coeff = np.sum(ylm * weighted_sig)
            l_arr[idx] = l
            m_arr[idx] = m
            amp_arr[idx] = coeff
            idx += 1

    # power per l-shell
    power_l = np.zeros(max_l + 1, dtype = float)
    for k in range(n_lm):
        power_l[l_arr[k]] += float(np.abs(amp_arr[k]) ** 2)

    print_info('multipole_decomposition: max_l={}, n_lm={}, power_l={}'.format(
            max_l, n_lm,
            ['{:.3e}'.format(v) for v in power_l]))

    return Box({
        'l': l_arr,
        'm': m_arr,
        'amplitudes': amp_arr,
        'power_l': power_l,
        'max_l': int(max_l),
        'center': center_arr})


def dipole_quadrupole_ratio(multipole: Box) -> float:

    power_l = np.asarray(multipole['power_l'])
    if len(power_l) < 3:
        return float('nan')

    p_dip = float(power_l[1])
    p_quad = float(power_l[2])

    if p_quad <= 0:
        return float('inf')

    return p_dip / p_quad


def dominant_l(multipole: Box) -> int:

    power_l = np.asarray(multipole['power_l'])
    # Skip l=0 (monopole — should be ~0 for neutral surface charge)
    if len(power_l) <= 1:
        return -1

    return int(np.argmax(power_l[1:]) + 1)


def _real_sph_harm(l: int,
        m: int,
        theta: np.ndarray,
        phi: np.ndarray) -> np.ndarray:

    from scipy.special import sph_harm

    # scipy.special.sph_harm expects (m, l, phi, theta) - returns complex Y_lm.
    # Convert to real spherical harmonics for stable, real expansion coefficients.
    if m > 0:
        ylm_p = sph_harm(m, l, phi, theta)
        ylm_n = sph_harm(-m, l, phi, theta)
        return float(np.sqrt(2.0)) * ((-1) ** m) * np.real(ylm_p)

    if m < 0:
        ylm_p = sph_harm(-m, l, phi, theta)
        return float(np.sqrt(2.0)) * ((-1) ** m) * np.imag(ylm_p)

    return np.real(sph_harm(0, l, phi, theta))
