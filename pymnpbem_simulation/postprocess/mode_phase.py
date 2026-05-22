from typing import Any, Optional

import numpy as np
from box import Box

from ..util import print_info
from .mode_compare import assign_bright_dark_multipole


# Inter-mode phase-difference (Delta-phi) workflow.
#
# Formalizes the delta_phi prototype (scratch/paper_figures/delta_phi_auag.py):
#   1. eigenmodes of the particle (quasi-static plasmonmode, or genuinely
#      retarded via retarded_eigen_full).
#   2. modal projection of the driven surface charge g = ul @ S.
#   3. multipole-based bright/dark assignment of the eigenmodes
#      (assign_bright_dark_multipole): bright = strongest dipolar mode,
#      dark = best-coupled genuinely higher-order (dominant l >= 2) mode.
#   4. inter-mode phase difference Delta-phi = arg(exp(1j*(phi_bright -
#      phi_dark))), WRAPPED to (-pi, pi] (NOT unwrapped).
#
# This is the analysis half of the Fano modal-decomposition cross-check
# (cf. mode_compare / fano_fit). It depends on MNPBEM only through
# plasmonmode / BEMRet (read-only) and is otherwise pure numpy.


def mode_phase_analysis(p: Any,
        sigma_matrix: np.ndarray,
        wavelengths: np.ndarray,
        method: str = 'qs',
        nev: int = 14,
        enei: Optional[float] = None,
        max_l: int = 4,
        **eig_options: Any) -> Box:
    """Inter-mode phase-difference (Delta-phi) analysis from a surface-charge sweep.

    Args:
        p: ComParticle (mnpbem). Provides the eigenmodes and the multipole
            geometry (p.pos / p.area).
        sigma_matrix: driven surface charge stacked as (nfaces, nlambda),
            sorted by wavelength to match <wavelengths>.
        wavelengths: (nlambda,) wavelengths in nm, ascending.
        method: 'qs'   -> quasi-static eigenmodes via mnpbem.bem.plasmonmode
                          (cheap, enei-independent eigenvectors).
                'retarded' -> genuinely retarded eigenmodes via
                          retarded_eigen_full at <enei>. O(N^3) time / O(N^2)
                          memory; impractical above ~6000 faces.
        nev: number of eigenmodes to request.
        enei: wavelength (nm) for method='retarded'. Required for 'retarded';
            ignored for 'qs'.
        max_l: maximum multipole order for the bright/dark assignment.
        **eig_options: forwarded to the eigensolver (plasmonmode /
            retarded_eigen_full).

    Returns:
        Box with:
            g           (K, nlambda) complex modal amplitudes g = ul @ S.
            magnitudes  (K, nlambda) |g|.
            phases      (K, nlambda) arg(g).
            bright      int, bright-mode index.
            dark        int, dark-mode index.
            dominant_l  (K,) int, dominant multipole order per mode.
            delta_phi   (nlambda,) WRAPPED inter-mode phase difference in
                        (-pi, pi]: arg(exp(1j*(phi_bright - phi_dark))).
            wavelengths (nlambda,) the input wavelengths (echoed).
            method, nev, enei: provenance.
            bright_l, dark_l, bright_resonance_nm, dark_resonance_nm: summary.
            coupling_strength_ratio: peak|g_bright| / peak|g_dark|.
    """
    sigma_matrix = np.asarray(sigma_matrix)
    wavelengths = np.asarray(wavelengths, dtype = float)

    if sigma_matrix.ndim != 2:
        raise ValueError('[error] <sigma_matrix> must be 2D (nfaces, nlambda)')

    nfaces = int(np.asarray(p.pos).shape[0])
    if sigma_matrix.shape[0] != nfaces:
        raise ValueError('[error] <sigma_matrix> rows ({}) != p.nfaces ({})'.format(
                sigma_matrix.shape[0], nfaces))
    if sigma_matrix.shape[1] != wavelengths.shape[0]:
        raise ValueError('[error] <sigma_matrix> cols ({}) != len(wavelengths) ({})'.format(
                sigma_matrix.shape[1], wavelengths.shape[0]))

    eigenvectors_r, eigenvectors_l = _compute_eigenmodes(
            p, method, nev, enei, **eig_options)

    # eigenvectors_l: (K, N) project basis; eigenvectors_r: (N, K) modes.
    g_complex = eigenvectors_l @ sigma_matrix.astype(complex)
    magnitudes = np.abs(g_complex)
    phases = np.angle(g_complex)

    assign = assign_bright_dark_multipole(
            eigenvectors_r, p, magnitudes, max_l = max_l)
    bright = int(assign.bright_idx)
    dark = int(assign.dark_idx)
    dominant_l = np.asarray(assign.dominant_l)

    # WRAPPED inter-mode phase difference (do NOT unwrap).
    delta_phi = np.angle(np.exp(1j * (phases[bright] - phases[dark])))

    bright_res = float(wavelengths[int(np.argmax(magnitudes[bright]))])
    dark_res = float(wavelengths[int(np.argmax(magnitudes[dark]))])

    print_info('mode_phase_analysis[{}]: bright=mode {} (l={}, res {:.0f}nm), dark=mode {} (l={}, res {:.0f}nm)'.format(
            method, bright, int(dominant_l[bright]), bright_res,
            dark, int(dominant_l[dark]), dark_res))

    return Box({
        'g': g_complex,
        'magnitudes': magnitudes,
        'phases': phases,
        'bright': bright,
        'dark': dark,
        'dominant_l': dominant_l,
        'delta_phi': delta_phi,
        'wavelengths': wavelengths,
        'method': method,
        'nev': int(nev),
        'enei': None if enei is None else float(enei),
        'bright_l': int(dominant_l[bright]),
        'dark_l': int(dominant_l[dark]),
        'bright_resonance_nm': bright_res,
        'dark_resonance_nm': dark_res,
        'coupling_strength_ratio': float(assign.coupling_strength_ratio)})


def _compute_eigenmodes(p: Any,
        method: str,
        nev: int,
        enei: Optional[float],
        **eig_options: Any):

    # Returns (eigenvectors_r (N, K), eigenvectors_l (K, N)).
    if method == 'qs':
        from mnpbem.bem.plasmonmode import plasmonmode

        _ene, ur, ul = plasmonmode(p, nev = nev, **eig_options)
        ur = np.asarray(ur)
        ul = np.asarray(ul)

        nfaces = int(np.asarray(p.pos).shape[0])
        # Normalize to ur (N, K) and ul (K, N).
        if ur.shape[0] != nfaces and ur.shape[1] == nfaces:
            ur = ur.T
        if ul.shape[1] != nfaces and ul.shape[0] == nfaces:
            ul = ul.T
        return ur, ul

    if method == 'retarded':
        if enei is None:
            raise ValueError('[error] method="retarded" requires <enei> (nm)')
        from .eigenmode import retarded_eigen_full

        res = retarded_eigen_full(p, enei = enei, n_modes = nev, **eig_options)
        ur = np.asarray(res.eigenvectors_r)
        ul = np.asarray(res.eigenvectors_l)
        return ur, ul

    raise ValueError('[error] invalid <method>: {} (use "qs" or "retarded")'.format(method))
