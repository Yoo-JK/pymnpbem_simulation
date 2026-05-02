import os

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from box import Box

from ..util import print_info


def qs_eigenmodes(p: Any,
        n_modes: int = 10,
        **options: Any) -> Box:

    from mnpbem.bem import BEMStatEig

    bem_eig = BEMStatEig(p, nev = n_modes, **options)

    n_actual = int(bem_eig.nev)

    eigenvalues = np.diag(bem_eig.ene).copy()
    eigenvectors_r = bem_eig.ur.copy()
    eigenvectors_l = bem_eig.ul.copy()

    inv_eps_p = _eigvals_to_inv_eps(eigenvalues)

    print_info('qs_eigenmodes: n_modes={}, n_actual={}, ene[0..2]={}'.format(
            n_modes, n_actual, eigenvalues[:min(3, n_actual)].real.tolist()))

    return Box({
        'eigenvalues': eigenvalues,
        'eigenvectors_r': eigenvectors_r,
        'eigenvectors_l': eigenvectors_l,
        'inv_eps_p': inv_eps_p,
        'n_modes': n_actual})


def svd_decomposition(sig_matrix: np.ndarray,
        rank_threshold: float = 1e-3) -> Box:

    sig_matrix = np.asarray(sig_matrix)

    if sig_matrix.ndim != 2:
        raise ValueError('[error] sig_matrix must be 2D, got shape <{}>'.format(
                sig_matrix.shape))

    u, s, vh = np.linalg.svd(sig_matrix, full_matrices = False)

    s_max = float(s[0]) if len(s) > 0 else 0.0
    if s_max > 0:
        rank_eff = int(np.sum(s > rank_threshold * s_max))
    else:
        rank_eff = 0

    energy = s ** 2
    energy_cum = np.cumsum(energy) / max(np.sum(energy), 1e-30)

    print_info('svd_decomposition: shape={}, rank_eff={}, s[0..2]={}'.format(
            sig_matrix.shape, rank_eff,
            s[:min(3, len(s))].tolist()))

    return Box({
        'u': u,
        'singular_values': s,
        'vh': vh,
        'rank_eff': rank_eff,
        'energy_cumulative': energy_cum})


def retarded_eigenmodes(p: Any,
        enei: float,
        n_modes: int = 10,
        epstab: Optional[List[Any]] = None,
        **options: Any) -> Box:

    from mnpbem.bem import BEMStatEig

    bem_eig = BEMStatEig(p, nev = n_modes, enei = enei, **options)

    n_actual = int(bem_eig.nev)

    eigenvalues = np.diag(bem_eig.ene).copy()
    eigenvectors_r = bem_eig.ur.copy()
    eigenvectors_l = bem_eig.ul.copy()

    resolvent = bem_eig.mat.copy() if bem_eig.mat is not None else None

    print_info('retarded_eigenmodes: enei={:.2f}, n_modes={}, n_actual={}'.format(
            enei, n_modes, n_actual))

    return Box({
        'enei': float(enei),
        'eigenvalues': eigenvalues,
        'eigenvectors_r': eigenvectors_r,
        'eigenvectors_l': eigenvectors_l,
        'resolvent': resolvent,
        'n_modes': n_actual})


def _eigvals_to_inv_eps(eigenvalues: np.ndarray) -> np.ndarray:

    return eigenvalues / (2.0 * np.pi)
