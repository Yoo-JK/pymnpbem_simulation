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

    # NOTE: despite the name, this routine is actually QUASI-STATIC. It wraps
    # BEMStatEig, whose eigenvectors are the eigenmodes of the static surface-
    # derivative operator F and are therefore enei-INDEPENDENT (only the
    # eigenvalue-to-eps mapping uses enei). For a GENUINELY retarded
    # eigendecomposition whose eigenvectors depend on enei via the full
    # BEMRet operator (G/H/Delta/Sigma at finite k), use
    # `retarded_eigen_full` below.
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


def project(ul: np.ndarray,
        sigma_matrix: np.ndarray) -> Box:

    # Bi-orthogonal modal projection of driven surface charge onto left
    # eigenvectors: g = ul @ S, where ul is (K, N) and sigma_matrix is the
    # driven surface charge stacked as (N, n_lambda). Returns the complex modal
    # amplitudes g (K, n_lambda) together with their magnitudes and phases
    # (phases feed the inter-mode Delta-phi pipeline).
    # Ported from mnpbem_simulation/.../eigenmode_analyzer.py:project (g = ul @ sigma).
    ul = np.asarray(ul)
    sigma_matrix = np.asarray(sigma_matrix)

    if ul.ndim != 2:
        raise ValueError('[error] <ul> must be 2D (K, N), got shape {}'.format(ul.shape))

    if sigma_matrix.ndim == 1:
        sigma_matrix = sigma_matrix[:, None]
    if sigma_matrix.ndim != 2:
        raise ValueError('[error] <sigma_matrix> must be 1D/2D, got shape {}'.format(
                sigma_matrix.shape))

    if ul.shape[1] != sigma_matrix.shape[0]:
        raise ValueError('[error] <ul> cols ({}) != <sigma_matrix> rows ({})'.format(
                ul.shape[1], sigma_matrix.shape[0]))

    g_complex = ul @ sigma_matrix.astype(complex)

    print_info('project: ul={}, sigma={}, g={}'.format(
            ul.shape, sigma_matrix.shape, g_complex.shape))

    return Box({
        'g_complex': g_complex,
        'magnitudes': np.abs(g_complex),
        'phases': np.angle(g_complex)})


def retarded_eigen_full(p: Any,
        enei: float,
        n_modes: int = 10,
        sort_mode: str = 'abs_real_asc',
        self_test: bool = True,
        self_test_rtol: float = 1e-8,
        **options: Any) -> Box:

    # Faithful Schur-biorthogonal RETARDED eigendecomposition at a fixed
    # wavelength enei. Unlike `retarded_eigenmodes` (quasi-static BEMStatEig),
    # the operator and hence the eigenvectors here depend genuinely on enei
    # through the finite-k BEMRet building blocks.
    #
    # Ported from mnpbem_simulation/.../retarded_eigen.py (RetardedEigenAnalyzer:
    # assemble_matrix + _eig_biorthogonal). The effective N x N operator on the
    # outer surface charge sig2 is reconstructed from BEMRet's stored internals
    # after `bem.init(enei)`:
    #
    #     Delta   = Sigma1 - Sigma2                 (Delta_lu = LU of Delta)
    #     Deltai  = inv(Delta) = lu_solve(Delta_lu, I)
    #     Sigma2  = Sigma1 - Delta
    #     L       = L1 - L2
    #     Sigma_eff = Sigma1*L1 - Sigma2*L2
    #                 + k^2 * (L*Deltai) * (nvec nvec^T) * L
    #
    # which matches BEMRet's own factored reduced operator (bem.Sigma_lu) to
    # machine precision (verified by the self-test below). The retarded plasmon
    # eigenmodes are then the (right) eigenvectors of Sigma_eff; the left
    # eigenvectors ul (bi-orthogonalised so ul @ ur = I) provide the projection
    # basis for `project()`.
    #
    # COST: this builds dense N x N inverses and a dense eigendecomposition, so
    # it is O(N^3) in time and O(N^2) in memory. It is intended for SMALL meshes
    # (validation / analysis, <~6000 faces); it is impractical for large
    # production meshes.
    import scipy.linalg

    from mnpbem.bem import BEMRet
    from mnpbem.utils.gpu import lu_solve_dispatch

    bem = BEMRet(p, **options)
    bem.init(enei)

    nfaces = int(p.nfaces)
    k = bem.k
    nvec = np.asarray(bem.nvec)

    eye = np.eye(nfaces, dtype = complex)

    # Reconstruct dense inverses from the stored CPU LU factorizations. We use
    # the pymnpbem LU dispatcher (returns host arrays) rather than scipy's
    # lu_solve directly, because pymnpbem stores LU as a ('cpu', lu, piv) tuple.
    deltai = lu_solve_dispatch(bem.Delta_lu, eye)

    sigma1 = np.asarray(bem.Sigma1)
    delta = np.linalg.inv(deltai)
    sigma2 = sigma1 - delta

    # L1/L2 may be scalars (uniform eps) or full (N, N) matrices.
    l1 = bem.L1
    l2 = bem.L2
    l_diff = l1 - l2
    nvec_outer = nvec @ nvec.T

    if np.isscalar(l_diff):
        sigma_eff = (sigma1 * l1 - sigma2 * l2 +
                     (k ** 2) * l_diff * (deltai * nvec_outer) * l_diff)
    else:
        l1_mat = np.diag(l1) if np.ndim(l1) == 1 else np.asarray(l1)
        l2_mat = np.diag(l2) if np.ndim(l2) == 1 else np.asarray(l2)
        l_diff_mat = l1_mat - l2_mat
        sigma_eff = (sigma1 @ l1_mat - sigma2 @ l2_mat +
                     (k ** 2) * ((l_diff_mat @ deltai) * nvec_outer) @ l_diff_mat)

    # Self-test: the reconstructed effective operator must match BEMRet's own
    # stored reduced operator (factored into bem.Sigma_lu) to machine precision.
    if self_test:
        sigma_stored = np.linalg.inv(lu_solve_dispatch(bem.Sigma_lu, eye))
        denom = np.linalg.norm(sigma_stored)
        rel = (np.linalg.norm(sigma_eff - sigma_stored) / denom) if denom > 0 else 0.0
        if not (rel <= self_test_rtol):
            raise AssertionError(
                    '[error] retarded_eigen_full self-test failed: Sigma_eff vs '
                    'bem.Sigma_lu rel diff {:.3e} > rtol {:.1e}'.format(rel, self_test_rtol))
        print_info('retarded_eigen_full self-test: Sigma_eff vs bem.Sigma_lu rel={:.3e} (OK)'.format(rel))

    # Dense bi-orthogonal eigendecomposition of the (generally non-Hermitian)
    # effective operator.
    eigenvalues, ur, ul = _eig_biorthogonal(sigma_eff)
    eigenvalues, ur, ul = _sort_retarded_modes(eigenvalues, ur, ul, n_modes, sort_mode)

    n_actual = int(eigenvalues.size)

    print_info('retarded_eigen_full: enei={:.2f}, n_modes={}, n_actual={}, nfaces={}'.format(
            enei, n_modes, n_actual, nfaces))

    return Box({
        'enei': float(enei),
        'eigenvalues': eigenvalues,
        'eigenvectors_r': ur,
        'eigenvectors_l': ul,
        'sigma_eff': sigma_eff,
        'k': complex(k),
        'n_modes': n_actual,
        'nfaces': nfaces})


def _eig_biorthogonal(mat: np.ndarray):

    # Right/left eigendecomposition with bi-orthonormalization (ul @ ur = I).
    # Ported from retarded_eigen.py:_eig_biorthogonal.
    import scipy.linalg

    eigvals_r, ur = scipy.linalg.eig(mat)
    eigvals_l, ul_cols = scipy.linalg.eig(mat.T)

    # Pair left eigenvectors to right ones by nearest-eigenvalue assignment.
    paired_ul = np.zeros_like(ul_cols)
    used = np.zeros(len(eigvals_l), dtype = bool)
    for idx_r, lam in enumerate(eigvals_r):
        diffs = np.abs(eigvals_l - lam)
        diffs[used] = np.inf
        idx_l = int(np.argmin(diffs))
        paired_ul[:, idx_r] = ul_cols[:, idx_l]
        used[idx_l] = True

    ul = paired_ul.T

    overlap = ul @ ur
    cond = np.linalg.cond(overlap)
    if cond > 1e12:
        print_info('_eig_biorthogonal: overlap ill-conditioned (cond={:.2e}); using pinv'.format(cond))
        ul = np.linalg.pinv(overlap) @ ul
    else:
        ul = np.linalg.solve(overlap, ul)

    return eigvals_r, ur, ul


def _sort_retarded_modes(ene: np.ndarray,
        ur: np.ndarray,
        ul: np.ndarray,
        n_modes: int,
        sort_mode: str):

    if sort_mode == 'real_asc':
        sort_idx = np.argsort(ene.real)
    elif sort_mode == 'abs_real_asc':
        sort_idx = np.argsort(np.abs(ene.real))
    elif sort_mode == 'abs_asc':
        sort_idx = np.argsort(np.abs(ene))
    else:
        raise ValueError('[error] invalid <sort_mode>: {}'.format(sort_mode))

    ene = ene[sort_idx]
    ur = ur[:, sort_idx]
    ul = ul[sort_idx, :]

    k = min(n_modes, len(ene))
    return ene[:k], ur[:, :k], ul[:k, :]
