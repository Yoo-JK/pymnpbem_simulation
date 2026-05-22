from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from box import Box

from ..util import print_info


# Literature:
#   Luk'yanchuk et al., Nat. Mater. 9, 707 (2010).
#   Hohenester & Trugler, Comput. Phys. Commun. 183, 370 (2012). (MNPBEM)
#   Stout et al., Phys. Rev. B 84, 195403 (2011). (SVD modal decomposition)
#
# Ported from mnpbem_simulation/postprocess/post_utils/mode_comparator.py.
# This module implements the inter-mode phase-difference (Delta-phi) pipeline
# used to cross-validate Fano modal decompositions (quasi-static vs retarded
# vs SVD). It is pure numpy and depends on no MNPBEM internals.


def mode_similarity_matrix(modes_a: np.ndarray,
        modes_b: np.ndarray) -> np.ndarray:

    modes_a = np.asarray(modes_a)
    modes_b = np.asarray(modes_b)

    if modes_a.ndim != 2 or modes_b.ndim != 2:
        raise ValueError('[error] <modes_a>, <modes_b> must be 2D (nfaces, K)')
    if modes_a.shape[0] != modes_b.shape[0]:
        raise ValueError('[error] inconsistent nfaces between two mode sets')

    norm_a = np.linalg.norm(modes_a, axis = 0)
    norm_b = np.linalg.norm(modes_b, axis = 0)

    eps = 1e-15
    norm_a_safe = np.where(norm_a > eps, norm_a, eps)
    norm_b_safe = np.where(norm_b > eps, norm_b, eps)

    a_normed = modes_a / norm_a_safe[np.newaxis, :]
    b_normed = modes_b / norm_b_safe[np.newaxis, :]

    similarity = np.abs(np.conjugate(a_normed).T @ b_normed)

    print_info('mode_similarity_matrix: shape={}, max={:.4f}'.format(
            similarity.shape, float(np.max(similarity)) if similarity.size else 0.0))

    return similarity


def match_modes(similarity_matrix: np.ndarray,
        threshold: float = 0.7) -> List[Tuple[int, int, float]]:

    similarity_matrix = np.asarray(similarity_matrix)
    if similarity_matrix.ndim != 2:
        raise ValueError('[error] <similarity_matrix> must be 2D')

    used_b = set()
    matches = []

    order_a = np.argsort(-np.max(similarity_matrix, axis = 1))

    for idx_a in order_a:
        row = similarity_matrix[idx_a].copy()
        for b in used_b:
            row[b] = -np.inf
        idx_b = int(np.argmax(row))
        sim = float(row[idx_b])
        if sim >= threshold:
            matches.append((int(idx_a), idx_b, sim))
            used_b.add(idx_b)

    matches.sort(key = lambda x: x[0])

    print_info('match_modes: matched {} modes (threshold={:.2f})'.format(
            len(matches), threshold))

    return matches


def delta_phi_curve(phases: np.ndarray,
        idx_i: int,
        idx_j: int) -> np.ndarray:

    # Inter-mode phase difference Delta-phi(lambda) = phi_i - phi_j, wrapped to
    # (-pi, pi] via angle(exp(1j*.)) then unwrapped along wavelength.
    phases = np.asarray(phases)
    phi_i = phases[idx_i]
    phi_j = phases[idx_j]
    diff = np.angle(np.exp(1j * (phi_i - phi_j)))
    return np.unwrap(diff)


def compare_delta_phi(analyzer_outputs: Dict[str, Dict[str, Any]],
        mode_pair: Tuple[int, int] = (0, 1),
        phase_consistency_threshold: float = np.pi / 8) -> Box:

    # analyzer_outputs: {method_name: {'wavelengths': (nl,), 'phases': (K, nl)}}.
    # Returns the per-method Delta-phi curves plus a circular-mean consistency
    # measure (max/mean angular deviation across methods).
    if len(analyzer_outputs) < 1:
        raise ValueError('[error] empty <analyzer_outputs>')

    idx_i, idx_j = mode_pair

    wavelengths_ref = None
    delta_phi_by_method = {}

    for method_name, output in analyzer_outputs.items():
        wl = np.asarray(output['wavelengths'])
        phases = np.asarray(output['phases'])

        if wavelengths_ref is None:
            wavelengths_ref = wl
        elif len(wl) != len(wavelengths_ref):
            raise ValueError('[error] inconsistent wavelengths across methods')

        if phases.shape[0] <= max(idx_i, idx_j):
            raise ValueError(
                    '[error] mode_pair exceeds available modes for method <{}>'.format(
                            method_name))

        delta_phi_by_method[method_name] = delta_phi_curve(phases, idx_i, idx_j)

    methods = list(delta_phi_by_method.keys())
    n_methods = len(methods)
    nlambda = len(wavelengths_ref)

    if n_methods >= 2:
        all_curves = np.empty((n_methods, nlambda), dtype = float)
        for k, m in enumerate(methods):
            all_curves[k] = delta_phi_by_method[m]

        mean_curve = np.angle(np.mean(np.exp(1j * all_curves), axis = 0))
        deviations = np.empty_like(all_curves)
        for k in range(n_methods):
            deviations[k] = np.abs(np.angle(np.exp(1j * (all_curves[k] - mean_curve))))

        max_deviation = float(np.max(deviations))
        mean_deviation = float(np.mean(deviations))
    else:
        max_deviation = 0.0
        mean_deviation = 0.0

    is_consistent = bool(max_deviation < phase_consistency_threshold)

    result = {
        'wavelengths': wavelengths_ref,
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'is_consistent': is_consistent,
        'mode_pair': tuple(mode_pair),
        'delta_phi': dict(delta_phi_by_method)}

    print_info('compare_delta_phi: max_dev={:.4f} rad, mean_dev={:.4f} rad, consistent={}'.format(
            max_deviation, mean_deviation, is_consistent))

    return Box(result)


def assign_bright_dark(multipole_output: Dict[str, Any],
        magnitudes: np.ndarray) -> Box:

    # bright = high dipole character x coupling, dark = high non-dipole x coupling.
    # multipole_output must provide 'dipole_mag' (K,) and 'character' (list, len K).
    magnitudes = np.asarray(magnitudes)
    if magnitudes.ndim != 2:
        raise ValueError('[error] <magnitudes> must be 2D (n_modes, nlambda)')

    dipole_mag = np.asarray(multipole_output['dipole_mag'])
    character = list(multipole_output['character'])

    n_modes = magnitudes.shape[0]
    if len(dipole_mag) != n_modes:
        raise ValueError('[error] <dipole_mag> length mismatch')
    if len(character) != n_modes:
        raise ValueError('[error] <character> length mismatch')

    peak_mag = np.max(magnitudes, axis = 1)

    eps = 1e-15
    dipole_norm = dipole_mag / (np.max(dipole_mag) + eps)
    coupling_norm = peak_mag / (np.max(peak_mag) + eps)

    bright_score = dipole_norm * coupling_norm
    bright_idx = int(np.argmax(bright_score))

    remaining = [i for i in range(n_modes) if i != bright_idx]
    dark_scores = []
    for i in remaining:
        non_dipole = 1.0 - dipole_norm[i]
        dark_scores.append((i, non_dipole * coupling_norm[i]))
    dark_scores.sort(key = lambda x: -x[1])
    dark_idx = int(dark_scores[0][0]) if dark_scores else bright_idx

    coupling_strength_ratio = float(peak_mag[bright_idx] / (peak_mag[dark_idx] + eps))

    result = {
        'bright_idx': bright_idx,
        'dark_idx': dark_idx,
        'bright_char': character[bright_idx],
        'dark_char': character[dark_idx],
        'coupling_strength_ratio': coupling_strength_ratio}

    print_info('assign_bright_dark: bright=mode {} ({}), dark=mode {} ({}), ratio={:.3f}'.format(
            bright_idx, character[bright_idx], dark_idx, character[dark_idx],
            coupling_strength_ratio))

    return Box(result)


# -------------------------------------------------------------------------
# Sanity test (mirrors mnpbem_simulation/.../mode_comparator.py __main__)
# -------------------------------------------------------------------------

if __name__ == '__main__':

    np.random.seed(0)

    nfaces = 200
    n_modes = 4
    nlambda = 120

    modes_qs = np.random.randn(nfaces, n_modes) + 1j * np.random.randn(nfaces, n_modes)
    noise = 0.05 * (np.random.randn(nfaces, n_modes) + 1j * np.random.randn(nfaces, n_modes))
    modes_svd = modes_qs + noise
    modes_ret = modes_qs + 2.0 * noise

    wavelengths = np.linspace(500.0, 900.0, nlambda)

    def _make_output(seed_offset):
        g_complex = np.empty((n_modes, nlambda), dtype = complex)
        for k in range(n_modes):
            center = 600.0 + 80.0 * k
            gamma = 30.0 + seed_offset
            denom = (wavelengths - center) + 1j * gamma
            g_complex[k] = 1.0 / denom
        return {
            'wavelengths': wavelengths,
            'g_complex': g_complex,
            'magnitudes': np.abs(g_complex),
            'phases': np.angle(g_complex)}

    out_qs = _make_output(0.0)
    out_svd = _make_output(0.3)
    out_ret = _make_output(0.6)

    sim_qs_svd = mode_similarity_matrix(modes_qs, modes_svd)
    matches = match_modes(sim_qs_svd, threshold = 0.5)
    assert len(matches) == n_modes, '[error] expected all modes matched'
    for (a, b, s) in matches:
        assert a == b, '[error] expected diagonal matching'

    outputs = {'qs': out_qs, 'svd': out_svd, 'retarded': out_ret}
    res = compare_delta_phi(outputs, mode_pair = (0, 1))
    assert res.is_consistent, '[error] expected consistent Delta-phi'

    multipole_qs = {
        'dipole_mag': np.array([1.0, 0.2, 0.3, 0.1]),
        'character': ['dipole', 'quadrupole', 'hybrid', 'hybrid']}
    bd = assign_bright_dark(multipole_qs, out_qs['magnitudes'])
    assert bd.bright_idx == 0, '[error] expected mode 0 as bright'
    assert bd.bright_char == 'dipole', '[error] expected dipole bright char'

    print('[info] mode_compare sanity test passed.')
