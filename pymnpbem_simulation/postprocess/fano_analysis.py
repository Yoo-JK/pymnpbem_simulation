# Fano resonance / bright-dark eigenmode analysis.
#
# Ported from the validated one-off scratch scripts (the algorithm source of
# truth):
#   compute_auag_dipole.py / analyze_dipole.py  -> radiating_dipole_spectrum
#   run_auag_eig_mkl.py                          -> quasistatic_full_eigenbasis
#   bright_dark_viz.py / auag_brightdark.py      -> bright_dark_decompose / plot_bright_dark
#   fano_fit_global.py / fano_sweep.py           -> multi_lorentzian_fano_fit / plot_fano_phase
#   combined_verify.py / auag_verify.py          -> plot_fano_verify
#   new_postprocess.py:geom_info                 -> geom_info
#
# Method (DO NOT change the math):
#   D(w) = sum_f sigma_f * x_f * A_f  (x = gap-axis coordinate), basis-free.
#   Quasistatic FULL eigendecomposition of F = CompGreenStat(p, p).F via
#   scipy.linalg.eig -> (ene, vr); modal dipole dvec = vr.T @ wvec.
#   bright = |dvec| >= thresh * max(|dvec|); a = solve(vr, sigma);
#   sigma_bright = vr[:, bright] @ a[bright]; sigma_dark = sigma - sigma_bright.
#   Multi-complex-Lorentzian fit of D(w):
#     model = a + b*(w - w0ref) + sum_k c_k / ((w - om_k) + 1j*gam_k).

import os

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from box import Box

from ..util import print_info
from .. import sigma_cache as _sc
from .core_shell import make_separator_from_config


HC = 1239.84198


# ---------------------------------------------------------------------------
# Geometry helper (port of new_postprocess.geom_info, masks via repo separator).
# ---------------------------------------------------------------------------

def geom_info(p: Any,
        cfg: Optional[Dict[str, Any]] = None) -> Box:

    pos = np.asarray(p.pos, dtype = float)
    nf = len(pos)

    idx = np.asarray(p.index).reshape(-1) if hasattr(p, 'index') else None
    if idx is not None and len(np.unique(idx)) == 2 and len(idx) == nf:
        fpid = idx.astype(int)
    else:
        ax = int(np.argmax(pos.var(0)))
        fpid = (pos[:, ax] > np.median(pos[:, ax])).astype(int)

    gap_axis = int(np.argmax(pos.var(0)))
    is_dimer = len(np.unique(fpid)) > 1

    core_mask = None
    shell_mask = None
    is_core_shell = False

    if cfg is not None:
        sep = make_separator_from_config(cfg)
        if sep.is_core_shell_structure():
            core_mask = sep.get_core_mask(pos)
            shell_mask = sep.get_shell_mask(pos)
            is_core_shell = True

    if not is_core_shell:
        core_mask, shell_mask = _core_shell_from_inout(p, nf)
        is_core_shell = core_mask is not None

    return Box({
        'fpid': fpid,
        'gap_axis': gap_axis,
        'is_dimer': is_dimer,
        'core_mask': core_mask,
        'shell_mask': shell_mask,
        'is_core_shell': is_core_shell})


def _core_shell_from_inout(p: Any,
        nf: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    # Fallback when no config: medium = outside-eps absent from inside-eps.
    io = np.asarray(p.inout) if hasattr(p, 'inout') else None
    if io is None or io.ndim != 2 or not hasattr(p, 'p') or len(p.p) <= 2:
        return None, None

    inside = set(int(x) for x in io[:, 0])
    outside = set(int(x) for x in io[:, 1])
    med = list(outside - inside)
    if not med:
        return None, None

    shell_mask = np.zeros(nf, dtype = bool)
    off = 0
    for i, sp in enumerate(p.p):
        n_sp = len(np.asarray(sp.pos))
        if int(io[i, 1]) == med[0]:
            shell_mask[off:off + n_sp] = True
        off += n_sp

    return ~shell_mask, shell_mask


# ---------------------------------------------------------------------------
# (1) Radiating longitudinal dipole spectrum  D(w) = sum_f sigma_f * x_f * A_f.
# ---------------------------------------------------------------------------

def radiating_dipole_spectrum(case_dir: str,
        pol: int = 0) -> Tuple[np.ndarray, np.ndarray]:

    import json
    import yaml

    from ..structures import build_structure

    case_dir = str(case_dir)
    cfg = yaml.safe_load(open(os.path.join(case_dir, 'config.yaml')))
    man = json.load(open(os.path.join(case_dir, 'sigma', 'manifest.json')))

    p, _epstab, _nf = build_structure(cfg['structure'], cfg.get('materials', dict()))
    gi = geom_info(p, cfg = cfg)
    ga = int(gi['gap_axis'])

    pos = np.asarray(p.pos, dtype = float)
    area = np.asarray(p.area).reshape(-1)
    wvec = pos[:, ga] * area

    wls = np.array(sorted(set(man['wavelengths_nm'])))
    energies = HC / wls
    exc = man['excitations']
    pols = [e['pol'] for e in exc]
    props = [e['prop_dir'] for e in exc]

    order = np.argsort(energies)
    d_list = []
    e_list = []
    miss = 0
    for wi in order:
        ds = _sc.load_sigma(case_dir, float(wls[wi]), pols, props)
        if ds is None:
            miss += 1
            continue
        sig = _sigma_column(ds, pol)
        d_list.append(wvec @ sig)
        e_list.append(energies[wi])

    es = np.asarray(e_list)
    d = np.asarray(d_list)

    print_info('radiating_dipole_spectrum: {} valid, {} missing (gap_axis={}, pol={})'.format(
            len(es), miss, ga, pol))

    return es, d


def _sigma_column(ds: Dict[str, np.ndarray],
        pol: int) -> np.ndarray:

    # retarded cache stores sig1 (outer surface charge); quasistatic stores sig.
    if ds.get('solver_type') == 'retarded':
        arr = np.asarray(ds['sig1'])
    else:
        arr = np.asarray(ds['sig'])
    if arr.ndim == 2:
        return arr[:, pol]
    return arr


# ---------------------------------------------------------------------------
# (2) Quasistatic FULL eigenbasis  F = CompGreenStat(p, p).F -> eig.
#     HEAVY: ~16 min, multi-GB output for 15072 faces. MUST cache.
# ---------------------------------------------------------------------------

def quasistatic_full_eigenbasis(p: Any,
        cache_path: Optional[str] = None,
        gap_axis: Optional[int] = None) -> Box:

    if cache_path is not None and os.path.exists(cache_path):
        d = np.load(cache_path)
        print_info('quasistatic_full_eigenbasis: loaded cache <{}>'.format(cache_path))
        return Box({
            'ene': np.asarray(d['ene']),
            'vr': np.asarray(d['vr']),
            'dvec': np.asarray(d['dvec'])})

    import time

    import scipy.linalg as sla

    from mnpbem.greenfun import CompGreenStat

    pos = np.asarray(p.pos, dtype = float)
    area = np.asarray(p.area).reshape(-1)
    ga = int(gap_axis) if gap_axis is not None else int(np.argmax(pos.var(0)))
    wvec = pos[:, ga] * area

    g = CompGreenStat(p, p)
    f_mat = g.F

    t0 = time.time()
    ene, vr = sla.eig(f_mat)
    print_info('quasistatic_full_eigenbasis: scipy.linalg.eig done in {:.0f}s (n={})'.format(
            time.time() - t0, f_mat.shape[0]))

    dvec = vr.T @ wvec

    if cache_path is not None:
        np.savez(cache_path, ene = ene, vr = vr, dvec = dvec)
        print_info('quasistatic_full_eigenbasis: saved cache <{}>'.format(cache_path))

    return Box({'ene': ene, 'vr': vr, 'dvec': dvec})


# ---------------------------------------------------------------------------
# (3) Bright / dark decomposition of the driven surface charge sigma.
# ---------------------------------------------------------------------------

def bright_dark_decompose(vr: np.ndarray,
        dvec: np.ndarray,
        sigma: np.ndarray,
        thresh: float = 0.30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    import scipy.linalg as sla

    vr = np.asarray(vr)
    dvec = np.asarray(dvec)
    sigma = np.asarray(sigma)

    bright_idx = np.where(np.abs(dvec) >= thresh * np.abs(dvec).max())[0]

    lu = sla.lu_factor(vr)
    a = sla.lu_solve(lu, sigma)

    sigma_bright = vr[:, bright_idx] @ a[bright_idx]
    sigma_dark = sigma - sigma_bright

    return sigma_bright, sigma_dark, bright_idx


# ---------------------------------------------------------------------------
# (4) Multi-complex-Lorentzian fit of D(w).
# ---------------------------------------------------------------------------

class FanoLorentzianFit(object):

    def __init__(self,
            params: np.ndarray,
            n_lor: int,
            w0ref: float,
            es: np.ndarray,
            d: np.ndarray) -> None:
        self.params = np.asarray(params)
        self.n_lor = int(n_lor)
        self.w0ref = float(w0ref)
        self.es = np.asarray(es)
        self.d = np.asarray(d)

    def unpack(self) -> Tuple[complex, complex, List[Tuple[complex, float, float]]]:
        prm = self.params
        a = prm[0] + 1j * prm[1]
        b = prm[2] + 1j * prm[3]
        lors = []
        for k in range(self.n_lor):
            base = 4 + 4 * k
            c = prm[base] + 1j * prm[base + 1]
            lors.append((c, prm[base + 2], prm[base + 3]))
        return a, b, lors

    def lorentzians(self) -> List[Tuple[complex, float, float]]:
        return self.unpack()[2]

    def centers(self) -> np.ndarray:
        return np.array([om for (_c, om, _g) in self.lorentzians()])

    def model(self,
            w: np.ndarray,
            drop: Optional[int] = None) -> np.ndarray:
        w = np.asarray(w, dtype = float)
        a, b, lors = self.unpack()
        out = a + b * (w - self.w0ref) + 0j
        for k, (c, om, gam) in enumerate(lors):
            if drop is not None and k == drop:
                continue
            out = out + _lor(w, c, om, gam)
        return out

    def rel_rms(self) -> float:
        resid = self.model(self.es) - self.d
        return float(np.sqrt(np.mean(np.abs(resid) ** 2)) / np.max(np.abs(self.d)))

    def nearest_lorentzian(self,
            target_ev: float) -> int:
        return int(np.argmin(np.abs(self.centers() - float(target_ev))))


def _lor(w: np.ndarray,
        c: complex,
        om0: float,
        gam: float) -> np.ndarray:
    return c / ((w - om0) + 1j * gam)


def multi_lorentzian_fano_fit(es: np.ndarray,
        d: np.ndarray,
        init_resonances: Sequence[Tuple[float, float]],
        window: Optional[Tuple[float, float]] = None,
        w0ref: float = 1.65,
        max_nfev: int = 120000) -> FanoLorentzianFit:

    from scipy.optimize import least_squares

    es = np.asarray(es)
    d = np.asarray(d)

    if window is not None:
        m = (es >= window[0]) & (es <= window[1])
        w = es[m]
        dd = d[m]
    else:
        w = es
        dd = d

    n_lor = len(init_resonances)

    def resid(prm):
        fit = FanoLorentzianFit(prm, n_lor, w0ref, w, dd)
        err = fit.model(w) - dd
        return np.concatenate([err.real, err.imag])

    p0 = [np.mean(dd).real, np.mean(dd).imag, 0.0, 0.0]
    sc = np.max(np.abs(dd))
    for (om0, gam) in init_resonances:
        p0 += [0.1 * sc, 0.0, float(om0), float(gam)]

    res = least_squares(resid, p0, args = (), max_nfev = max_nfev)

    fit = FanoLorentzianFit(res.x, n_lor, w0ref, w, dd)
    print_info('multi_lorentzian_fano_fit: n_lor={}, rel-RMS={:.4f}'.format(
            n_lor, fit.rel_rms()))
    for k, (c, om, gam) in enumerate(fit.lorentzians()):
        print_info('  L{}: w0={:.3f} eV  gam={:.3f}  |c|={:.3g}'.format(
                k, om, abs(gam), abs(c)))

    return fit


# ---------------------------------------------------------------------------
# (5) Fano phase sweep plot:  |D|^2 + |Delta phi|  (peak~0 -> flank~pi/2 -> dip~pi).
# ---------------------------------------------------------------------------

def plot_fano_phase(es: np.ndarray,
        d: np.ndarray,
        fit: FanoLorentzianFit,
        dip_ev: float,
        window: Tuple[float, float],
        out_path: str,
        narrow_target: Optional[float] = None) -> str:

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    es = np.asarray(es)
    d = np.asarray(d)
    lo, hi = float(window[0]), float(window[1])

    kk = fit.nearest_lorentzian(narrow_target if narrow_target is not None else dip_ev)
    c, om0, gam = fit.lorentzians()[kk]

    wf = np.linspace(lo, hi, 500)
    full = fit.model(wf)
    narrow = _lor(wf, c, om0, gam)
    bg = full - narrow
    dphi = np.abs(np.angle(narrow / bg)) / np.pi

    full2 = np.abs(full) ** 2
    nrm = float(np.max(full2)) if np.max(full2) > 0 else 1.0

    fig, ax = plt.subplots(figsize = (6.5, 4.2))
    ax.plot(wf, full2 / nrm, 'k-', lw = 2.0, label = r'$|D|^2$ (norm)')
    sel = (es >= lo) & (es <= hi)
    ax.plot(es[sel], np.abs(d[sel]) ** 2 / nrm, 'ko', ms = 4, alpha = 0.6)
    ax.axvline(dip_ev, color = 'r', ls = '--', lw = 1.2)

    ax2 = ax.twinx()
    ax2.plot(wf, dphi * np.pi, color = 'C0', lw = 2.0)
    kx = int(np.argmin(np.abs(wf - dip_ev)))
    ax2.plot(dip_ev, dphi[kx] * np.pi, 'o', color = 'r', ms = 7)
    ax2.annotate(r'dip $|\Delta\varphi|={:.2f}\pi$'.format(dphi[kx]),
            (dip_ev, dphi[kx] * np.pi),
            textcoords = 'offset points', xytext = (6, 4), fontsize = 9, color = 'r')

    ax2.set_ylim(-0.05 * np.pi, 1.15 * np.pi)
    ax2.set_yticks([0.0, np.pi / 2.0, np.pi])
    ax2.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax2.set_ylabel(r'$|\Delta\varphi|$ (narrow $-$ background)', color = 'C0')
    ax2.tick_params(axis = 'y', colors = 'C0')

    ax.set_xlim(lo, hi)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(r'$|D|^2$ (norm)')
    ax.set_title('Fano phase sweep @ {:.2f} eV (narrow L{}, $\\omega_0$={:.3f})'.format(
            dip_ev, kk, om0))
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print_info('plot_fano_phase: saved <{}>'.format(out_path))
    return out_path


# ---------------------------------------------------------------------------
# (6) Mode-removal verify plot: data, full fit, fit-with-one-Lorentzian-dropped.
# ---------------------------------------------------------------------------

def plot_fano_verify(es: np.ndarray,
        d: np.ndarray,
        fit: FanoLorentzianFit,
        drop_idx: int,
        window: Tuple[float, float],
        marks: Sequence[Tuple[float, str, str]],
        out_path: str,
        legend_loc: str = 'upper right') -> str:

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    es = np.asarray(es)
    d = np.asarray(d)
    lo, hi = float(window[0]), float(window[1])

    wf = np.linspace(lo, hi, 600)
    full = np.abs(fit.model(wf)) ** 2
    nodrop = np.abs(fit.model(wf, drop = drop_idx)) ** 2
    nrm = float(max(full.max(), nodrop.max()))
    if nrm <= 0:
        nrm = 1.0

    drop_w0 = fit.lorentzians()[drop_idx][1]

    fig, ax = plt.subplots(figsize = (8.5, 2.8))
    sel = (es >= lo) & (es <= hi)
    ax.plot(es[sel], np.abs(d[sel]) ** 2 / nrm, 'ko', ms = 5, label = r'data $|D|^2$')
    ax.plot(wf, full / nrm, '-', color = 'red', lw = 2.6, label = 'full fit')
    ax.plot(wf, nodrop / nrm, '--', color = 'tab:blue', lw = 2.4,
            label = 'without {:.2f} mode'.format(drop_w0))

    trans = ax.get_xaxis_transform()
    for ev, lab, posn in marks:
        ax.axvline(ev, color = '0.5', ls = ':', lw = 1.2)
        yy, va = (0.90, 'top') if posn == 'top' else (0.06, 'bottom')
        ax.annotate(lab, (ev, yy), xycoords = trans, fontsize = 12, color = '0.25',
                ha = 'center', va = va)

    ax.set_xlabel('Energy (eV)', fontsize = 15)
    ax.set_ylabel(r'$|D|^2$ (norm)', fontsize = 15)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize = 13)
    ax.legend(fontsize = 12, frameon = False, loc = legend_loc)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print_info('plot_fano_verify: saved <{}>'.format(out_path))
    return out_path


# ---------------------------------------------------------------------------
# (7) Bright / dark / total figure: 2 rows (3D body, gap) x 3 cols.
#     Core-shell: semi-transparent shell (alpha) + opaque core; gamma 0.6 boost.
# ---------------------------------------------------------------------------

def plot_bright_dark(p: Any,
        geom: Box,
        sigma_bright: np.ndarray,
        sigma_dark: np.ndarray,
        sigma_total: np.ndarray,
        out_path: str,
        shell_alpha: float = 0.6,
        gamma: float = 0.6,
        title: str = '',
        cmap_name: str = 'RdBu_r') -> str:

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cmap = matplotlib.colormaps[cmap_name]
    ga = int(geom['gap_axis'])

    verts = np.asarray(p.verts, dtype = float)
    faces_i = _clean_faces(p.faces, len(verts))
    pos = np.asarray(p.pos, dtype = float)
    nvec = np.asarray(p.nvec, dtype = float)

    is_cs = bool(geom['is_core_shell'])
    if is_cs:
        shell = np.asarray(geom['shell_mask'], dtype = bool)
        core = np.asarray(geom['core_mask'], dtype = bool)
    else:
        shell = np.zeros(len(pos), dtype = bool)
        core = np.ones(len(pos), dtype = bool)
    sf = faces_i[shell]
    cf = faces_i[core]

    p1, _p2 = _detect_gap_facing(pos, nvec, ga)
    if is_cs:
        p1 = p1[shell[p1]]

    aax, bax = [k for k in range(3) if k != ga]
    used = np.unique(cf if not is_cs else sf)
    mn = verts[used].min(0)
    mx = verts[used].max(0)

    def gcol(vals, vmax):
        x = np.clip(vals / vmax, -1, 1)
        xg = np.sign(x) * np.abs(x) ** gamma
        return cmap(0.5 * (xg + 1))

    def draw3d(ax, fc, vmax):
        ax.computed_zorder = False
        if is_cs:
            ax.add_collection3d(Poly3DCollection(verts[cf], facecolors = gcol(fc[core], vmax),
                    edgecolors = (0, 0, 0, 0.03), linewidths = 0.04,
                    antialiased = True, zorder = 5))
            cols = gcol(fc[shell], vmax).copy()
            cols[:, 3] = shell_alpha
            ax.add_collection3d(Poly3DCollection(verts[sf], facecolors = cols,
                    edgecolors = (0, 0, 0, 0.03 * shell_alpha), linewidths = 0.04,
                    antialiased = True, zorder = 10))
        else:
            ax.add_collection3d(Poly3DCollection(verts[cf], facecolors = gcol(fc[core], vmax),
                    edgecolors = (0, 0, 0, 0.05), linewidths = 0.08,
                    antialiased = True, zorder = 10))
        rng = mx - mn
        rng[rng < 1e-9] = 1e-9
        ax.set_box_aspect(tuple(rng / rng.max()))
        ax.set_xlim(mn[0], mx[0])
        ax.set_ylim(mn[1], mx[1])
        ax.set_zlim(mn[2], mx[2])
        ax.view_init(elev = 18, azim = -62)
        ax.set_axis_off()

    parts = [('bright (superradiant dipole)', np.asarray(sigma_bright).real),
             ('dark (subradiant)', np.asarray(sigma_dark).real),
             ('total = bright + dark', np.asarray(sigma_total).real)]

    ref = np.asarray(sigma_total).real
    vbody = np.percentile(np.abs(ref[shell] if is_cs else ref), 95)
    vgap = np.percentile(np.abs(ref[p1]), 96) if len(p1) > 0 else (vbody or 1.0)
    if vbody <= 0:
        vbody = 1.0
    if vgap <= 0:
        vgap = 1.0

    fig = plt.figure(figsize = (12.5, 8.2))
    gs = GridSpec(2, 3, figure = fig, height_ratios = [1.5, 1.0], hspace = 0.04,
            wspace = 0.04, left = 0.03, right = 0.995, top = 0.92, bottom = 0.02)
    for col, (lab, fc) in enumerate(parts):
        ax3 = fig.add_subplot(gs[0, col], projection = '3d')
        draw3d(ax3, fc, vbody)
        ax3.set_title(lab, fontsize = 12, pad = -2)
        axg = fig.add_subplot(gs[1, col])
        if len(p1) > 0:
            axg.scatter(pos[p1, aax], pos[p1, bax],
                    c = np.clip(fc[p1] / vgap, -1, 1), cmap = cmap_name,
                    vmin = -1, vmax = 1, s = 18, edgecolors = 'none')
        axg.set_aspect('equal')
        axg.set_xticks([])
        axg.set_yticks([])
        if col == 0:
            ax3.text2D(-0.04, 0.5, '3D body', rotation = 90, transform = ax3.transAxes,
                    va = 'center', fontsize = 11)
            axg.set_ylabel('particle1->gap', fontsize = 11)

    fig.suptitle(title or 'bright / dark decomposition', fontsize = 13, y = 0.97)
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print_info('plot_bright_dark: saved <{}>'.format(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Private mesh helpers (ports of surface_charge_viz.clean_faces / detect_gap_facing).
# The repo's plot_surface_charge._detect_gap_facing_faces hardcodes the x-axis and
# returns a dict; these gap-axis-aware variants are what the bright/dark figure needs.
# ---------------------------------------------------------------------------

def _clean_faces(faces: np.ndarray,
        n_verts: int) -> np.ndarray:

    # MNPBEM faces are (N, 4) float; triangles store NaN in trailing column(s)
    # and indices may be 1-based (MATLAB). Return (N, 4) int, 0-based, with a
    # triangle's missing vertex duplicated (degenerate quad renders as triangle).
    f = np.asarray(faces, dtype = float).copy()
    for k in range(1, f.shape[1]):
        bad = np.isnan(f[:, k])
        f[bad, k] = f[bad, k - 1]
    fi = f.astype(int)
    if fi.min() >= 1 and fi.max() >= n_verts:
        fi = fi - 1
    return fi


def _detect_gap_facing(centroids: np.ndarray,
        normals: np.ndarray,
        gap_axis: int = 0,
        threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:

    c = np.asarray(centroids, dtype = float)
    nrm = np.asarray(normals, dtype = float)
    x = c[:, gap_axis]
    gap_center = 0.5 * (x.min() + x.max())
    nn = nrm / (np.linalg.norm(nrm, axis = 1, keepdims = True) + 1e-10)
    d1 = np.zeros(3)
    d1[gap_axis] = 1.0
    p1 = (x < gap_center) & (nn @ d1 > threshold)
    p2 = (x > gap_center) & (nn @ (-d1) > threshold)
    return np.where(p1)[0], np.where(p2)[0]


# ---------------------------------------------------------------------------
# (8) Orchestrator.
# ---------------------------------------------------------------------------

def analyze_fano(case_dir: str,
        features: Sequence[float],
        out_dir: str,
        pol: int = 0,
        eig_cache_path: Optional[str] = None,
        init_resonances: Optional[Sequence[Tuple[float, float]]] = None,
        thresh: float = 0.30,
        fit_window: Optional[Tuple[float, float]] = None,
        phase_halfwidth: float = 0.10) -> Box:

    import json
    import yaml

    from ..structures import build_structure

    case_dir = str(case_dir)
    os.makedirs(out_dir, exist_ok = True)

    cfg = yaml.safe_load(open(os.path.join(case_dir, 'config.yaml')))
    man = json.load(open(os.path.join(case_dir, 'sigma', 'manifest.json')))
    p, _epstab, _nf = build_structure(cfg['structure'], cfg.get('materials', dict()))
    gi = geom_info(p, cfg = cfg)
    ga = int(gi['gap_axis'])

    # (a) dipole spectrum.
    es, d = radiating_dipole_spectrum(case_dir, pol = pol)

    # (b) Lorentzian fit. Default init: one Lorentzian centered near each feature
    #     plus a broad bright continuum, all with narrow widths to capture dips.
    if init_resonances is None:
        init_resonances = _default_init_resonances(es, d, features)
    if fit_window is None:
        fit_window = (float(es.min()), float(es.max()))

    fit = multi_lorentzian_fano_fit(es, d, init_resonances, window = fit_window)

    # (c) quasistatic full eigenbasis (cached) + bright/dark per feature.
    eig = quasistatic_full_eigenbasis(p, cache_path = eig_cache_path, gap_axis = ga)

    pos = np.asarray(p.pos, dtype = float)
    area = np.asarray(p.area).reshape(-1)
    wvec = pos[:, ga] * area

    wls = np.array(sorted(set(man['wavelengths_nm'])))
    energies = HC / wls
    exc = man['excitations']
    pols = [e['pol'] for e in exc]
    props = [e['prop_dir'] for e in exc]

    feature_info = []
    for ev0 in features:
        wi = int(np.argmin(np.abs(energies - ev0)))
        wl = float(wls[wi])
        ev = float(energies[wi])
        ds = _sc.load_sigma(case_dir, wl, pols, props)
        if ds is None:
            print_info('analyze_fano: missing sigma for {:.3f} eV (skip)'.format(ev0))
            continue
        sigma = _sigma_column(ds, pol)
        s_br, s_dk, bright_idx = bright_dark_decompose(eig['vr'], eig['dvec'], sigma,
                thresh = thresh)

        tag = '{:.0f}'.format(ev0 * 100)
        bd_path = os.path.join(out_dir, 'brightdark_{}.png'.format(tag))
        plot_bright_dark(p, gi, s_br, s_dk, sigma, bd_path,
                title = '{} -- {:.2f} eV Fano dip : bright / dark decomposition'.format(
                        os.path.basename(case_dir), ev))

        win = (ev0 - phase_halfwidth, ev0 + phase_halfwidth)
        ph_path = os.path.join(out_dir, 'fano_phase_{}.png'.format(tag))
        plot_fano_phase(es, d, fit, ev0, win, ph_path)

        drop_idx = fit.nearest_lorentzian(ev0)
        marks = [(ev0, '{:.2f} dip'.format(ev0), 'bot')]
        vf_path = os.path.join(out_dir, 'fano_verify_{}.png'.format(tag))
        plot_fano_verify(es, d, fit, drop_idx, win, marks, vf_path)

        dbr = wvec @ s_br
        ddk = wvec @ s_dk
        dt = wvec @ sigma
        feature_info.append({
                'feature_ev': float(ev0),
                'snapped_ev': ev,
                'wavelength_nm': wl,
                'n_bright': int(len(bright_idx)),
                'dark_over_total_dipole': float(abs(ddk) / abs(dt)) if abs(dt) > 0 else 0.0,
                'recon_rel_err': float(np.linalg.norm(s_br + s_dk - sigma) /
                        (np.linalg.norm(sigma) + 1e-30)),
                'narrow_lor_idx': int(drop_idx),
                'bright_dipole': float(abs(dbr)),
                'dark_dipole': float(abs(ddk)),
                'total_dipole': float(abs(dt)),
                'bright_png': bd_path,
                'phase_png': ph_path,
                'verify_png': vf_path})

    summary = Box({
            'case': os.path.basename(case_dir),
            'gap_axis': ga,
            'pol': int(pol),
            'n_wavelengths_valid': int(len(es)),
            'is_core_shell': bool(gi['is_core_shell']),
            'fit_rel_rms': fit.rel_rms(),
            'lorentzians': [{'w0': float(om), 'gamma': float(abs(gam)), 'abs_c': float(abs(c))}
                    for (c, om, gam) in fit.lorentzians()],
            'features': feature_info})

    with open(os.path.join(out_dir, 'fano_summary.json'), 'w') as fh:
        json.dump(summary.to_dict(), fh, indent = 2)

    return summary


def _default_init_resonances(es: np.ndarray,
        d: np.ndarray,
        features: Sequence[float]) -> List[Tuple[float, float]]:

    # One narrow Lorentzian per requested Fano feature (gam ~ 0.05) plus a broad
    # bright continuum at the |D|^2 maximum (gam ~ 0.12). Mirrors the hand-tuned
    # INIT lists in the scratch scripts without hardcoding case-specific values.
    es = np.asarray(es)
    d = np.asarray(d)
    inits = []
    broad_ev = float(es[int(np.argmax(np.abs(d) ** 2))])
    inits.append((broad_ev, 0.12))
    for ev in features:
        inits.append((float(ev), 0.05))
    return inits
