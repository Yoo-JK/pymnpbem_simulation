"""
Inter-mode phase-difference (Delta-phi) visualizations.

Formalizes the delta_phi prototype plots
(scratch/paper_figures/delta_phi_auag.py and dphi_altviz.py):

    plot_mode_phase      -> 2-panel figure: (a) |g_k(lambda)| modal-projection
                            spectra (bright/dark bold), (b) |Delta-phi|/pi.
    plot_mode_phase_alt  -> 3 alternative views:
                            'phasor'  complex-plane g(lambda) trajectory,
                            'heatmap' mode x lambda phase map arg(g)/pi,
                            'cos'     interference term cos(Delta-phi).

Both consume the Box returned by mode_phase_analysis.
"""

import os

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..util import print_info


_ARIAL = '/home/yoojk20/miniconda3/pkgs/mscorefonts-0.0.1-3/fonts/arial.ttf'


def _use_arial() -> None:
    # Match the prototype: prefer Arial if the bundled mscorefonts ttf exists,
    # otherwise leave the default family untouched.
    if os.path.isfile(_ARIAL):
        from matplotlib import font_manager
        try:
            font_manager.fontManager.addfont(_ARIAL)
            matplotlib.rcParams['font.family'] = 'Arial'
        except Exception:
            pass
    matplotlib.rcParams['font.size'] = 10


def plot_mode_phase(result: Dict[str, Any],
        wavelengths: Optional[np.ndarray] = None,
        out_path: Optional[str] = None,
        n_show: int = 5,
        title_prefix: str = 'Au@Ag dimer',
        dpi: int = 200,
        figsize = (7.0, 8.0)) -> Any:
    """2-panel Delta-phi figure from a mode_phase_analysis result.

    Args:
        result: Box/dict from mode_phase_analysis (needs 'magnitudes', 'phases',
            'bright', 'dark', 'dominant_l', 'delta_phi'; 'wavelengths' optional).
        wavelengths: (nlambda,) nm. If None, taken from result['wavelengths'].
        out_path: if given, the figure is saved there.
        n_show: total number of modes to draw in panel (a) (bright + dark always
            included, padded with the next strongest-coupled modes).
        title_prefix: prefixes panel (a) title (e.g. material/case name).

    Returns: the matplotlib Figure.
    """
    _use_arial()

    mag = np.asarray(result['magnitudes'])
    phase = np.asarray(result['phases'])
    dom_l = np.asarray(result['dominant_l'])
    bright = int(result['bright'])
    dark = int(result['dark'])
    delta_phi = np.asarray(result['delta_phi'])

    if wavelengths is None:
        wavelengths = np.asarray(result['wavelengths'], dtype = float)
    else:
        wavelengths = np.asarray(wavelengths, dtype = float)

    n_modes = mag.shape[0]
    peak = mag.max(axis = 1)
    order = np.argsort(peak)[::-1]

    show = list(dict.fromkeys([bright, dark] + list(order)))[:max(n_show, 2)]

    fig, (axa, axb) = plt.subplots(2, 1, figsize = figsize)

    for m in show:
        tag = ' (bright)' if m == bright else (' (dark)' if m == dark else '')
        axa.plot(wavelengths, mag[m],
                lw = 2.4 if m in (bright, dark) else 1.0,
                label = 'mode {}, l={}{}'.format(m, int(dom_l[m]), tag))
    axa.set_xlabel('Wavelength (nm)')
    axa.set_ylabel(r'$|g_k(\lambda)|$')
    axa.set_title('(a) Mode projection magnitudes -- {} ({})'.format(
            title_prefix, result.get('method', 'qs')))
    axa.legend(frameon = False, fontsize = 9)
    axa.grid(alpha = 0.3)
    for s in ('top', 'right'):
        axa.spines[s].set_visible(False)

    axb.plot(wavelengths, np.abs(delta_phi) / np.pi, color = 'C3', lw = 2.4,
            label = '|bright(m{}) - dark(m{})|'.format(bright, dark))
    axb.set_xlabel('Wavelength (nm)')
    axb.set_ylabel(r'$|\Delta\varphi(\lambda)|\ /\ \pi$')
    axb.set_title('(b) Inter-mode phase difference (bright vs dark)')
    axb.legend(frameon = False, fontsize = 9)
    axb.grid(alpha = 0.3)
    for s in ('top', 'right'):
        axb.spines[s].set_visible(False)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi = dpi)
        print_info('plot_mode_phase: saved <{}>'.format(out_path))

    return fig


def plot_mode_phase_alt(result: Dict[str, Any],
        wavelengths: Optional[np.ndarray] = None,
        style: str = 'phasor',
        out_path: Optional[str] = None,
        dpi: int = 160,
        figsize = None) -> Any:
    """Alternative Delta-phi visualization (single style) from dphi_altviz.py.

    Args:
        result: Box/dict from mode_phase_analysis.
        wavelengths: (nlambda,) nm. If None, taken from result['wavelengths'].
        style: 'phasor'  -> complex-plane g(lambda) trajectory of bright & dark,
                            colored by wavelength.
               'heatmap' -> mode x lambda phase map arg(g)/pi (cyclic twilight),
                            modes ordered by coupling.
               'cos'     -> interference term cos(Delta-phi) with
                            constructive/destructive fill.
        out_path: if given, the figure is saved there.

    Returns: the matplotlib Figure.
    """
    _use_arial()

    g = np.asarray(result['g'])
    mag = np.asarray(result['magnitudes'])
    phase = np.asarray(result['phases'])
    bright = int(result['bright'])
    dark = int(result['dark'])
    n_modes = g.shape[0]

    if wavelengths is None:
        wavelengths = np.asarray(result['wavelengths'], dtype = float)
    else:
        wavelengths = np.asarray(wavelengths, dtype = float)

    if style == 'phasor':
        fig, ax = plt.subplots(1, 1, figsize = figsize or (5.0, 4.6))
        for idx, lab, cmth in [(bright, 'bright m{}'.format(bright), 'Blues'),
                               (dark, 'dark m{}'.format(dark), 'Reds')]:
            gg = g[idx]
            ax.scatter(gg.real, gg.imag, c = wavelengths, cmap = cmth,
                    s = 14, label = lab)
        ax.axhline(0, color = 'k', lw = 0.5, alpha = 0.4)
        ax.axvline(0, color = 'k', lw = 0.5, alpha = 0.4)
        ax.set_xlabel('Re g')
        ax.set_ylabel('Im g')
        ax.set_aspect('equal')
        ax.set_title('phasor trajectory g($\\lambda$)\n(color = wavelength)',
                fontsize = 10)
        ax.legend(fontsize = 8, frameon = False)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    elif style == 'heatmap':
        fig, ax = plt.subplots(1, 1, figsize = figsize or (6.0, 4.6))
        peak = mag.max(axis = 1)
        ord_c = np.argsort(peak)[::-1]
        ph_sorted = phase[ord_c]
        im = ax.pcolormesh(wavelengths, np.arange(n_modes), ph_sorted / np.pi,
                cmap = 'twilight', vmin = -1, vmax = 1, shading = 'auto')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('mode (by coupling)')
        ax.set_title('mode phase map  arg(g)/$\\pi$', fontsize = 10)
        cb = fig.colorbar(im, ax = ax, fraction = 0.046, ticks = [-1, 0, 1])
        cb.set_label('arg(g)/$\\pi$')

    elif style == 'cos':
        fig, ax = plt.subplots(1, 1, figsize = figsize or (5.0, 4.6))
        # cos(Delta-phi); unwrap only for a smooth interference trace.
        dphi = np.unwrap(np.angle(np.exp(1j * (phase[bright] - phase[dark]))))
        cos_dphi = np.cos(dphi)
        ax.plot(wavelengths, cos_dphi, color = 'C2', lw = 2.4)
        ax.axhline(0, color = 'k', lw = 0.6, alpha = 0.5)
        ax.fill_between(wavelengths, cos_dphi, 0, where = cos_dphi > 0,
                color = 'C0', alpha = 0.15)
        ax.fill_between(wavelengths, cos_dphi, 0, where = cos_dphi < 0,
                color = 'C3', alpha = 0.15)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(r'$\cos\,\Delta\varphi$')
        ax.set_title('interference term cos($\\Delta\\varphi$)\n'
                '(+ constructive / - destructive)', fontsize = 10)
        ax.set_ylim(-1.05, 1.05)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    else:
        raise ValueError(
                '[error] invalid <style>: {} (use "phasor"/"heatmap"/"cos")'.format(style))

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi = dpi)
        print_info('plot_mode_phase_alt[{}]: saved <{}>'.format(style, out_path))

    return fig
