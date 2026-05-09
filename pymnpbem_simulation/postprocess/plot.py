import os

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..util import ensure_dir, print_info


# E(eV) = h*c / λ ≈ 1239.84 / λ(nm)
NM_TO_EV = 1239.84


def plot_spectrum(out_dir: str,
        result: Dict[str, Any],
        title: str = '',
        xaxis: str = 'wavelength',
        polarization_labels: Optional[List[str]] = None,
        plot_format: Optional[List[str]] = None,
        dpi: int = 150) -> str:

    ensure_dir(out_dir)

    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])

    n_pol = ext.shape[1]

    if xaxis == 'energy':
        # E(eV) = 1239.84 / λ(nm); reverse so x increases.
        xdata = NM_TO_EV / wavelength
        xdata = xdata[::-1]
        ext = ext[::-1, :]
        sca = sca[::-1, :]
        xlabel = 'Energy (eV)'
    else:
        xdata = wavelength
        xlabel = 'Wavelength (nm)'

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    for i in range(n_pol):
        label = polarization_labels[i] if (polarization_labels is not None
                and i < len(polarization_labels)) else 'pol {}'.format(i)
        axes[0].plot(xdata, ext[:, i], label = label)
        axes[1].plot(xdata, sca[:, i], label = label)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Extinction')
    axes[0].set_title('Extinction')
    axes[0].legend(fontsize = 9)
    axes[0].grid(True, alpha = 0.3)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Scattering')
    axes[1].set_title('Scattering')
    axes[1].legend(fontsize = 9)
    axes[1].grid(True, alpha = 0.3)

    if title != '':
        fig.suptitle(title)

    fig.tight_layout()

    if plot_format is None:
        plot_format = ['png']

    saved_paths = []
    for fmt in plot_format:
        path = os.path.join(out_dir, 'spectrum.{}'.format(fmt))
        fig.savefig(path, dpi = dpi)
        saved_paths.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)

    return saved_paths[0] if saved_paths else ''


def plot_polarization_comparison(out_dir: str,
        result: Dict[str, Any],
        title: str = '',
        xaxis: str = 'wavelength',
        polarization_labels: Optional[List[str]] = None,
        plot_format: Optional[List[str]] = None,
        dpi: int = 150) -> List[str]:
    """Plot per-channel comparison of polarizations (extinction / scattering / absorption).

    Creates 3 figures:
        spectrum_polarization_extinction.{fmt}
        spectrum_polarization_scattering.{fmt}
        spectrum_polarization_absorption.{fmt}
    """
    ensure_dir(out_dir)

    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])
    abs_ = np.asarray(result['abs'])

    n_pol = ext.shape[1]

    if xaxis == 'energy':
        xdata = NM_TO_EV / wavelength
        xdata = xdata[::-1]
        ext = ext[::-1, :]
        sca = sca[::-1, :]
        abs_ = abs_[::-1, :]
        xlabel = 'Energy (eV)'
    else:
        xdata = wavelength
        xlabel = 'Wavelength (nm)'

    if plot_format is None:
        plot_format = ['png']

    if polarization_labels is None:
        polarization_labels = ['pol {}'.format(i) for i in range(n_pol)]

    colors = plt.cm.viridis(np.linspace(0, 1, max(n_pol, 1)))

    saved_files = []
    channels = [
            ('extinction', ext, 'Extinction Cross Section (nm$^2$)'),
            ('scattering', sca, 'Scattering Cross Section (nm$^2$)'),
            ('absorption', abs_, 'Absorption Cross Section (nm$^2$)')]

    for ch_name, ch_data, ylabel in channels:
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            label = polarization_labels[i] if i < len(polarization_labels) else 'pol {}'.format(i)
            ax.plot(xdata, ch_data[:, i],
                    color = colors[i], linewidth = 2,
                    label = label)

        ax.set_xlabel(xlabel, fontsize = 12)
        ax.set_ylabel(ylabel, fontsize = 12)
        ax.set_title('Polarization Comparison - {}'.format(ch_name.capitalize()),
                fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if title != '':
            fig.suptitle(title)

        fig.tight_layout()

        for fmt in plot_format:
            path = os.path.join(out_dir, 'spectrum_polarization_{}.{}'.format(ch_name, fmt))
            fig.savefig(path, dpi = dpi)
            saved_files.append(path)
            print_info('saved <{}>'.format(path))

        plt.close(fig)

    return saved_files


def plot_unpolarized_spectrum(out_dir: str,
        result: Dict[str, Any],
        unpol_data: Dict[str, Any],
        title: str = '',
        xaxis: str = 'wavelength',
        plot_format: Optional[List[str]] = None,
        dpi: int = 150) -> List[str]:
    """Plot unpolarized (FDTD-style incoherent average) spectrum standalone.

    `unpol_data` is the dict returned by `analyze_spectrum_unpolarized`,
    containing keys: wavelength, extinction, scattering, absorption, n_averaged, method.
    """
    ensure_dir(out_dir)

    wavelength = np.asarray(unpol_data['wavelength'])
    unpol_ext = np.asarray(unpol_data['extinction'])
    unpol_sca = np.asarray(unpol_data['scattering'])
    unpol_abs = np.asarray(unpol_data['absorption'])

    if xaxis == 'energy':
        xdata = NM_TO_EV / wavelength
        xdata = xdata[::-1]
        unpol_ext = unpol_ext[::-1]
        unpol_sca = unpol_sca[::-1]
        unpol_abs = unpol_abs[::-1]
        xlabel = 'Energy (eV)'
    else:
        xdata = wavelength
        xlabel = 'Wavelength (nm)'

    if plot_format is None:
        plot_format = ['png']

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(xdata, unpol_ext, 'b-', linewidth = 2, label = 'Extinction')
    ax.plot(xdata, unpol_sca, 'r--', linewidth = 2, label = 'Scattering')
    ax.plot(xdata, unpol_abs, 'g:', linewidth = 2, label = 'Absorption')

    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel('Cross Section (nm$^2$)', fontsize = 12)
    ax.set_title('Unpolarized Spectrum (FDTD-style {} pol average)'.format(
            unpol_data.get('n_averaged', '?')),
            fontsize = 14, fontweight = 'bold')
    ax.legend(fontsize = 11)
    ax.grid(True, alpha = 0.3)

    if title != '':
        fig.suptitle(title)

    fig.tight_layout()

    saved_files = []
    for fmt in plot_format:
        path = os.path.join(out_dir, 'spectrum_unpolarized.{}'.format(fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)

    return saved_files


def plot_polarization_vs_unpolarized(out_dir: str,
        result: Dict[str, Any],
        unpol_data: Dict[str, Any],
        title: str = '',
        xaxis: str = 'wavelength',
        polarization_labels: Optional[List[str]] = None,
        plot_format: Optional[List[str]] = None,
        dpi: int = 150) -> List[str]:
    """Plot comparison of all polarizations + unpolarized average.

    Creates 4 figures:
        spectrum_comparison_extinction_unpolarized
        spectrum_comparison_scattering_unpolarized
        spectrum_comparison_absorption_unpolarized
        spectrum_comparison_all_unpolarized (3 subplots)
    """
    ensure_dir(out_dir)

    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])
    abs_ = np.asarray(result['abs'])

    n_pol = ext.shape[1]

    unpol_ext = np.asarray(unpol_data['extinction'])
    unpol_sca = np.asarray(unpol_data['scattering'])
    unpol_abs = np.asarray(unpol_data['absorption'])

    if xaxis == 'energy':
        xdata = NM_TO_EV / wavelength
        xdata = xdata[::-1]
        ext = ext[::-1, :]
        sca = sca[::-1, :]
        abs_ = abs_[::-1, :]
        unpol_ext = unpol_ext[::-1]
        unpol_sca = unpol_sca[::-1]
        unpol_abs = unpol_abs[::-1]
        xlabel = 'Energy (eV)'
    else:
        xdata = wavelength
        xlabel = 'Wavelength (nm)'

    if plot_format is None:
        plot_format = ['png']

    if polarization_labels is None:
        polarization_labels = ['pol {}'.format(i) for i in range(n_pol)]

    colors = plt.cm.tab10(np.linspace(0, 0.7, max(n_pol, 1)))

    saved_files = []
    channels = [
            ('extinction', ext, unpol_ext, 'Extinction Cross Section (nm$^2$)'),
            ('scattering', sca, unpol_sca, 'Scattering Cross Section (nm$^2$)'),
            ('absorption', abs_, unpol_abs, 'Absorption Cross Section (nm$^2$)')]

    for ch_name, ch_data, ch_unpol, ylabel in channels:
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            label = polarization_labels[i] if i < len(polarization_labels) else 'pol {}'.format(i)
            ax.plot(xdata, ch_data[:, i], color = colors[i], linewidth = 1.5,
                    linestyle = '--', alpha = 0.7, label = label)

        ax.plot(xdata, ch_unpol, 'k-', linewidth = 2.5, label = 'Unpolarized')

        ax.set_xlabel(xlabel, fontsize = 12)
        ax.set_ylabel(ylabel, fontsize = 12)
        ax.set_title('{}: Polarizations vs Unpolarized'.format(ch_name.capitalize()),
                fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if title != '':
            fig.suptitle(title)

        fig.tight_layout()

        for fmt in plot_format:
            path = os.path.join(out_dir, 'spectrum_comparison_{}_unpolarized.{}'.format(ch_name, fmt))
            fig.savefig(path, dpi = dpi)
            saved_files.append(path)
            print_info('saved <{}>'.format(path))

        plt.close(fig)

    # All-in-one (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))

    for ax_i, (ch_name, ch_data, ch_unpol, ylabel) in enumerate(channels):
        for i in range(n_pol):
            label = polarization_labels[i] if i < len(polarization_labels) else 'pol {}'.format(i)
            axes[ax_i].plot(xdata, ch_data[:, i], color = colors[i], linewidth = 1.5,
                    linestyle = '--', alpha = 0.7, label = label)
        axes[ax_i].plot(xdata, ch_unpol, 'k-', linewidth = 2.5, label = 'Unpolarized')

        axes[ax_i].set_xlabel(xlabel, fontsize = 11)
        axes[ax_i].set_ylabel(ylabel, fontsize = 11)
        axes[ax_i].set_title(ch_name.capitalize(), fontsize = 12, fontweight = 'bold')
        axes[ax_i].legend(fontsize = 8)
        axes[ax_i].grid(True, alpha = 0.3)

    fig.suptitle(title if title != '' else 'Polarizations vs Unpolarized',
            fontsize = 14, fontweight = 'bold')
    fig.tight_layout()

    for fmt in plot_format:
        path = os.path.join(out_dir, 'spectrum_comparison_all_unpolarized.{}'.format(fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)

    return saved_files


def plot_multipole_bar(out_dir: str,
        multipole: Dict[str, Any],
        title: str = '',
        plot_format: Optional[List[str]] = None,
        dpi: int = 150) -> List[str]:
    """Bar chart of power per l-shell from multipole_decomposition output.

    `multipole` must have keys 'power_l' (1D array) and 'max_l' (int).
    Saves multipole_power_l.{fmt}.
    """
    ensure_dir(out_dir)

    power_l = np.asarray(multipole['power_l'])
    max_l = int(multipole.get('max_l', len(power_l) - 1))

    if plot_format is None:
        plot_format = ['png']

    l_axis = np.arange(0, max_l + 1)

    fig, ax = plt.subplots(figsize = (8, 5))

    ax.bar(l_axis, power_l, color = plt.cm.viridis(np.linspace(0.2, 0.9, len(l_axis))),
            edgecolor = 'k')

    ax.set_xlabel('Multipole order $\\ell$', fontsize = 12)
    ax.set_ylabel('Power $\\sum_m |a_{\\ell m}|^2$', fontsize = 12)
    ax.set_title(title if title else 'Multipole decomposition power per l',
            fontsize = 13, fontweight = 'bold')
    ax.set_xticks(l_axis)

    if power_l.max() > 0 and power_l.min() >= 0:
        ax.set_yscale('log')

    ax.grid(True, alpha = 0.3, axis = 'y', which = 'both')

    fig.tight_layout()

    saved_files = []
    for fmt in plot_format:
        path = os.path.join(out_dir, 'multipole_power_l.{}'.format(fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)
    return saved_files


def plot_fano_fit(out_dir: str,
        x: np.ndarray,
        spectrum: np.ndarray,
        fano_result: Dict[str, Any],
        title: str = '',
        x_label: str = 'Wavelength (nm)',
        y_label: str = 'Cross section (nm$^2$)',
        plot_format: Optional[List[str]] = None,
        dpi: int = 150,
        fname: str = 'fano_fit') -> List[str]:
    """Plot data + best-fit Fano curve with parameter annotation.

    `fano_result` must contain 'fit_curve' and 'best_fit' (dict of
    amp/x0/gamma/q/c). Compatible with fano_fit single-peak output.
    """
    ensure_dir(out_dir)

    if plot_format is None:
        plot_format = ['png']

    x = np.asarray(x)
    spectrum = np.asarray(spectrum)
    fit_curve = np.asarray(fano_result.get('fit_curve', np.zeros_like(spectrum)))

    fig, ax = plt.subplots(figsize = (10, 6))

    ax.plot(x, spectrum, 'o', markersize = 4, color = 'steelblue', alpha = 0.6,
            label = 'data')
    ax.plot(x, fit_curve, '-', linewidth = 2, color = 'crimson', label = 'fit')
    ax.plot(x, spectrum - fit_curve, ':', linewidth = 1, color = 'gray',
            label = 'residual')

    ax.set_xlabel(x_label, fontsize = 12)
    ax.set_ylabel(y_label, fontsize = 12)
    ax.set_title(title if title else 'Fano resonance fit', fontsize = 13, fontweight = 'bold')
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    bf = fano_result.get('best_fit', dict())
    redchi = float(fano_result.get('redchi', float('nan')))

    annotation_lines = []
    for key in ('x0', 'gamma', 'q', 'amp', 'c'):
        if key in bf:
            annotation_lines.append('{} = {:.3g}'.format(key, bf[key]))

    if annotation_lines:
        annotation_lines.append('redchi = {:.3e}'.format(redchi))
        ax.text(0.02, 0.98, '\n'.join(annotation_lines),
                transform = ax.transAxes, fontsize = 9,
                verticalalignment = 'top',
                bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.85))

    fig.tight_layout()

    saved_files = []
    for fmt in plot_format:
        path = os.path.join(out_dir, '{}.{}'.format(fname, fmt))
        fig.savefig(path, dpi = dpi)
        saved_files.append(path)
        print_info('saved <{}>'.format(path))

    plt.close(fig)
    return saved_files
