import os
import sys
import json
import glob
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ======================================================================
# Path configuration
# ======================================================================

MATLAB_BASE = os.path.expanduser('~/temporary/mat2py/rod')
PYTHON_BASE = os.path.expanduser('~/scratch/mat2py/rod')
OUTPUT_DIR = os.path.expanduser('~/scratch/mat2py/rod/comparison')

SOLVERS = ['stat', 'ret']

MATLAB_PATHS = {
    'stat': os.path.join(MATLAB_BASE, 'stat', 'aunr_22x47'),
    'ret': os.path.join(MATLAB_BASE, 'ret', 'aunr_22x47'),
}

PYTHON_PATHS = {
    'stat': os.path.join(PYTHON_BASE, 'stat', 'aunr_22x47', 'aunr_22x47'),
    'ret': os.path.join(PYTHON_BASE, 'ret', 'aunr_22x47', 'aunr_22x47'),
}


# ======================================================================
# Data loading
# ======================================================================

class SpectraData(object):

    def __init__(self,
            wavelength: np.ndarray,
            scattering: np.ndarray,
            extinction: np.ndarray,
            absorption: np.ndarray):

        self.wavelength = wavelength
        self.scattering = scattering
        self.extinction = extinction
        self.absorption = absorption


class FieldData(object):

    def __init__(self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            enhancement: np.ndarray,
            intensity: np.ndarray,
            grid_shape: Tuple[int, int]):

        self.x = x
        self.y = y
        self.z = z
        self.enhancement = enhancement
        self.intensity = intensity
        self.grid_shape = grid_shape


def load_spectra(filepath: str) -> Optional[SpectraData]:
    if not os.path.isfile(filepath):
        return None
    data = np.loadtxt(filepath, comments = '#')
    return SpectraData(
        wavelength = data[:, 0],
        scattering = data[:, 1],
        extinction = data[:, 2],
        absorption = data[:, 3])


def load_field(filepath: str) -> Optional[FieldData]:
    if not os.path.isfile(filepath):
        return None
    data = np.loadtxt(filepath, comments = '#')
    x = data[:, 0]
    z = data[:, 2]
    enhancement = data[:, 3]
    intensity = data[:, 4]

    # Determine grid shape from unique coordinates
    x_unique = np.unique(x)
    z_unique = np.unique(z)
    n_x = len(x_unique)
    n_z = len(z_unique)

    return FieldData(
        x = x, y = data[:, 1], z = z,
        enhancement = enhancement,
        intensity = intensity,
        grid_shape = (n_x, n_z))


def load_field_analysis(filepath: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r') as fh:
        return json.load(fh)


def load_analysis_summary(filepath: str) -> Optional[str]:
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r') as fh:
        return fh.read()


def find_field_files(directory: str) -> List[str]:
    pattern = os.path.join(directory, 'field_pol*_*nm.txt')
    return sorted(glob.glob(pattern))


def parse_field_filename(filepath: str) -> Tuple[str, str]:
    # e.g. field_pol1_609nm.txt -> ('pol1', '609nm')
    basename = os.path.basename(filepath)
    parts = basename.replace('.txt', '').split('_')
    pol = parts[1]  # pol1 or pol2
    wl = parts[2]   # 609nm
    return pol, wl


# ======================================================================
# Metrics
# ======================================================================

def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_r_squared(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def find_peak(wavelength: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    idx = np.argmax(values)
    return float(wavelength[idx]), float(values[idx])


# ======================================================================
# Plotting
# ======================================================================

def plot_spectra_comparison(
        matlab_unpol: SpectraData,
        python_unpol: Optional[SpectraData],
        matlab_pol1: SpectraData,
        python_pol1: Optional[SpectraData],
        matlab_pol2: SpectraData,
        python_pol2: Optional[SpectraData],
        solver: str,
        output_dir: str) -> None:

    # -- Unpolarized overlay plot --
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    fig.suptitle('Rod (aunr 22x47) - {} solver - Unpolarized Spectra'.format(solver.upper()), fontsize = 14)

    labels = ['Absorption', 'Scattering', 'Extinction']
    matlab_data_list = [matlab_unpol.absorption, matlab_unpol.scattering, matlab_unpol.extinction]
    wl = matlab_unpol.wavelength

    for i, (ax, label, m_data) in enumerate(zip(axes, labels, matlab_data_list)):
        ax.plot(wl, m_data, 'b-', linewidth = 1.5, label = 'MATLAB')
        if python_unpol is not None:
            p_data = [python_unpol.absorption, python_unpol.scattering, python_unpol.extinction][i]
            ax.plot(python_unpol.wavelength, p_data, 'r--', linewidth = 1.5, label = 'Python')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('{} cross-section (nm^2)'.format(label))
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectra_unpolarized_{}.png'.format(solver)), dpi = 150)
    plt.close()

    # -- Polarization-resolved overlay plot --
    fig, axes = plt.subplots(2, 3, figsize = (18, 10))
    fig.suptitle('Rod (aunr 22x47) - {} solver - Polarization-resolved Spectra'.format(solver.upper()), fontsize = 14)

    pol_names = ['Pol1 (longitudinal)', 'Pol2 (transverse)']
    matlab_pols = [matlab_pol1, matlab_pol2]
    python_pols = [python_pol1, python_pol2]

    for row, (pol_name, m_pol, p_pol) in enumerate(zip(pol_names, matlab_pols, python_pols)):
        for col, (label, attr) in enumerate(zip(labels, ['absorption', 'scattering', 'extinction'])):
            ax = axes[row, col]
            m_vals = getattr(m_pol, attr)
            ax.plot(m_pol.wavelength, m_vals, 'b-', linewidth = 1.5, label = 'MATLAB')
            if p_pol is not None:
                p_vals = getattr(p_pol, attr)
                ax.plot(p_pol.wavelength, p_vals, 'r--', linewidth = 1.5, label = 'Python')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('{} (nm^2)'.format(label))
            ax.set_title('{} - {}'.format(pol_name, label))
            ax.legend()
            ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectra_polarized_{}.png'.format(solver)), dpi = 150)
    plt.close()


def reshape_field_to_2d(field: FieldData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_x, n_z = field.grid_shape
    x_2d = field.x.reshape(n_z, n_x)
    z_2d = field.z.reshape(n_z, n_x)
    enh_2d = field.enhancement.reshape(n_z, n_x)
    return x_2d, z_2d, enh_2d


def plot_field_comparison(
        matlab_field: FieldData,
        python_field: Optional[FieldData],
        pol: str,
        wl: str,
        solver: str,
        output_dir: str) -> None:

    m_x, m_z, m_enh = reshape_field_to_2d(matlab_field)

    if python_field is not None:
        fig, axes = plt.subplots(1, 3, figsize = (20, 6))
        fig.suptitle('Rod - {} - {} {} - Field Enhancement'.format(solver.upper(), pol, wl), fontsize = 14)

        vmin = 0
        vmax = max(m_enh.max(), 5.0)

        # MATLAB
        im0 = axes[0].pcolormesh(m_x, m_z, m_enh, shading = 'auto', cmap = 'hot', vmin = vmin, vmax = vmax)
        axes[0].set_title('MATLAB')
        axes[0].set_xlabel('x (nm)')
        axes[0].set_ylabel('z (nm)')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax = axes[0], label = '|E|/|E0|')

        # Python
        p_x, p_z, p_enh = reshape_field_to_2d(python_field)
        vmax = max(vmax, p_enh.max())

        # Re-draw MATLAB with updated vmax
        axes[0].clear()
        im0 = axes[0].pcolormesh(m_x, m_z, m_enh, shading = 'auto', cmap = 'hot', vmin = vmin, vmax = vmax)
        axes[0].set_title('MATLAB')
        axes[0].set_xlabel('x (nm)')
        axes[0].set_ylabel('z (nm)')
        axes[0].set_aspect('equal')

        im1 = axes[1].pcolormesh(p_x, p_z, p_enh, shading = 'auto', cmap = 'hot', vmin = vmin, vmax = vmax)
        axes[1].set_title('Python')
        axes[1].set_xlabel('x (nm)')
        axes[1].set_ylabel('z (nm)')
        axes[1].set_aspect('equal')

        # Difference
        diff = p_enh - m_enh
        diff_max = max(abs(diff.min()), abs(diff.max()), 1e-10)
        im2 = axes[2].pcolormesh(m_x, m_z, diff, shading = 'auto', cmap = 'RdBu_r', vmin = -diff_max, vmax = diff_max)
        axes[2].set_title('Difference (Python - MATLAB)')
        axes[2].set_xlabel('x (nm)')
        axes[2].set_ylabel('z (nm)')
        axes[2].set_aspect('equal')

        # Shared colorbar for MATLAB and Python
        fig.colorbar(im0, ax = axes[:2], label = '|E|/|E0|', shrink = 0.8)
        fig.colorbar(im2, ax = axes[2], label = 'Difference', shrink = 0.8)

    else:
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        fig.suptitle('Rod - {} - {} {} - Field Enhancement (MATLAB only)'.format(solver.upper(), pol, wl), fontsize = 14)

        im = ax.pcolormesh(m_x, m_z, m_enh, shading = 'auto', cmap = 'hot')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('z (nm)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax = ax, label = '|E|/|E0|')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'field_{}_{}_{}_.png'.format(pol, wl, solver)), dpi = 150)
    plt.close()


# ======================================================================
# Summary generation
# ======================================================================

def generate_spectra_metrics(
        matlab_spec: SpectraData,
        python_spec: Optional[SpectraData],
        label: str) -> List[str]:

    lines = []
    m_abs_peak_wl, m_abs_peak_val = find_peak(matlab_spec.wavelength, matlab_spec.absorption)
    m_ext_peak_wl, m_ext_peak_val = find_peak(matlab_spec.wavelength, matlab_spec.extinction)
    m_sca_peak_wl, m_sca_peak_val = find_peak(matlab_spec.wavelength, matlab_spec.scattering)

    lines.append('  {} - MATLAB:'.format(label))
    lines.append('    Peak absorption:  {:.2f} nm^2 @ {:.1f} nm'.format(m_abs_peak_val, m_abs_peak_wl))
    lines.append('    Peak extinction:  {:.2f} nm^2 @ {:.1f} nm'.format(m_ext_peak_val, m_ext_peak_wl))
    lines.append('    Peak scattering:  {:.2f} nm^2 @ {:.1f} nm'.format(m_sca_peak_val, m_sca_peak_wl))

    if python_spec is not None:
        p_abs_peak_wl, p_abs_peak_val = find_peak(python_spec.wavelength, python_spec.absorption)
        p_ext_peak_wl, p_ext_peak_val = find_peak(python_spec.wavelength, python_spec.extinction)
        p_sca_peak_wl, p_sca_peak_val = find_peak(python_spec.wavelength, python_spec.scattering)

        lines.append('  {} - Python:'.format(label))
        lines.append('    Peak absorption:  {:.2f} nm^2 @ {:.1f} nm'.format(p_abs_peak_val, p_abs_peak_wl))
        lines.append('    Peak extinction:  {:.2f} nm^2 @ {:.1f} nm'.format(p_ext_peak_val, p_ext_peak_wl))
        lines.append('    Peak scattering:  {:.2f} nm^2 @ {:.1f} nm'.format(p_sca_peak_val, p_sca_peak_wl))

        lines.append('  {} - Comparison:'.format(label))

        # Peak wavelength differences
        lines.append('    Abs peak wavelength diff: {:.2f} nm'.format(p_abs_peak_wl - m_abs_peak_wl))
        lines.append('    Ext peak wavelength diff: {:.2f} nm'.format(p_ext_peak_wl - m_ext_peak_wl))
        lines.append('    Sca peak wavelength diff: {:.2f} nm'.format(p_sca_peak_wl - m_sca_peak_wl))

        # Peak value differences
        for name, m_val, p_val in [('Abs peak value', m_abs_peak_val, p_abs_peak_val),
                                    ('Ext peak value', m_ext_peak_val, p_ext_peak_val),
                                    ('Sca peak value', m_sca_peak_val, p_sca_peak_val)]:
            diff = p_val - m_val
            rel_pct = 100.0 * diff / m_val if m_val != 0 else 0.0
            lines.append('    {} diff: {:.2f} nm^2 ({:+.2f}%)'.format(name, diff, rel_pct))

        # Whole-spectrum RMSE and R^2
        # Interpolate python onto matlab wavelength grid if needed
        if len(matlab_spec.wavelength) == len(python_spec.wavelength) and np.allclose(matlab_spec.wavelength, python_spec.wavelength, atol = 0.01):
            p_abs = python_spec.absorption
            p_ext = python_spec.extinction
            p_sca = python_spec.scattering
        else:
            p_abs = np.interp(matlab_spec.wavelength, python_spec.wavelength, python_spec.absorption)
            p_ext = np.interp(matlab_spec.wavelength, python_spec.wavelength, python_spec.extinction)
            p_sca = np.interp(matlab_spec.wavelength, python_spec.wavelength, python_spec.scattering)

        for name, m_arr, p_arr in [('Absorption', matlab_spec.absorption, p_abs),
                                    ('Extinction', matlab_spec.extinction, p_ext),
                                    ('Scattering', matlab_spec.scattering, p_sca)]:
            rmse = compute_rmse(m_arr, p_arr)
            r2 = compute_r_squared(m_arr, p_arr)
            lines.append('    {} RMSE: {:.4f}, R^2: {:.6f}'.format(name, rmse, r2))

    return lines


def generate_field_metrics(
        matlab_analysis: Dict[str, Any],
        python_analysis: Optional[Dict[str, Any]]) -> List[str]:

    lines = []

    for field_entry in matlab_analysis.get('fields', []):
        pol_idx = field_entry['polarization_index']
        pol_vec = field_entry['polarization']
        wl = field_entry['wavelength_nm']
        stats = field_entry['analysis']['enhancement_stats']
        hotspots = field_entry['analysis']['hotspots']

        lines.append('  Polarization {} {} @ {:.1f} nm - MATLAB:'.format(pol_idx, pol_vec, wl))
        lines.append('    Max enhancement:  {:.4f}'.format(stats['max']))
        lines.append('    Mean enhancement: {:.4f}'.format(stats['mean']))
        if hotspots:
            top = hotspots[0]
            lines.append('    Top hotspot: ({}, {}, {}) enhancement={:.4f}'.format(
                top['position'][0], top['position'][1], top['position'][2], top['enhancement']))

    if python_analysis is not None:
        for field_entry in python_analysis.get('fields', []):
            pol_idx = field_entry['polarization_index']
            pol_vec = field_entry['polarization']
            wl = field_entry['wavelength_nm']
            stats = field_entry['analysis']['enhancement_stats']
            hotspots = field_entry['analysis']['hotspots']

            lines.append('  Polarization {} {} @ {:.1f} nm - Python:'.format(pol_idx, pol_vec, wl))
            lines.append('    Max enhancement:  {:.4f}'.format(stats['max']))
            lines.append('    Mean enhancement: {:.4f}'.format(stats['mean']))
            if hotspots:
                top = hotspots[0]
                lines.append('    Top hotspot: ({}, {}, {}) enhancement={:.4f}'.format(
                    top['position'][0], top['position'][1], top['position'][2], top['enhancement']))

        # Compare matching polarizations
        m_fields = {f['polarization_index']: f for f in matlab_analysis.get('fields', [])}
        p_fields = {f['polarization_index']: f for f in python_analysis.get('fields', [])}

        for pol_idx in sorted(set(m_fields.keys()) & set(p_fields.keys())):
            m_f = m_fields[pol_idx]
            p_f = p_fields[pol_idx]
            m_stats = m_f['analysis']['enhancement_stats']
            p_stats = p_f['analysis']['enhancement_stats']

            max_diff = p_stats['max'] - m_stats['max']
            max_rel = 100.0 * max_diff / m_stats['max'] if m_stats['max'] != 0 else 0.0

            lines.append('  Polarization {} - Comparison:'.format(pol_idx))
            lines.append('    Max enhancement diff: {:.4f} ({:+.2f}%)'.format(max_diff, max_rel))
            lines.append('    Wavelength: MATLAB {:.1f} nm vs Python {:.1f} nm'.format(
                m_f['wavelength_nm'], p_f['wavelength_nm']))

            # Hotspot position comparison
            m_hotspots = m_f['analysis']['hotspots']
            p_hotspots = p_f['analysis']['hotspots']
            if m_hotspots and p_hotspots:
                m_top = m_hotspots[0]['position']
                p_top = p_hotspots[0]['position']
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(m_top, p_top)))
                lines.append('    Top hotspot distance: {:.2f} nm (MATLAB {} vs Python {})'.format(
                    dist, m_top, p_top))

    return lines


def generate_comparison_table(
        matlab_data: Dict[str, Any],
        python_data: Dict[str, Any],
        solver: str) -> List[str]:

    lines = []
    lines.append('+' + '-' * 78 + '+')
    lines.append('| {:^76s} |'.format('{} Solver - Comparison Table'.format(solver.upper())))
    lines.append('+' + '-' * 30 + '+' + '-' * 23 + '+' + '-' * 23 + '+')
    lines.append('| {:<28s} | {:<21s} | {:<21s} |'.format('Metric', 'MATLAB', 'Python'))
    lines.append('+' + '-' * 30 + '+' + '-' * 23 + '+' + '-' * 23 + '+')

    # Spectra peaks - unpolarized
    m_unpol = matlab_data.get('spectra_unpolarized')
    p_unpol = python_data.get('spectra_unpolarized')

    if m_unpol is not None:
        m_abs_wl, m_abs_val = find_peak(m_unpol.wavelength, m_unpol.absorption)
        m_ext_wl, m_ext_val = find_peak(m_unpol.wavelength, m_unpol.extinction)

        p_abs_str = 'N/A'
        p_ext_str = 'N/A'
        if p_unpol is not None:
            p_abs_wl, p_abs_val = find_peak(p_unpol.wavelength, p_unpol.absorption)
            p_ext_wl, p_ext_val = find_peak(p_unpol.wavelength, p_unpol.extinction)
            p_abs_str = '{:.1f}nm / {:.0f}'.format(p_abs_wl, p_abs_val)
            p_ext_str = '{:.1f}nm / {:.0f}'.format(p_ext_wl, p_ext_val)

        lines.append('| {:<28s} | {:<21s} | {:<21s} |'.format(
            'Unpol abs peak (wl/val)',
            '{:.1f}nm / {:.0f}'.format(m_abs_wl, m_abs_val),
            p_abs_str))
        lines.append('| {:<28s} | {:<21s} | {:<21s} |'.format(
            'Unpol ext peak (wl/val)',
            '{:.1f}nm / {:.0f}'.format(m_ext_wl, m_ext_val),
            p_ext_str))

    # Pol1 (longitudinal) peak
    m_pol1 = matlab_data.get('spectra_pol1')
    p_pol1 = python_data.get('spectra_pol1')
    if m_pol1 is not None:
        m_wl, m_val = find_peak(m_pol1.wavelength, m_pol1.extinction)
        p_str = 'N/A'
        if p_pol1 is not None:
            p_wl, p_val = find_peak(p_pol1.wavelength, p_pol1.extinction)
            p_str = '{:.1f}nm / {:.0f}'.format(p_wl, p_val)
        lines.append('| {:<28s} | {:<21s} | {:<21s} |'.format(
            'Pol1 ext peak (long.)',
            '{:.1f}nm / {:.0f}'.format(m_wl, m_val),
            p_str))

    # Pol2 (transverse) peak
    m_pol2 = matlab_data.get('spectra_pol2')
    p_pol2 = python_data.get('spectra_pol2')
    if m_pol2 is not None:
        m_wl, m_val = find_peak(m_pol2.wavelength, m_pol2.extinction)
        p_str = 'N/A'
        if p_pol2 is not None:
            p_wl, p_val = find_peak(p_pol2.wavelength, p_pol2.extinction)
            p_str = '{:.1f}nm / {:.0f}'.format(p_wl, p_val)
        lines.append('| {:<28s} | {:<21s} | {:<21s} |'.format(
            'Pol2 ext peak (trans.)',
            '{:.1f}nm / {:.0f}'.format(m_wl, m_val),
            p_str))

    # Field enhancement
    m_analysis = matlab_data.get('field_analysis')
    p_analysis = python_data.get('field_analysis')
    if m_analysis is not None:
        for field_entry in m_analysis.get('fields', []):
            pol_idx = field_entry['polarization_index']
            m_max = field_entry['analysis']['enhancement_stats']['max']
            p_str = 'N/A'
            if p_analysis is not None:
                for pf in p_analysis.get('fields', []):
                    if pf['polarization_index'] == pol_idx:
                        p_str = '{:.4f}'.format(pf['analysis']['enhancement_stats']['max'])
                        break
            lines.append('| {:<28s} | {:<21s} | {:<21s} |'.format(
                'Max enh. pol{}'.format(pol_idx),
                '{:.4f}'.format(m_max),
                p_str))

    lines.append('+' + '-' * 30 + '+' + '-' * 23 + '+' + '-' * 23 + '+')
    return lines


# ======================================================================
# Main comparison logic
# ======================================================================

def run_comparison_for_solver(solver: str) -> List[str]:

    matlab_dir = MATLAB_PATHS[solver]
    python_dir = PYTHON_PATHS[solver]

    matlab_exists = os.path.isdir(matlab_dir)
    python_exists = os.path.isdir(python_dir)

    report_lines = []
    report_lines.append('=' * 70)
    report_lines.append('  {} SOLVER'.format(solver.upper()))
    report_lines.append('=' * 70)

    if not matlab_exists:
        report_lines.append('[warn] MATLAB results not found: {}'.format(matlab_dir))
        return report_lines

    if not python_exists:
        report_lines.append('[info] Python results not found yet: {}'.format(python_dir))
        report_lines.append('[info] Showing MATLAB results only.')
        report_lines.append('')

    # -- Load data --
    matlab_data = {}
    python_data = {}

    # Spectra
    matlab_data['spectra_unpolarized'] = load_spectra(os.path.join(matlab_dir, 'spectra_unpolarized.txt'))
    matlab_data['spectra_pol1'] = load_spectra(os.path.join(matlab_dir, 'spectra_pol1.txt'))
    matlab_data['spectra_pol2'] = load_spectra(os.path.join(matlab_dir, 'spectra_pol2.txt'))

    if python_exists:
        python_data['spectra_unpolarized'] = load_spectra(os.path.join(python_dir, 'spectra_unpolarized.txt'))
        python_data['spectra_pol1'] = load_spectra(os.path.join(python_dir, 'spectra_pol1.txt'))
        python_data['spectra_pol2'] = load_spectra(os.path.join(python_dir, 'spectra_pol2.txt'))

    # Field analysis
    matlab_data['field_analysis'] = load_field_analysis(os.path.join(matlab_dir, 'field_analysis.json'))
    if python_exists:
        python_data['field_analysis'] = load_field_analysis(os.path.join(python_dir, 'field_analysis.json'))

    # -- Spectra comparison --
    report_lines.append('')
    report_lines.append('--- Spectra ---')

    if matlab_data['spectra_unpolarized'] is not None:
        report_lines.extend(generate_spectra_metrics(
            matlab_data['spectra_unpolarized'],
            python_data.get('spectra_unpolarized'),
            'Unpolarized'))
        report_lines.append('')

    if matlab_data['spectra_pol1'] is not None:
        report_lines.extend(generate_spectra_metrics(
            matlab_data['spectra_pol1'],
            python_data.get('spectra_pol1'),
            'Pol1 (longitudinal, [1,0,0])'))
        report_lines.append('')

    if matlab_data['spectra_pol2'] is not None:
        report_lines.extend(generate_spectra_metrics(
            matlab_data['spectra_pol2'],
            python_data.get('spectra_pol2'),
            'Pol2 (transverse, [0,1,0])'))
        report_lines.append('')

    # -- Spectra plots --
    if matlab_data['spectra_unpolarized'] is not None:
        plot_spectra_comparison(
            matlab_data['spectra_unpolarized'],
            python_data.get('spectra_unpolarized'),
            matlab_data['spectra_pol1'],
            python_data.get('spectra_pol1'),
            matlab_data['spectra_pol2'],
            python_data.get('spectra_pol2'),
            solver, OUTPUT_DIR)
        print('[info] Spectra plots saved for {} solver'.format(solver))

    # -- Field comparison --
    report_lines.append('--- Field Enhancement ---')

    if matlab_data['field_analysis'] is not None:
        report_lines.extend(generate_field_metrics(
            matlab_data['field_analysis'],
            python_data.get('field_analysis')))
        report_lines.append('')

    # Field map plots
    matlab_field_files = find_field_files(matlab_dir)
    for mf_path in matlab_field_files:
        pol, wl = parse_field_filename(mf_path)
        matlab_field = load_field(mf_path)
        if matlab_field is None:
            continue

        python_field = None
        if python_exists:
            # Try to find matching python field file
            python_field_files = find_field_files(python_dir)
            for pf_path in python_field_files:
                p_pol, p_wl = parse_field_filename(pf_path)
                if p_pol == pol:
                    python_field = load_field(pf_path)
                    break

        plot_field_comparison(matlab_field, python_field, pol, wl, solver, OUTPUT_DIR)
        print('[info] Field plot saved: {} {} ({})'.format(pol, wl, solver))

    # -- Comparison table --
    report_lines.append('--- Summary Table ---')
    report_lines.extend(generate_comparison_table(matlab_data, python_data, solver))
    report_lines.append('')

    return report_lines


def main() -> None:

    os.makedirs(OUTPUT_DIR, exist_ok = True)

    print('[info] Rod comparison: MATLAB (mnpbem) vs Python (pymnpbem)')
    print('[info] Output directory: {}'.format(OUTPUT_DIR))
    print('')

    all_report_lines = []
    all_report_lines.append('=' * 70)
    all_report_lines.append('  ROD (AUNR 22x47) COMPARISON REPORT')
    all_report_lines.append('  MATLAB mnpbem vs Python pymnpbem')
    all_report_lines.append('=' * 70)
    all_report_lines.append('')

    for solver in SOLVERS:
        print('[info] Processing {} solver...'.format(solver))
        solver_lines = run_comparison_for_solver(solver)
        all_report_lines.extend(solver_lines)
        all_report_lines.append('')

    # -- Write summary --
    summary_path = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write('\n'.join(all_report_lines))
        fh.write('\n')

    print('')
    print('[info] Comparison summary saved: {}'.format(summary_path))
    print('')

    # Print to stdout as well
    for line in all_report_lines:
        print(line)


if __name__ == '__main__':
    main()
