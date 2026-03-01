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


# ==============================================================================
# Constants
# ==============================================================================

MATLAB_STAT_DIR = os.path.expanduser('~/temporary/mat2py/sphere/stat/sphere_30nm')
MATLAB_RET_DIR = os.path.expanduser('~/temporary/mat2py/sphere/ret/sphere_30nm')
PYTHON_STAT_DIR = os.path.expanduser('~/scratch/mat2py/sphere/stat/sphere_30nm/sphere_30nm')
PYTHON_RET_DIR = os.path.expanduser('~/scratch/mat2py/sphere/ret/sphere_30nm/sphere_30nm')
OUTPUT_DIR = os.path.expanduser('~/scratch/mat2py/sphere/comparison')


# ==============================================================================
# Data loading utilities
# ==============================================================================

def load_spectra(file_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(file_path):
        return None

    wavelength = []
    scattering = []
    extinction = []
    absorption = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split('\t')
            wavelength.append(float(parts[0]))
            scattering.append(float(parts[1]))
            extinction.append(float(parts[2]))
            absorption.append(float(parts[3]))

    return {
        'wavelength': np.array(wavelength),
        'scattering': np.array(scattering),
        'extinction': np.array(extinction),
        'absorption': np.array(absorption),
    }


def load_field(file_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(file_path):
        return None

    x_vals = []
    y_vals = []
    z_vals = []
    enhancement = []
    intensity = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split('\t')
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))
            z_vals.append(float(parts[2]))
            enhancement.append(float(parts[3]))
            intensity.append(float(parts[4]))

    return {
        'x': np.array(x_vals),
        'y': np.array(y_vals),
        'z': np.array(z_vals),
        'enhancement': np.array(enhancement),
        'intensity': np.array(intensity),
    }


def load_field_analysis(file_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        return json.load(f)


def load_analysis_summary(file_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return None

    result = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Peak wavelength:' in line:
                result['peak_wavelength'] = float(line.split(':')[1].strip().replace(' nm', ''))
            elif 'Peak absorption:' in line:
                result['peak_absorption'] = float(line.split(':')[1].strip().replace(' nm^2', ''))
            elif 'Peak extinction:' in line:
                result['peak_extinction'] = float(line.split(':')[1].strip().replace(' nm^2', ''))
            elif 'Peak scattering:' in line:
                result['peak_scattering'] = float(line.split(':')[1].strip().replace(' nm^2', ''))
            elif 'max_enhancement=' in line:
                # e.g., "  Field 1: pol1, lambda=505.1nm, max_enhancement=16.48..."
                parts = line.split('max_enhancement=')
                if len(parts) == 2:
                    key = 'max_enhancement_pol1' if 'pol1' in line else 'max_enhancement_pol2'
                    result[key] = float(parts[1])
    return result


def find_field_files(directory: str) -> Dict[str, str]:
    pattern = os.path.join(directory, 'field_*nm.txt')
    files = glob.glob(pattern)
    result = {}
    for fp in sorted(files):
        basename = os.path.basename(fp)
        # e.g., field_pol1_505nm.txt, field_pol2_505nm.txt, field_unpolarized_505nm.txt
        name_no_ext = basename.replace('.txt', '')
        result[name_no_ext] = fp
    return result


# ==============================================================================
# Metric computation
# ==============================================================================

def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_r_squared(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_peak_info(wavelength: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    idx = np.argmax(values)
    return float(wavelength[idx]), float(values[idx])


# ==============================================================================
# Plotting: spectra comparison
# ==============================================================================

def plot_spectra_comparison(
        matlab_spectra: Dict[str, np.ndarray],
        python_spectra: Optional[Dict[str, np.ndarray]],
        solver_label: str,
        output_dir: str) -> None:

    properties = ['absorption', 'scattering', 'extinction']

    if python_spectra is not None:
        # Both MATLAB and Python available: overlay plot
        fig, axes = plt.subplots(1, 3, figsize = (18, 5))
        fig.suptitle('{} - Unpolarized Spectra: MATLAB vs Python'.format(solver_label), fontsize = 14)

        for ax, prop in zip(axes, properties):
            wl_m = matlab_spectra['wavelength']
            val_m = matlab_spectra[prop]
            wl_p = python_spectra['wavelength']
            val_p = python_spectra[prop]

            ax.plot(wl_m, val_m, 'b-', linewidth = 1.5, label = 'MATLAB')
            ax.plot(wl_p, val_p, 'r--', linewidth = 1.5, label = 'Python')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('{} (nm^2)'.format(prop.capitalize()))
            ax.set_title(prop.capitalize())
            ax.legend()
            ax.grid(True, alpha = 0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_spectra_comparison.png'.format(solver_label.lower())), dpi = 150)
        plt.close(fig)

        # Difference plot
        fig, axes = plt.subplots(1, 3, figsize = (18, 5))
        fig.suptitle('{} - Spectra Difference (Python - MATLAB)'.format(solver_label), fontsize = 14)

        # Interpolate Python onto MATLAB wavelength grid if needed
        for ax, prop in zip(axes, properties):
            wl_m = matlab_spectra['wavelength']
            val_m = matlab_spectra[prop]
            wl_p = python_spectra['wavelength']
            val_p = python_spectra[prop]

            if len(wl_m) == len(wl_p) and np.allclose(wl_m, wl_p, atol = 0.01):
                diff = val_p - val_m
                rel_diff = np.where(np.abs(val_m) > 1e-10, (val_p - val_m) / val_m * 100.0, 0.0)
            else:
                val_p_interp = np.interp(wl_m, wl_p, val_p)
                diff = val_p_interp - val_m
                rel_diff = np.where(np.abs(val_m) > 1e-10, diff / val_m * 100.0, 0.0)

            ax_abs = ax
            ax_abs.plot(wl_m, diff, 'k-', linewidth = 1.0)
            ax_abs.set_xlabel('Wavelength (nm)')
            ax_abs.set_ylabel('Abs. Diff (nm^2)')
            ax_abs.set_title('{} Difference'.format(prop.capitalize()))
            ax_abs.grid(True, alpha = 0.3)
            ax_abs.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)

            ax_rel = ax_abs.twinx()
            ax_rel.plot(wl_m, rel_diff, 'r-', linewidth = 0.8, alpha = 0.6)
            ax_rel.set_ylabel('Rel. Diff (%)', color = 'r')
            ax_rel.tick_params(axis = 'y', labelcolor = 'r')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_spectra_difference.png'.format(solver_label.lower())), dpi = 150)
        plt.close(fig)

    else:
        # MATLAB only
        fig, axes = plt.subplots(1, 3, figsize = (18, 5))
        fig.suptitle('{} - Unpolarized Spectra (MATLAB only)'.format(solver_label), fontsize = 14)

        for ax, prop in zip(axes, properties):
            wl_m = matlab_spectra['wavelength']
            val_m = matlab_spectra[prop]
            ax.plot(wl_m, val_m, 'b-', linewidth = 1.5, label = 'MATLAB')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('{} (nm^2)'.format(prop.capitalize()))
            ax.set_title(prop.capitalize())
            ax.legend()
            ax.grid(True, alpha = 0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_spectra_matlab_only.png'.format(solver_label.lower())), dpi = 150)
        plt.close(fig)


# ==============================================================================
# Plotting: field comparison
# ==============================================================================

def reshape_field_to_2d(field_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = field_data['x']
    z = field_data['z']
    enh = field_data['enhancement']

    x_unique = np.unique(x)
    z_unique = np.unique(z)
    nx = len(x_unique)
    nz = len(z_unique)

    enh_2d = enh.reshape(nz, nx)
    return x_unique, z_unique, enh_2d


def plot_field_comparison(
        matlab_field: Dict[str, np.ndarray],
        python_field: Optional[Dict[str, np.ndarray]],
        label: str,
        solver_label: str,
        output_dir: str) -> None:

    x_m, z_m, enh_m = reshape_field_to_2d(matlab_field)

    if python_field is not None:
        x_p, z_p, enh_p = reshape_field_to_2d(python_field)

        fig, axes = plt.subplots(1, 3, figsize = (20, 5.5))
        fig.suptitle('{} - {} Field Enhancement (|E|/|E0|)'.format(solver_label, label), fontsize = 14)

        # MATLAB
        vmin = min(enh_m.min(), enh_p.min())
        vmax = max(enh_m.max(), enh_p.max())

        im0 = axes[0].pcolormesh(x_m, z_m, enh_m, shading = 'auto', cmap = 'hot', vmin = vmin, vmax = vmax)
        axes[0].set_title('MATLAB')
        axes[0].set_xlabel('x (nm)')
        axes[0].set_ylabel('z (nm)')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax = axes[0], shrink = 0.8)

        # Python
        im1 = axes[1].pcolormesh(x_p, z_p, enh_p, shading = 'auto', cmap = 'hot', vmin = vmin, vmax = vmax)
        axes[1].set_title('Python')
        axes[1].set_xlabel('x (nm)')
        axes[1].set_ylabel('z (nm)')
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax = axes[1], shrink = 0.8)

        # Difference
        if enh_m.shape == enh_p.shape:
            diff = enh_p - enh_m
        else:
            diff = enh_p - enh_m[:enh_p.shape[0], :enh_p.shape[1]]

        abs_max_diff = max(abs(diff.min()), abs(diff.max()))
        if abs_max_diff < 1e-15:
            abs_max_diff = 1.0

        im2 = axes[2].pcolormesh(x_m, z_m, diff, shading = 'auto', cmap = 'RdBu_r', vmin = -abs_max_diff, vmax = abs_max_diff)
        axes[2].set_title('Difference (Python - MATLAB)')
        axes[2].set_xlabel('x (nm)')
        axes[2].set_ylabel('z (nm)')
        axes[2].set_aspect('equal')
        plt.colorbar(im2, ax = axes[2], shrink = 0.8)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_{}_field_comparison.png'.format(solver_label.lower(), label)), dpi = 150)
        plt.close(fig)

    else:
        # MATLAB only
        fig, ax = plt.subplots(1, 1, figsize = (7, 5.5))
        fig.suptitle('{} - {} Field Enhancement (MATLAB only)'.format(solver_label, label), fontsize = 14)

        im = ax.pcolormesh(x_m, z_m, enh_m, shading = 'auto', cmap = 'hot')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('z (nm)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax = ax, shrink = 0.8)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_{}_field_matlab_only.png'.format(solver_label.lower(), label)), dpi = 150)
        plt.close(fig)


# ==============================================================================
# Summary report generation
# ==============================================================================

class SphereComparisonReport(object):

    def __init__(self) -> None:
        self.lines = []

    def add_header(self, text: str) -> None:
        self.lines.append('=' * 72)
        self.lines.append(text)
        self.lines.append('=' * 72)
        self.lines.append('')

    def add_section(self, text: str) -> None:
        self.lines.append('--- {} ---'.format(text))

    def add_line(self, text: str) -> None:
        self.lines.append(text)

    def add_blank(self) -> None:
        self.lines.append('')

    def add_table_row(self, metric: str, matlab_val: str, python_val: str, diff_val: str) -> None:
        self.lines.append('  {:<30s} {:>15s} {:>15s} {:>15s}'.format(metric, matlab_val, python_val, diff_val))

    def add_table_header(self) -> None:
        self.lines.append('  {:<30s} {:>15s} {:>15s} {:>15s}'.format('Metric', 'MATLAB', 'Python', 'Diff (%)'))
        self.lines.append('  ' + '-' * 75)

    def get_text(self) -> str:
        return '\n'.join(self.lines)

    def save(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(self.get_text())


def format_value(val: Optional[float], fmt: str = '{:.4f}') -> str:
    if val is None:
        return 'N/A'
    return fmt.format(val)


def format_diff_pct(matlab_val: Optional[float], python_val: Optional[float]) -> str:
    if matlab_val is None or python_val is None:
        return 'N/A'
    if abs(matlab_val) < 1e-15:
        return 'N/A'
    pct = (python_val - matlab_val) / matlab_val * 100.0
    return '{:+.4f}%'.format(pct)


# ==============================================================================
# Compare one solver (stat or ret)
# ==============================================================================

def compare_single_solver(
        solver_name: str,
        matlab_dir: str,
        python_dir: str,
        output_dir: str,
        report: SphereComparisonReport) -> None:

    matlab_exists = os.path.isdir(matlab_dir)
    python_exists = os.path.isdir(python_dir)

    report.add_section('{} Solver'.format(solver_name.upper()))
    report.add_line('  MATLAB dir: {}'.format(matlab_dir))
    report.add_line('  Python dir: {}'.format(python_dir))
    report.add_line('  MATLAB available: {}'.format(matlab_exists))
    report.add_line('  Python available: {}'.format(python_exists))
    report.add_blank()

    if not matlab_exists:
        report.add_line('  [warn] MATLAB results not found. Skipping {}.\n'.format(solver_name))
        return

    # --- Load MATLAB data ---
    matlab_spectra_unpol = load_spectra(os.path.join(matlab_dir, 'spectra_unpolarized.txt'))
    matlab_spectra_pol1 = load_spectra(os.path.join(matlab_dir, 'spectra_pol1.txt'))
    matlab_spectra_pol2 = load_spectra(os.path.join(matlab_dir, 'spectra_pol2.txt'))
    matlab_field_analysis = load_field_analysis(os.path.join(matlab_dir, 'field_analysis.json'))
    matlab_summary = load_analysis_summary(os.path.join(matlab_dir, 'analysis_summary.txt'))
    matlab_field_files = find_field_files(matlab_dir)

    # --- Load Python data (if available) ---
    python_spectra_unpol = None
    python_spectra_pol1 = None
    python_spectra_pol2 = None
    python_field_analysis = None
    python_summary = None
    python_field_files = {}

    if python_exists:
        python_spectra_unpol = load_spectra(os.path.join(python_dir, 'spectra_unpolarized.txt'))
        python_spectra_pol1 = load_spectra(os.path.join(python_dir, 'spectra_pol1.txt'))
        python_spectra_pol2 = load_spectra(os.path.join(python_dir, 'spectra_pol2.txt'))
        python_field_analysis = load_field_analysis(os.path.join(python_dir, 'field_analysis.json'))
        python_summary = load_analysis_summary(os.path.join(python_dir, 'analysis_summary.txt'))
        python_field_files = find_field_files(python_dir)

    # --- Spectra comparison ---
    report.add_section('{} - Spectra Comparison'.format(solver_name.upper()))

    if matlab_spectra_unpol is not None:
        m_peak_wl_abs, m_peak_val_abs = compute_peak_info(matlab_spectra_unpol['wavelength'], matlab_spectra_unpol['absorption'])
        m_peak_wl_ext, m_peak_val_ext = compute_peak_info(matlab_spectra_unpol['wavelength'], matlab_spectra_unpol['extinction'])
        m_peak_wl_sca, m_peak_val_sca = compute_peak_info(matlab_spectra_unpol['wavelength'], matlab_spectra_unpol['scattering'])

        p_peak_wl_abs = p_peak_val_abs = None
        p_peak_wl_ext = p_peak_val_ext = None
        p_peak_wl_sca = p_peak_val_sca = None

        rmse_abs = rmse_ext = rmse_sca = None
        r2_abs = r2_ext = r2_sca = None

        if python_spectra_unpol is not None:
            p_peak_wl_abs, p_peak_val_abs = compute_peak_info(python_spectra_unpol['wavelength'], python_spectra_unpol['absorption'])
            p_peak_wl_ext, p_peak_val_ext = compute_peak_info(python_spectra_unpol['wavelength'], python_spectra_unpol['extinction'])
            p_peak_wl_sca, p_peak_val_sca = compute_peak_info(python_spectra_unpol['wavelength'], python_spectra_unpol['scattering'])

            wl_m = matlab_spectra_unpol['wavelength']
            wl_p = python_spectra_unpol['wavelength']

            if len(wl_m) == len(wl_p) and np.allclose(wl_m, wl_p, atol = 0.01):
                rmse_abs = compute_rmse(matlab_spectra_unpol['absorption'], python_spectra_unpol['absorption'])
                rmse_ext = compute_rmse(matlab_spectra_unpol['extinction'], python_spectra_unpol['extinction'])
                rmse_sca = compute_rmse(matlab_spectra_unpol['scattering'], python_spectra_unpol['scattering'])
                r2_abs = compute_r_squared(matlab_spectra_unpol['absorption'], python_spectra_unpol['absorption'])
                r2_ext = compute_r_squared(matlab_spectra_unpol['extinction'], python_spectra_unpol['extinction'])
                r2_sca = compute_r_squared(matlab_spectra_unpol['scattering'], python_spectra_unpol['scattering'])
            else:
                # Interpolate Python onto MATLAB wavelength grid
                abs_p_interp = np.interp(wl_m, wl_p, python_spectra_unpol['absorption'])
                ext_p_interp = np.interp(wl_m, wl_p, python_spectra_unpol['extinction'])
                sca_p_interp = np.interp(wl_m, wl_p, python_spectra_unpol['scattering'])
                rmse_abs = compute_rmse(matlab_spectra_unpol['absorption'], abs_p_interp)
                rmse_ext = compute_rmse(matlab_spectra_unpol['extinction'], ext_p_interp)
                rmse_sca = compute_rmse(matlab_spectra_unpol['scattering'], sca_p_interp)
                r2_abs = compute_r_squared(matlab_spectra_unpol['absorption'], abs_p_interp)
                r2_ext = compute_r_squared(matlab_spectra_unpol['extinction'], ext_p_interp)
                r2_sca = compute_r_squared(matlab_spectra_unpol['scattering'], sca_p_interp)

        report.add_blank()
        report.add_table_header()
        report.add_table_row('Peak Abs. Wavelength (nm)', format_value(m_peak_wl_abs, '{:.2f}'), format_value(p_peak_wl_abs, '{:.2f}'), format_diff_pct(m_peak_wl_abs, p_peak_wl_abs))
        report.add_table_row('Peak Abs. Value (nm^2)', format_value(m_peak_val_abs, '{:.4f}'), format_value(p_peak_val_abs, '{:.4f}'), format_diff_pct(m_peak_val_abs, p_peak_val_abs))
        report.add_table_row('Peak Ext. Wavelength (nm)', format_value(m_peak_wl_ext, '{:.2f}'), format_value(p_peak_wl_ext, '{:.2f}'), format_diff_pct(m_peak_wl_ext, p_peak_wl_ext))
        report.add_table_row('Peak Ext. Value (nm^2)', format_value(m_peak_val_ext, '{:.4f}'), format_value(p_peak_val_ext, '{:.4f}'), format_diff_pct(m_peak_val_ext, p_peak_val_ext))
        report.add_table_row('Peak Sca. Wavelength (nm)', format_value(m_peak_wl_sca, '{:.2f}'), format_value(p_peak_wl_sca, '{:.2f}'), format_diff_pct(m_peak_wl_sca, p_peak_wl_sca))
        report.add_table_row('Peak Sca. Value (nm^2)', format_value(m_peak_val_sca, '{:.4f}'), format_value(p_peak_val_sca, '{:.4f}'), format_diff_pct(m_peak_val_sca, p_peak_val_sca))
        report.add_table_row('Abs. RMSE', '', '', format_value(rmse_abs, '{:.6e}'))
        report.add_table_row('Ext. RMSE', '', '', format_value(rmse_ext, '{:.6e}'))
        report.add_table_row('Sca. RMSE', '', '', format_value(rmse_sca, '{:.6e}'))
        report.add_table_row('Abs. R^2', '', '', format_value(r2_abs, '{:.8f}'))
        report.add_table_row('Ext. R^2', '', '', format_value(r2_ext, '{:.8f}'))
        report.add_table_row('Sca. R^2', '', '', format_value(r2_sca, '{:.8f}'))
        report.add_blank()

        # Generate spectra plot
        plot_spectra_comparison(matlab_spectra_unpol, python_spectra_unpol, solver_name.upper(), output_dir)

    # --- Field comparison ---
    report.add_section('{} - Field Comparison'.format(solver_name.upper()))
    report.add_blank()

    # Extract max enhancement from field_analysis.json
    m_max_enh_per_pol = {}
    if matlab_field_analysis is not None:
        for field_entry in matlab_field_analysis.get('fields', []):
            pol_idx = field_entry['polarization_index']
            stats = field_entry.get('analysis', {}).get('enhancement_stats', {})
            m_max_enh_per_pol[pol_idx] = stats.get('max', None)
            hotspots = field_entry.get('analysis', {}).get('hotspots', [])

            pol_label = 'pol{}'.format(pol_idx)
            report.add_line('  MATLAB {} - Max Enhancement: {}'.format(pol_label, format_value(stats.get('max'), '{:.4f}')))
            if len(hotspots) > 0:
                top = hotspots[0]
                report.add_line('    Top hotspot: position={}, enhancement={}'.format(
                    top.get('position', []), format_value(top.get('enhancement'), '{:.4f}')))
        report.add_blank()

    p_max_enh_per_pol = {}
    if python_field_analysis is not None:
        for field_entry in python_field_analysis.get('fields', []):
            pol_idx = field_entry['polarization_index']
            stats = field_entry.get('analysis', {}).get('enhancement_stats', {})
            p_max_enh_per_pol[pol_idx] = stats.get('max', None)
            hotspots = field_entry.get('analysis', {}).get('hotspots', [])

            pol_label = 'pol{}'.format(pol_idx)
            report.add_line('  Python {} - Max Enhancement: {}'.format(pol_label, format_value(stats.get('max'), '{:.4f}')))
            if len(hotspots) > 0:
                top = hotspots[0]
                report.add_line('    Top hotspot: position={}, enhancement={}'.format(
                    top.get('position', []), format_value(top.get('enhancement'), '{:.4f}')))
        report.add_blank()

    # Field enhancement comparison table
    if len(m_max_enh_per_pol) > 0:
        report.add_table_header()
        for pol_idx in sorted(m_max_enh_per_pol.keys()):
            m_val = m_max_enh_per_pol.get(pol_idx)
            p_val = p_max_enh_per_pol.get(pol_idx)
            report.add_table_row(
                'Max Enh. pol{}'.format(pol_idx),
                format_value(m_val, '{:.4f}'),
                format_value(p_val, '{:.4f}'),
                format_diff_pct(m_val, p_val))
        report.add_blank()

    # Hotspot position comparison
    if matlab_field_analysis is not None and python_field_analysis is not None:
        report.add_line('  Hotspot Position Comparison:')
        m_fields = matlab_field_analysis.get('fields', [])
        p_fields = python_field_analysis.get('fields', [])

        for m_entry, p_entry in zip(m_fields, p_fields):
            pol_idx = m_entry['polarization_index']
            m_hotspots = m_entry.get('analysis', {}).get('hotspots', [])
            p_hotspots = p_entry.get('analysis', {}).get('hotspots', [])

            if len(m_hotspots) > 0 and len(p_hotspots) > 0:
                m_pos = np.array(m_hotspots[0]['position'])
                p_pos = np.array(p_hotspots[0]['position'])
                dist = float(np.linalg.norm(m_pos - p_pos))
                report.add_line('    pol{}: MATLAB top={}, Python top={}, distance={:.2f} nm'.format(
                    pol_idx, m_hotspots[0]['position'], p_hotspots[0]['position'], dist))
        report.add_blank()

    # Plot field maps
    for field_name, matlab_path in sorted(matlab_field_files.items()):
        matlab_field = load_field(matlab_path)
        if matlab_field is None:
            continue

        # Try to find matching Python field file
        python_field = None
        if python_exists:
            # Match by field type (pol1, pol2, unpolarized)
            for pf_name, pf_path in python_field_files.items():
                # Match if they share the same polarization/type prefix
                m_type = _extract_field_type(field_name)
                p_type = _extract_field_type(pf_name)
                if m_type == p_type:
                    python_field = load_field(pf_path)
                    break

        plot_field_comparison(matlab_field, python_field, field_name, solver_name.upper(), output_dir)

    report.add_blank()


def _extract_field_type(field_name: str) -> str:
    # field_pol1_505nm -> pol1
    # field_pol2_505nm -> pol2
    # field_unpolarized_505nm -> unpolarized
    parts = field_name.replace('field_', '').split('_')
    # parts could be: ['pol1', '505nm'] or ['unpolarized', '505nm']
    return parts[0]


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    report = SphereComparisonReport()
    report.add_header('SPHERE 30nm - MATLAB (mnpbem) vs Python (pymnpbem) Comparison')
    report.add_line('Generated by compare_sphere.py')
    report.add_blank()

    # Check availability
    report.add_section('DATA AVAILABILITY')
    report.add_line('  MATLAB stat: {}'.format('YES' if os.path.isdir(MATLAB_STAT_DIR) else 'NO'))
    report.add_line('  MATLAB ret:  {}'.format('YES' if os.path.isdir(MATLAB_RET_DIR) else 'NO'))
    report.add_line('  Python stat: {}'.format('YES' if os.path.isdir(PYTHON_STAT_DIR) else 'NO'))
    report.add_line('  Python ret:  {}'.format('YES' if os.path.isdir(PYTHON_RET_DIR) else 'NO'))
    report.add_blank()

    has_python = os.path.isdir(PYTHON_STAT_DIR) or os.path.isdir(PYTHON_RET_DIR)
    if has_python:
        report.add_line('  Mode: MATLAB vs Python comparison')
    else:
        report.add_line('  Mode: MATLAB results summary only (Python results not yet available)')
    report.add_blank()

    # Compare stat solver
    print('[info] Comparing STAT solver...')
    compare_single_solver('stat', MATLAB_STAT_DIR, PYTHON_STAT_DIR, OUTPUT_DIR, report)

    # Compare ret solver
    print('[info] Comparing RET solver...')
    compare_single_solver('ret', MATLAB_RET_DIR, PYTHON_RET_DIR, OUTPUT_DIR, report)

    # --- Stat vs Ret cross-comparison (MATLAB only) ---
    report.add_section('STAT vs RET Cross-Comparison (MATLAB)')
    report.add_blank()

    matlab_stat_unpol = load_spectra(os.path.join(MATLAB_STAT_DIR, 'spectra_unpolarized.txt'))
    matlab_ret_unpol = load_spectra(os.path.join(MATLAB_RET_DIR, 'spectra_unpolarized.txt'))

    if matlab_stat_unpol is not None and matlab_ret_unpol is not None:
        # Plot stat vs ret overlay
        properties = ['absorption', 'scattering', 'extinction']
        fig, axes = plt.subplots(1, 3, figsize = (18, 5))
        fig.suptitle('MATLAB Sphere 30nm: STAT vs RET', fontsize = 14)

        for ax, prop in zip(axes, properties):
            wl_s = matlab_stat_unpol['wavelength']
            val_s = matlab_stat_unpol[prop]
            wl_r = matlab_ret_unpol['wavelength']
            val_r = matlab_ret_unpol[prop]

            ax.plot(wl_s, val_s, 'b-', linewidth = 1.5, label = 'STAT')
            ax.plot(wl_r, val_r, 'r--', linewidth = 1.5, label = 'RET')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('{} (nm^2)'.format(prop.capitalize()))
            ax.set_title(prop.capitalize())
            ax.legend()
            ax.grid(True, alpha = 0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'matlab_stat_vs_ret_spectra.png'), dpi = 150)
        plt.close(fig)

        # Summary table for stat vs ret
        s_peak_wl_abs, s_peak_val_abs = compute_peak_info(matlab_stat_unpol['wavelength'], matlab_stat_unpol['absorption'])
        r_peak_wl_abs, r_peak_val_abs = compute_peak_info(matlab_ret_unpol['wavelength'], matlab_ret_unpol['absorption'])
        s_peak_wl_ext, s_peak_val_ext = compute_peak_info(matlab_stat_unpol['wavelength'], matlab_stat_unpol['extinction'])
        r_peak_wl_ext, r_peak_val_ext = compute_peak_info(matlab_ret_unpol['wavelength'], matlab_ret_unpol['extinction'])

        report.add_line('  {:<30s} {:>15s} {:>15s} {:>15s}'.format('Metric', 'STAT', 'RET', 'Diff (%)'))
        report.add_line('  ' + '-' * 75)
        report.add_table_row('Peak Abs. Wavelength (nm)', format_value(s_peak_wl_abs, '{:.2f}'), format_value(r_peak_wl_abs, '{:.2f}'), format_diff_pct(s_peak_wl_abs, r_peak_wl_abs))
        report.add_table_row('Peak Abs. Value (nm^2)', format_value(s_peak_val_abs, '{:.4f}'), format_value(r_peak_val_abs, '{:.4f}'), format_diff_pct(s_peak_val_abs, r_peak_val_abs))
        report.add_table_row('Peak Ext. Wavelength (nm)', format_value(s_peak_wl_ext, '{:.2f}'), format_value(r_peak_wl_ext, '{:.2f}'), format_diff_pct(s_peak_wl_ext, r_peak_wl_ext))
        report.add_table_row('Peak Ext. Value (nm^2)', format_value(s_peak_val_ext, '{:.4f}'), format_value(r_peak_val_ext, '{:.4f}'), format_diff_pct(s_peak_val_ext, r_peak_val_ext))
        report.add_blank()

        # Field enhancement comparison
        stat_fa = load_field_analysis(os.path.join(MATLAB_STAT_DIR, 'field_analysis.json'))
        ret_fa = load_field_analysis(os.path.join(MATLAB_RET_DIR, 'field_analysis.json'))

        if stat_fa is not None and ret_fa is not None:
            for s_field, r_field in zip(stat_fa.get('fields', []), ret_fa.get('fields', [])):
                pol_idx = s_field['polarization_index']
                s_max = s_field.get('analysis', {}).get('enhancement_stats', {}).get('max')
                r_max = r_field.get('analysis', {}).get('enhancement_stats', {}).get('max')
                report.add_table_row(
                    'Max Enh. pol{}'.format(pol_idx),
                    format_value(s_max, '{:.4f}'),
                    format_value(r_max, '{:.4f}'),
                    format_diff_pct(s_max, r_max))

    report.add_blank()

    # Save report
    summary_path = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')
    report.save(summary_path)
    print('[info] Summary saved to: {}'.format(summary_path))
    print('[info] Plots saved to: {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
