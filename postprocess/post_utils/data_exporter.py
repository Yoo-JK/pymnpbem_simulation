"""
Data Exporter for pyMNPBEM simulation results.

Exports data to various formats:
- TXT (tab-separated)
- CSV (comma-separated)
- JSON (with metadata)
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional, List


class DataExporter:
    """
    Exports simulation results to various file formats.

    Supports:
    - Spectrum data (TXT, CSV)
    - Field data (TXT, JSON)
    - Surface charge data (JSON)
    - Summary reports (JSON)
    """

    def __init__(self, output_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data exporter.

        Args:
            output_dir: Directory to save exported files
            config: Configuration dictionary
        """
        self.output_dir = output_dir
        self.config = config or {}

        os.makedirs(output_dir, exist_ok=True)

    def export_spectrum_txt(self, spectrum_data: Dict[str, np.ndarray],
                            pol_idx: int = 0,
                            filename: Optional[str] = None):
        """
        Export spectrum data to TXT file.

        Args:
            spectrum_data: Dictionary with wavelengths and cross-sections
            pol_idx: Polarization index
            filename: Output filename (without extension)
        """
        if filename is None:
            filename = f'spectrum_pol{pol_idx + 1}'

        filepath = os.path.join(self.output_dir, f'{filename}.txt')

        wavelengths = spectrum_data['wavelengths']

        # Handle both 1D and 2D arrays
        def get_column(data, idx):
            if data is None:
                return np.zeros_like(wavelengths)
            return data[:, idx] if data.ndim > 1 else data

        scattering = get_column(spectrum_data.get('scattering'), pol_idx)
        absorption = get_column(spectrum_data.get('absorption'), pol_idx)
        extinction = get_column(spectrum_data.get('extinction'), pol_idx)

        with open(filepath, 'w') as f:
            f.write("# Optical Cross-Section Spectrum\n")
            f.write("# Wavelength(nm)\tScattering(nm^2)\tAbsorption(nm^2)\tExtinction(nm^2)\n")

            for i in range(len(wavelengths)):
                f.write(f"{wavelengths[i]:.2f}\t{scattering[i]:.6e}\t"
                        f"{absorption[i]:.6e}\t{extinction[i]:.6e}\n")

    def export_spectrum_csv(self, spectrum_data: Dict[str, np.ndarray],
                            pol_idx: int = 0,
                            filename: Optional[str] = None):
        """
        Export spectrum data to CSV file.

        Args:
            spectrum_data: Dictionary with wavelengths and cross-sections
            pol_idx: Polarization index
            filename: Output filename
        """
        if filename is None:
            filename = f'spectrum_pol{pol_idx + 1}'

        filepath = os.path.join(self.output_dir, f'{filename}.csv')

        wavelengths = spectrum_data['wavelengths']

        # Handle both 1D and 2D arrays
        def get_column(data, idx):
            if data is None:
                return np.zeros_like(wavelengths)
            return data[:, idx] if data.ndim > 1 else data

        scattering = get_column(spectrum_data.get('scattering'), pol_idx)
        absorption = get_column(spectrum_data.get('absorption'), pol_idx)
        extinction = get_column(spectrum_data.get('extinction'), pol_idx)

        with open(filepath, 'w') as f:
            f.write("Wavelength_nm,Scattering_nm2,Absorption_nm2,Extinction_nm2\n")

            for i in range(len(wavelengths)):
                f.write(f"{wavelengths[i]:.2f},{scattering[i]:.6e},"
                        f"{absorption[i]:.6e},{extinction[i]:.6e}\n")

    def export_spectrum_unpolarized(self, unpolarized_data: Dict[str, np.ndarray],
                                     filename: str = 'spectrum_unpolarized'):
        """
        Export unpolarized spectrum data.

        Args:
            unpolarized_data: Dictionary with unpolarized spectra
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, f'{filename}.txt')

        wavelengths = unpolarized_data['wavelengths']

        with open(filepath, 'w') as f:
            f.write("# Unpolarized Optical Cross-Section Spectrum\n")
            f.write("# Wavelength(nm)\tScattering(nm^2)\tAbsorption(nm^2)\tExtinction(nm^2)\n")

            for i in range(len(wavelengths)):
                sca = unpolarized_data.get('scattering_unpolarized', np.zeros_like(wavelengths))[i]
                abs_ = unpolarized_data.get('absorption_unpolarized', np.zeros_like(wavelengths))[i]
                ext = unpolarized_data.get('extinction_unpolarized', np.zeros_like(wavelengths))[i]
                f.write(f"{wavelengths[i]:.2f}\t{sca:.6e}\t{abs_:.6e}\t{ext:.6e}\n")

    def export_field_txt(self, field_data: Dict[str, np.ndarray],
                         pol_idx: int = 0,
                         filename: Optional[str] = None):
        """
        Export 2D field data to TXT file.

        Args:
            field_data: Dictionary with field enhancement data
            pol_idx: Polarization index
            filename: Output filename
        """
        if filename is None:
            filename = f'field_pol{pol_idx + 1}'

        filepath = os.path.join(self.output_dir, f'{filename}.txt')

        enhancement = field_data.get('enhancement')
        x = field_data.get('x', np.array([]))
        y = field_data.get('y', np.array([]))
        z_arr = field_data.get('z', np.array([]))
        z = z_arr if len(y) <= 1 else y

        if enhancement is None or len(x) == 0 or len(z) == 0:
            return  # Skip export if data is missing

        with open(filepath, 'w') as f:
            f.write(f"# Field Enhancement |E|^2/|E0|^2\n")
            f.write(f"# Grid size: {len(x)} x {len(z)}\n")
            if len(x) > 0:
                f.write(f"# X range: {x[0]:.1f} to {x[-1]:.1f} nm\n")
            if len(z) > 0:
                f.write(f"# Z range: {z[0]:.1f} to {z[-1]:.1f} nm\n\n")

            # Get actual enhancement array shape
            enh_shape = enhancement.shape

            # Header row (x values) - use actual shape
            f.write("x/z")
            n_x = min(len(x), enh_shape[0]) if len(enh_shape) > 0 else len(x)
            for i in range(n_x):
                xi = x[i] if i < len(x) else i
                f.write(f"\t{xi:.1f}")
            f.write("\n")

            # Data rows - use actual shape
            n_z = min(len(z), enh_shape[1]) if len(enh_shape) > 1 else len(z)
            for j in range(n_z):
                zi = z[j] if j < len(z) else j
                f.write(f"{zi:.1f}")
                for i in range(n_x):
                    # Safe indexing with bounds check
                    if i < enh_shape[0] and j < enh_shape[1]:
                        f.write(f"\t{enhancement[i, j]:.4e}")
                    else:
                        f.write(f"\t0.0000e+00")
                f.write("\n")

    def export_field_json(self, field_data: Dict[str, np.ndarray],
                          field_stats: Dict[str, float],
                          hotspots: List[Dict],
                          pol_idx: int = 0,
                          filename: Optional[str] = None):
        """
        Export field data with metadata to JSON.

        Args:
            field_data: Dictionary with field enhancement data
            field_stats: Dictionary with field statistics
            hotspots: List of hotspot information
            pol_idx: Polarization index
            filename: Output filename
        """
        if filename is None:
            filename = f'field_pol{pol_idx + 1}'

        filepath = os.path.join(self.output_dir, f'{filename}.json')

        # Safely get arrays with defaults
        x = field_data.get('x', np.array([0]))
        y = field_data.get('y', np.array([0]))
        z = field_data.get('z', np.array([0]))

        export_data = {
            'grid': {
                'x_range': [float(x[0]), float(x[-1])] if len(x) > 0 else [0, 0],
                'y_range': [float(y[0]), float(y[-1])] if len(y) > 0 else [0, 0],
                'z_range': [float(z[0]), float(z[-1])] if len(z) > 0 else [0, 0],
                'nx': len(x),
                'ny': len(y),
                'nz': len(z),
            },
            'statistics': field_stats,
            'hotspots': hotspots,
            'wavelength': float(field_data.get('wavelength', 0)),
            'polarization_idx': pol_idx,
        }

        # Optionally include downsampled field data
        if self.config.get('export_field_arrays', False):
            # Downsample for reasonable file size
            step = max(1, len(field_data['x']) // 50)
            export_data['enhancement_downsampled'] = field_data['enhancement'][::step, ::step].tolist()
            export_data['downsample_step'] = step

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_surface_charges_json(self, charge_data: Dict[str, np.ndarray],
                                     mode_info: Dict[str, Any],
                                     pol_idx: int = 0,
                                     filename: Optional[str] = None):
        """
        Export surface charge data to JSON.

        Args:
            charge_data: Dictionary with charge data
            mode_info: Dictionary with mode analysis
            pol_idx: Polarization index
            filename: Output filename
        """
        if filename is None:
            filename = f'surface_charges_pol{pol_idx + 1}'

        filepath = os.path.join(self.output_dir, f'{filename}.json')

        export_data = {
            'wavelength': float(charge_data.get('wavelength', 0)),
            'polarization_idx': pol_idx,
            'mode_analysis': mode_info,
            'charge_statistics': {
                'max_real': float(np.max(charge_data['charge_real'])),
                'min_real': float(np.min(charge_data['charge_real'])),
                'rms': float(np.sqrt(np.mean(charge_data['charge_magnitude']**2))),
            },
            'n_faces': len(charge_data['charges']),
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_summary(self, analysis_results: Dict[str, Any],
                       filename: str = 'analysis_summary'):
        """
        Export analysis summary to JSON.

        Args:
            analysis_results: Dictionary with all analysis results
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, f'{filename}.json')

        # Make numpy arrays JSON serializable
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        export_data = convert(analysis_results)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_all(self, data: Dict[str, Any], analysis: Dict[str, Any]):
        """
        Export all available data.

        Args:
            data: Raw simulation data
            analysis: Analysis results
        """
        # Export spectra
        if 'spectrum' in data:
            ext_data = data['spectrum'].get('extinction')
            n_pol = ext_data.shape[1] if ext_data is not None and ext_data.ndim > 1 else 1
            for pol_idx in range(n_pol):
                self.export_spectrum_txt(data['spectrum'], pol_idx)
                self.export_spectrum_csv(data['spectrum'], pol_idx)

        # Export unpolarized spectrum
        if 'spectrum_unpolarized' in analysis:
            self.export_spectrum_unpolarized(analysis['spectrum_unpolarized'])

        # Export field data
        if 'field' in data:
            for pol_idx, field_data in data['field'].items():
                self.export_field_txt(field_data, pol_idx)

                stats = analysis.get('field_stats', {}).get(pol_idx, {})
                hotspots = analysis.get('hotspots', {}).get(pol_idx, [])
                self.export_field_json(field_data, stats, hotspots, pol_idx)

        # Export summary
        self.export_summary(analysis)

