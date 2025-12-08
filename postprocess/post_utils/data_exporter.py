"""
Data Exporter

Exports simulation data to TXT files for external plotting.
Supports spectrum data, field 2D maps, and unpolarized calculations.
"""

import os
import numpy as np
from datetime import datetime


class DataExporter:
    """Exports simulation data to various text formats."""

    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = os.path.join(
            config.get('output_dir'),
            config.get('simulation_name')
        )
        self.simulation_name = config.get('simulation_name', 'simulation')

    def export_all(self, data, analysis_results=None):
        """
        Export all data to TXT files.

        Args:
            data: Simulation data dictionary
            analysis_results: Results from spectrum analyzer (optional)

        Returns:
            list: Paths of exported files
        """
        exported_files = []

        # Export spectrum data
        spectrum_files = self.export_spectrum(data, analysis_results)
        exported_files.extend(spectrum_files)

        # Export field data
        if 'fields' in data and data['fields']:
            field_files = self.export_fields(data, analysis_results)
            exported_files.extend(field_files)

        # Export summary
        summary_file = self.export_summary(data, analysis_results)
        if summary_file:
            exported_files.append(summary_file)

        return exported_files

    def export_spectrum(self, data, analysis_results=None):
        """
        Export spectrum data to TXT files.

        Creates:
        - spectra_pol1.txt, spectra_pol2.txt, ... (per-polarization)
        - spectra_unpolarized.txt (if applicable)
        - spectra_all.txt (combined file)
        """
        exported_files = []
        wavelength = data['wavelength']
        n_pol = data['n_polarizations']

        # Get polarization labels
        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        # Export each polarization separately
        for ipol in range(n_pol):
            filename = f'spectra_pol{ipol + 1}.txt'
            filepath = os.path.join(self.output_dir, filename)

            pol_label = self._format_polarization_label(polarizations, ipol)

            header = self._create_spectrum_header(
                f'Polarization {ipol + 1}: {pol_label}',
                ['wavelength(nm)', 'scattering(nm^2)', 'extinction(nm^2)', 'absorption(nm^2)']
            )

            data_array = np.column_stack([
                wavelength,
                data['scattering'][:, ipol],
                data['extinction'][:, ipol],
                data['absorption'][:, ipol]
            ])

            self._write_txt_file(filepath, header, data_array)
            exported_files.append(filepath)

            if self.verbose:
                print(f"  Exported: {filename}")

        # Export unpolarized spectrum if available
        if analysis_results and 'unpolarized_spectrum' in analysis_results:
            unpol = analysis_results['unpolarized_spectrum']
            filename = 'spectra_unpolarized.txt'
            filepath = os.path.join(self.output_dir, filename)

            header = self._create_spectrum_header(
                f'Unpolarized (averaged from {unpol["n_averaged"]} polarizations, method: {unpol["method"]})',
                ['wavelength(nm)', 'scattering(nm^2)', 'extinction(nm^2)', 'absorption(nm^2)']
            )

            data_array = np.column_stack([
                unpol['wavelength'],
                unpol['scattering'],
                unpol['extinction'],
                unpol['absorption']
            ])

            self._write_txt_file(filepath, header, data_array)
            exported_files.append(filepath)

            if self.verbose:
                print(f"  Exported: {filename}")

        # Export combined file (all polarizations + unpolarized)
        combined_file = self._export_combined_spectrum(data, analysis_results)
        if combined_file:
            exported_files.append(combined_file)

        return exported_files

    def _export_combined_spectrum(self, data, analysis_results):
        """Export combined spectrum file with all polarizations."""
        filename = 'spectra_all.txt'
        filepath = os.path.join(self.output_dir, filename)

        wavelength = data['wavelength']
        n_pol = data['n_polarizations']
        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        # Build column names
        columns = ['wavelength(nm)']
        for ipol in range(n_pol):
            pol_label = self._format_polarization_short(polarizations, ipol)
            columns.extend([
                f'sca_{pol_label}(nm^2)',
                f'ext_{pol_label}(nm^2)',
                f'abs_{pol_label}(nm^2)'
            ])

        # Add unpolarized columns if available
        has_unpol = analysis_results and 'unpolarized_spectrum' in analysis_results
        if has_unpol:
            columns.extend([
                'sca_unpol(nm^2)',
                'ext_unpol(nm^2)',
                'abs_unpol(nm^2)'
            ])

        # Build data array
        data_columns = [wavelength]
        for ipol in range(n_pol):
            data_columns.extend([
                data['scattering'][:, ipol],
                data['extinction'][:, ipol],
                data['absorption'][:, ipol]
            ])

        if has_unpol:
            unpol = analysis_results['unpolarized_spectrum']
            data_columns.extend([
                unpol['scattering'],
                unpol['extinction'],
                unpol['absorption']
            ])

        data_array = np.column_stack(data_columns)

        header = self._create_spectrum_header(
            f'Combined spectrum data ({n_pol} polarizations' +
            (', including unpolarized average)' if has_unpol else ')'),
            columns
        )

        self._write_txt_file(filepath, header, data_array)

        if self.verbose:
            print(f"  Exported: {filename}")

        return filepath

    def export_fields(self, data, analysis_results=None):
        """
        Export field data to TXT files.

        Creates:
        - field_pol1_<wavelength>nm.txt, field_pol2_<wavelength>nm.txt, ...
        - field_unpolarized_<wavelength>nm.txt (if applicable)
        """
        exported_files = []
        fields = data.get('fields', [])

        if not fields:
            return exported_files

        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        # Group fields by wavelength for unpolarized calculation
        fields_by_wavelength = {}
        for field in fields:
            wl = field.get('wavelength', 0)
            wl_key = f"{wl:.1f}"
            if wl_key not in fields_by_wavelength:
                fields_by_wavelength[wl_key] = []
            fields_by_wavelength[wl_key].append(field)

        # Export each field
        for field in fields:
            pol_idx = field.get('polarization_idx', 0)
            wavelength = field.get('wavelength', 0)

            filename = f'field_pol{pol_idx + 1}_{wavelength:.0f}nm.txt'
            filepath = os.path.join(self.output_dir, filename)

            pol_label = self._format_polarization_label(polarizations, pol_idx)

            exported = self._export_single_field(
                filepath, field, f'Polarization {pol_idx + 1}: {pol_label}'
            )
            if exported:
                exported_files.append(filepath)

                if self.verbose:
                    print(f"  Exported: {filename}")

        # Export unpolarized fields if conditions are met
        unpol_info = analysis_results.get('unpolarized', {}) if analysis_results else {}
        if unpol_info.get('can_calculate', False):
            unpol_field_files = self._export_unpolarized_fields(
                fields_by_wavelength, unpol_info, analysis_results
            )
            exported_files.extend(unpol_field_files)

        return exported_files

    def _export_single_field(self, filepath, field_data, description):
        """Export a single field to TXT file."""
        x_grid = field_data.get('x_grid')
        y_grid = field_data.get('y_grid')
        z_grid = field_data.get('z_grid')
        enhancement = field_data.get('enhancement')
        intensity = field_data.get('intensity')
        wavelength = field_data.get('wavelength', 0)

        if enhancement is None:
            return False

        # Handle complex data
        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)
        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        # Determine plane type and flatten data
        plane_type = self._determine_plane_type(x_grid, y_grid, z_grid)

        # Flatten grids and data
        if isinstance(x_grid, np.ndarray):
            x_flat = x_grid.flatten()
            y_flat = y_grid.flatten()
            z_flat = z_grid.flatten()
            enh_flat = enhancement.flatten()
            int_flat = intensity.flatten() if intensity is not None else np.zeros_like(enh_flat)
        else:
            x_flat = np.array([x_grid])
            y_flat = np.array([y_grid])
            z_flat = np.array([z_grid])
            enh_flat = np.array([enhancement])
            int_flat = np.array([intensity]) if intensity is not None else np.array([0])

        # Create header
        grid_shape = enhancement.shape if isinstance(enhancement, np.ndarray) else (1, 1)
        header_lines = [
            f'# Field Data - {description}',
            f'# Wavelength: {wavelength:.1f} nm',
            f'# Plane: {plane_type}',
            f'# Grid shape: {grid_shape}',
            f'# Export time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '#',
            '# Columns: x(nm), y(nm), z(nm), enhancement(|E|/|E0|), intensity(|E|^2)'
        ]
        header = '\n'.join(header_lines)

        # Stack data
        data_array = np.column_stack([x_flat, y_flat, z_flat, enh_flat, int_flat])

        self._write_txt_file(filepath, header, data_array,
                            fmt=['%.4f', '%.4f', '%.4f', '%.6e', '%.6e'])
        return True

    def _export_unpolarized_fields(self, fields_by_wavelength, unpol_info, analysis_results):
        """
        Export unpolarized field data.

        For unpolarized field calculation (FDTD-style incoherent average):
        - Intensity: I_unpol = mean(I_pol1, I_pol2, ...)
        - Enhancement: enh_unpol = sqrt(mean(enh1^2, enh2^2, ...))
        """
        exported_files = []
        method = unpol_info.get('method', '')

        # Determine expected number of polarizations
        if method == 'orthogonal_2pol_average':
            expected_n_pol = 2
        elif method == 'orthogonal_3dir_average':
            expected_n_pol = 3
        else:
            return exported_files

        for wl_key, wl_fields in fields_by_wavelength.items():
            # Check if we have all required polarizations at this wavelength
            if len(wl_fields) != expected_n_pol:
                continue

            # Sort by polarization index
            wl_fields_sorted = sorted(wl_fields, key=lambda f: f.get('polarization_idx', 0))

            # Get reference field for grid info
            ref_field = wl_fields_sorted[0]
            x_grid = ref_field.get('x_grid')
            y_grid = ref_field.get('y_grid')
            z_grid = ref_field.get('z_grid')
            wavelength = ref_field.get('wavelength', 0)

            # Calculate unpolarized intensity and enhancement
            intensities = []
            enhancements_sq = []

            for field in wl_fields_sorted:
                enh = field.get('enhancement')
                inten = field.get('intensity')

                if enh is None:
                    continue

                if np.iscomplexobj(enh):
                    enh = np.abs(enh)
                if inten is not None and np.iscomplexobj(inten):
                    inten = np.abs(inten)

                enhancements_sq.append(enh ** 2)
                if inten is not None:
                    intensities.append(inten)

            if len(enhancements_sq) != expected_n_pol:
                continue

            # Incoherent average
            unpol_enh_sq = np.mean(enhancements_sq, axis=0)
            unpol_enhancement = np.sqrt(unpol_enh_sq)

            if intensities and len(intensities) == expected_n_pol:
                unpol_intensity = np.mean(intensities, axis=0)
            else:
                unpol_intensity = unpol_enh_sq  # Use enh^2 as proxy for intensity

            # Create unpolarized field dict
            unpol_field = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid,
                'enhancement': unpol_enhancement,
                'intensity': unpol_intensity,
                'wavelength': wavelength
            }

            # Export
            filename = f'field_unpolarized_{wavelength:.0f}nm.txt'
            filepath = os.path.join(self.output_dir, filename)

            description = f'Unpolarized (incoherent average of {expected_n_pol} polarizations)'
            exported = self._export_single_field(filepath, unpol_field, description)

            if exported:
                exported_files.append(filepath)
                if self.verbose:
                    print(f"  Exported: {filename}")

        return exported_files

    def export_summary(self, data, analysis_results=None):
        """Export analysis summary to TXT file."""
        filename = 'analysis_summary.txt'
        filepath = os.path.join(self.output_dir, filename)

        lines = [
            '=' * 70,
            f'SIMULATION ANALYSIS SUMMARY',
            f'Simulation: {self.simulation_name}',
            f'Export time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '=' * 70,
            '',
            '--- SPECTRUM DATA ---',
            f'Wavelength range: {data["wavelength"][0]:.1f} - {data["wavelength"][-1]:.1f} nm',
            f'Number of points: {len(data["wavelength"])}',
            f'Number of polarizations: {data["n_polarizations"]}',
            ''
        ]

        # Polarization info
        polarizations = data.get('polarizations', [])
        for ipol in range(data['n_polarizations']):
            pol_label = self._format_polarization_label(polarizations, ipol)
            lines.append(f'  Polarization {ipol + 1}: {pol_label}')

            if analysis_results:
                peak_wl = analysis_results.get('peak_wavelengths', [])[ipol] if ipol < len(analysis_results.get('peak_wavelengths', [])) else 'N/A'
                lines.append(f'    Peak wavelength: {peak_wl} nm')

        # Unpolarized info
        if analysis_results and 'unpolarized_spectrum' in analysis_results:
            unpol = analysis_results['unpolarized_spectrum']
            lines.extend([
                '',
                '--- UNPOLARIZED (FDTD-style incoherent average) ---',
                f'Method: {unpol["method"]}',
                f'Averaged polarizations: {unpol["n_averaged"]}',
                f'Peak wavelength: {unpol["peak_wavelength"]:.1f} nm',
                f'Peak absorption: {unpol["peak_absorption"]:.2f} nm^2',
                f'Peak extinction: {unpol["peak_extinction"]:.2f} nm^2',
                f'Peak scattering: {unpol["peak_scattering"]:.2f} nm^2',
            ])
        elif analysis_results and 'unpolarized' in analysis_results:
            unpol_info = analysis_results['unpolarized']
            lines.extend([
                '',
                '--- UNPOLARIZED ---',
                f'Can calculate: {unpol_info["can_calculate"]}',
                f'Reason: {unpol_info["reason"]}',
            ])

        # Field info
        if 'fields' in data and data['fields']:
            lines.extend([
                '',
                '--- FIELD DATA ---',
                f'Number of field entries: {len(data["fields"])}',
            ])
            for idx, field in enumerate(data['fields']):
                pol_idx = field.get('polarization_idx', idx)
                wl = field.get('wavelength', 0)
                enh = field.get('enhancement')
                max_enh = np.nanmax(np.abs(enh)) if enh is not None else 'N/A'
                lines.append(f'  Field {idx + 1}: pol{pol_idx + 1}, λ={wl:.1f}nm, max_enhancement={max_enh}')

        lines.extend(['', '=' * 70])

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        if self.verbose:
            print(f"  Exported: {filename}")

        return filepath

    def _create_spectrum_header(self, description, columns):
        """Create header for spectrum TXT file."""
        header_lines = [
            f'# {description}',
            f'# Simulation: {self.simulation_name}',
            f'# Export time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '#',
            '# ' + '\t'.join(columns)
        ]
        return '\n'.join(header_lines)

    def _write_txt_file(self, filepath, header, data_array, fmt='%.6e'):
        """Write data to TXT file with header."""
        with open(filepath, 'w') as f:
            f.write(header + '\n')
            if isinstance(fmt, list):
                for row in data_array:
                    line = '\t'.join(f % v for f, v in zip(fmt, row))
                    f.write(line + '\n')
            else:
                np.savetxt(f, data_array, fmt=fmt, delimiter='\t')

    def _format_polarization_label(self, polarizations, ipol):
        """Format full polarization label."""
        if ipol < len(polarizations):
            pol = polarizations[ipol]
            if isinstance(pol, (list, np.ndarray)) and len(pol) == 3:
                return f'[{pol[0]:.2g}, {pol[1]:.2g}, {pol[2]:.2g}]'
        return f'pol{ipol + 1}'

    def _format_polarization_short(self, polarizations, ipol):
        """Format short polarization label for column names."""
        if ipol < len(polarizations):
            pol = polarizations[ipol]
            if isinstance(pol, (list, np.ndarray)) and len(pol) == 3:
                # Check for axis-aligned polarizations
                if abs(pol[0]) > 0.9 and abs(pol[1]) < 0.1 and abs(pol[2]) < 0.1:
                    return 'x'
                elif abs(pol[0]) < 0.1 and abs(pol[1]) > 0.9 and abs(pol[2]) < 0.1:
                    return 'y'
                elif abs(pol[0]) < 0.1 and abs(pol[1]) < 0.1 and abs(pol[2]) > 0.9:
                    return 'z'
        return f'pol{ipol + 1}'

    def _determine_plane_type(self, x_grid, y_grid, z_grid):
        """Determine which 2D plane the field is calculated on."""
        if not isinstance(x_grid, np.ndarray):
            return 'point'

        x_unique = len(np.unique(x_grid))
        y_unique = len(np.unique(y_grid))
        z_unique = len(np.unique(z_grid))

        if x_unique == 1:
            return 'yz'
        elif y_unique == 1:
            return 'xz'
        elif z_unique == 1:
            return 'xy'
        else:
            return '3d'
