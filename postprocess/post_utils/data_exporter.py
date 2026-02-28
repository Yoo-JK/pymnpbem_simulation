import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime

import numpy as np


class DataExporter(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose
        self.output_dir = os.path.join(
            config.get('output_dir'),
            config.get('simulation_name')
        )
        self.simulation_name = config.get('simulation_name', 'simulation')

    def export_all(self,
            data: Dict[str, Any],
            analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:

        exported_files = []

        spectrum_files = self.export_spectrum(data, analysis_results)
        exported_files.extend(spectrum_files)

        if 'fields' in data and data['fields']:
            field_files = self.export_fields(data, analysis_results)
            exported_files.extend(field_files)

        summary_file = self.export_summary(data, analysis_results)
        if summary_file:
            exported_files.append(summary_file)

        return exported_files

    def export_spectrum(self,
            data: Dict[str, Any],
            analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:

        exported_files = []
        wavelength = data.get('wavelength')
        n_pol = data.get('n_polarizations', 0)

        extinction = data.get('extinction')
        if wavelength is None or extinction is None:
            return exported_files
        if not isinstance(extinction, np.ndarray) or extinction.size == 0:
            return exported_files
        if isinstance(wavelength, np.ndarray) and len(wavelength) > 0:
            if extinction.ndim == 1 and len(extinction) != len(wavelength):
                return exported_files
            elif extinction.ndim == 2 and extinction.shape[0] != len(wavelength):
                return exported_files

        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        for ipol in range(n_pol):
            filename = 'spectra_pol{}.txt'.format(ipol + 1)
            filepath = os.path.join(self.output_dir, filename)

            pol_label = self._format_polarization_label(polarizations, ipol)

            header = self._create_spectrum_header(
                'Polarization {}: {}'.format(ipol + 1, pol_label),
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
                print('  Exported: {}'.format(filename))

        if analysis_results and 'unpolarized_spectrum' in analysis_results:
            unpol = analysis_results['unpolarized_spectrum']
            filename = 'spectra_unpolarized.txt'
            filepath = os.path.join(self.output_dir, filename)

            header = self._create_spectrum_header(
                'Unpolarized (averaged from {} polarizations, method: {})'.format(
                    unpol['n_averaged'], unpol['method']),
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
                print('  Exported: {}'.format(filename))

        combined_file = self._export_combined_spectrum(data, analysis_results)
        if combined_file:
            exported_files.append(combined_file)

        return exported_files

    def _export_combined_spectrum(self,
            data: Dict[str, Any],
            analysis_results: Optional[Dict[str, Any]]) -> Optional[str]:

        filename = 'spectra_all.txt'
        filepath = os.path.join(self.output_dir, filename)

        wavelength = data['wavelength']
        n_pol = data['n_polarizations']
        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        columns = ['wavelength(nm)']
        for ipol in range(n_pol):
            pol_label = self._format_polarization_short(polarizations, ipol)
            columns.extend([
                'sca_{}(nm^2)'.format(pol_label),
                'ext_{}(nm^2)'.format(pol_label),
                'abs_{}(nm^2)'.format(pol_label)
            ])

        has_unpol = analysis_results and 'unpolarized_spectrum' in analysis_results
        if has_unpol:
            columns.extend([
                'sca_unpol(nm^2)',
                'ext_unpol(nm^2)',
                'abs_unpol(nm^2)'
            ])

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
            'Combined spectrum data ({} polarizations'.format(n_pol) +
            (', including unpolarized average)' if has_unpol else ')'),
            columns
        )

        self._write_txt_file(filepath, header, data_array)

        if self.verbose:
            print('  Exported: {}'.format(filename))

        return filepath

    def export_fields(self,
            data: Dict[str, Any],
            analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:

        exported_files = []
        fields = data.get('fields', [])

        if not fields:
            return exported_files

        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        fields_by_wavelength = {}
        for field in fields:
            wl = field.get('wavelength', 0)
            wl_key = '{:.1f}'.format(wl)
            if wl_key not in fields_by_wavelength:
                fields_by_wavelength[wl_key] = []
            fields_by_wavelength[wl_key].append(field)

        for field in fields:
            pol_idx = field.get('polarization_idx', 0)
            wavelength = field.get('wavelength', 0)

            filename = 'field_pol{}_{:.0f}nm.txt'.format(pol_idx + 1, wavelength)
            filepath = os.path.join(self.output_dir, filename)

            pol_label = self._format_polarization_label(polarizations, pol_idx)

            exported = self._export_single_field(
                filepath, field, 'Polarization {}: {}'.format(pol_idx + 1, pol_label)
            )
            if exported:
                exported_files.append(filepath)

                if self.verbose:
                    print('  Exported: {}'.format(filename))

        unpol_info = analysis_results.get('unpolarized', {}) if analysis_results else {}
        if unpol_info.get('can_calculate', False):
            unpol_field_files = self._export_unpolarized_fields(
                fields_by_wavelength, unpol_info, analysis_results
            )
            exported_files.extend(unpol_field_files)

        return exported_files

    def _export_single_field(self,
            filepath: str,
            field_data: Dict[str, Any],
            description: str) -> bool:

        x_grid = field_data.get('x_grid')
        y_grid = field_data.get('y_grid')
        z_grid = field_data.get('z_grid')
        enhancement = field_data.get('enhancement')
        intensity = field_data.get('intensity')
        wavelength = field_data.get('wavelength', 0)

        if enhancement is None:
            return False

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)
        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        plane_type = self._determine_plane_type(x_grid, y_grid, z_grid)

        def to_1d_array(val):
            if val is None:
                return np.array([0])
            if not isinstance(val, np.ndarray):
                return np.array([val])
            if val.ndim == 0:
                return np.array([val.item()])
            return np.unique(val)

        x_coords = to_1d_array(x_grid)
        y_coords = to_1d_array(y_grid)
        z_coords = to_1d_array(z_grid)

        enh_arr = enhancement if isinstance(enhancement, np.ndarray) else np.array([[enhancement]])
        if enh_arr.ndim == 1:
            enh_arr = enh_arr.reshape(1, -1)

        int_arr = intensity if intensity is not None else None
        if int_arr is not None:
            if not isinstance(int_arr, np.ndarray):
                int_arr = np.array([[int_arr]])
            if int_arr.ndim == 1:
                int_arr = int_arr.reshape(1, -1)

        n_total = enh_arr.size

        def make_coord_grid(coords, target_shape, is_row_varying):
            if len(coords) == 1:
                return np.full(target_shape, coords[0])

            try:
                if is_row_varying:
                    reshaped = coords.reshape(-1, 1)
                    return np.broadcast_to(reshaped, target_shape).copy()
                else:
                    reshaped = coords.reshape(1, -1)
                    return np.broadcast_to(reshaped, target_shape).copy()
            except ValueError:
                repeated = np.tile(coords, (n_total // len(coords)) + 1)[:n_total]
                return repeated.reshape(target_shape)

        if plane_type == 'xz':
            if len(x_coords) * len(z_coords) == n_total:
                X, Z = np.meshgrid(x_coords, z_coords)
                Y = np.full_like(X, y_coords[0] if len(y_coords) > 0 else 0, dtype = float)
            else:
                X = make_coord_grid(x_coords, enh_arr.shape, is_row_varying = False)
                Z = make_coord_grid(z_coords, enh_arr.shape, is_row_varying = True)
                Y = np.full(enh_arr.shape, y_coords[0] if len(y_coords) > 0 else 0, dtype = float)
        elif plane_type == 'xy':
            if len(x_coords) * len(y_coords) == n_total:
                X, Y = np.meshgrid(x_coords, y_coords)
                Z = np.full_like(X, z_coords[0] if len(z_coords) > 0 else 0, dtype = float)
            else:
                X = make_coord_grid(x_coords, enh_arr.shape, is_row_varying = False)
                Y = make_coord_grid(y_coords, enh_arr.shape, is_row_varying = True)
                Z = np.full(enh_arr.shape, z_coords[0] if len(z_coords) > 0 else 0, dtype = float)
        elif plane_type == 'yz':
            if len(y_coords) * len(z_coords) == n_total:
                Y, Z = np.meshgrid(y_coords, z_coords)
                X = np.full_like(Y, x_coords[0] if len(x_coords) > 0 else 0, dtype = float)
            else:
                Y = make_coord_grid(y_coords, enh_arr.shape, is_row_varying = False)
                Z = make_coord_grid(z_coords, enh_arr.shape, is_row_varying = True)
                X = np.full(enh_arr.shape, x_coords[0] if len(x_coords) > 0 else 0, dtype = float)
        else:
            if isinstance(x_grid, np.ndarray) and x_grid.shape == enh_arr.shape:
                X, Y, Z = x_grid.copy(), y_grid.copy(), z_grid.copy()
            else:
                X = np.tile(x_coords, (n_total // len(x_coords)) + 1)[:n_total].reshape(enh_arr.shape)
                Y = np.tile(y_coords, (n_total // len(y_coords)) + 1)[:n_total].reshape(enh_arr.shape)
                Z = np.tile(z_coords, (n_total // len(z_coords)) + 1)[:n_total].reshape(enh_arr.shape)

        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        enh_flat = enh_arr.flatten()
        int_flat = int_arr.flatten() if int_arr is not None else np.zeros_like(enh_flat)

        grid_shape = enh_arr.shape
        header_lines = [
            '# Field Data - {}'.format(description),
            '# Wavelength: {:.1f} nm'.format(wavelength),
            '# Plane: {}'.format(plane_type),
            '# Grid shape: {}'.format(grid_shape),
            '# Export time: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            '#',
            '# Columns: x(nm), y(nm), z(nm), enhancement(|E|/|E0|), intensity(|E|^2)'
        ]
        header = '\n'.join(header_lines)

        data_array = np.column_stack([x_flat, y_flat, z_flat, enh_flat, int_flat])

        self._write_txt_file(filepath, header, data_array,
                            fmt = ['%.4f', '%.4f', '%.4f', '%.6e', '%.6e'])
        return True

    def _export_unpolarized_fields(self,
            fields_by_wavelength: Dict[str, List[Dict]],
            unpol_info: Dict[str, Any],
            analysis_results: Dict[str, Any]) -> List[str]:

        exported_files = []
        method = unpol_info.get('method', '')

        if method == 'orthogonal_2pol_average':
            expected_n_pol = 2
        elif method == 'orthogonal_3dir_average':
            expected_n_pol = 3
        else:
            return exported_files

        for wl_key, wl_fields in fields_by_wavelength.items():
            if len(wl_fields) != expected_n_pol:
                continue

            wl_fields_sorted = sorted(wl_fields, key = lambda f: f.get('polarization_idx', 0))

            ref_field = wl_fields_sorted[0]
            x_grid = ref_field.get('x_grid')
            y_grid = ref_field.get('y_grid')
            z_grid = ref_field.get('z_grid')
            wavelength = ref_field.get('wavelength', 0)

            intensities = []
            enhancements = []

            for field in wl_fields_sorted:
                enh = field.get('enhancement')
                inten = field.get('intensity')

                if enh is None:
                    continue

                if np.iscomplexobj(enh):
                    enh = np.abs(enh)
                if inten is not None and np.iscomplexobj(inten):
                    inten = np.abs(inten)

                enhancements.append(enh)
                if inten is not None:
                    intensities.append(inten)

            if len(enhancements) != expected_n_pol:
                continue

            unpol_enhancement = np.mean(enhancements, axis = 0)

            if intensities and len(intensities) == expected_n_pol:
                unpol_intensity = np.mean(intensities, axis = 0)
            else:
                unpol_intensity = unpol_enhancement

            unpol_field = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid,
                'enhancement': unpol_enhancement,
                'intensity': unpol_intensity,
                'wavelength': wavelength
            }

            filename = 'field_unpolarized_{:.0f}nm.txt'.format(wavelength)
            filepath = os.path.join(self.output_dir, filename)

            description = 'Unpolarized (incoherent average of {} polarizations)'.format(expected_n_pol)
            exported = self._export_single_field(filepath, unpol_field, description)

            if exported:
                exported_files.append(filepath)
                if self.verbose:
                    print('  Exported: {}'.format(filename))

        return exported_files

    def export_summary(self,
            data: Dict[str, Any],
            analysis_results: Optional[Dict[str, Any]] = None) -> Optional[str]:

        filename = 'analysis_summary.txt'
        filepath = os.path.join(self.output_dir, filename)

        lines = [
            '=' * 70,
            'SIMULATION ANALYSIS SUMMARY',
            'Simulation: {}'.format(self.simulation_name),
            'Export time: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            '=' * 70,
            '',
            '--- SPECTRUM DATA ---',
            'Wavelength range: {:.1f} - {:.1f} nm'.format(
                data['wavelength'][0], data['wavelength'][-1]),
            'Number of points: {}'.format(len(data['wavelength'])),
            'Number of polarizations: {}'.format(data['n_polarizations']),
            ''
        ]

        polarizations = data.get('polarizations', [])
        for ipol in range(data['n_polarizations']):
            pol_label = self._format_polarization_label(polarizations, ipol)
            lines.append('  Polarization {}: {}'.format(ipol + 1, pol_label))

            if analysis_results:
                peak_wls = analysis_results.get('peak_wavelengths', [])
                peak_wl = peak_wls[ipol] if ipol < len(peak_wls) else 'N/A'
                lines.append('    Peak wavelength: {} nm'.format(peak_wl))

        if analysis_results and 'unpolarized_spectrum' in analysis_results:
            unpol = analysis_results['unpolarized_spectrum']
            lines.extend([
                '',
                '--- UNPOLARIZED (FDTD-style incoherent average) ---',
                'Method: {}'.format(unpol['method']),
                'Averaged polarizations: {}'.format(unpol['n_averaged']),
                'Peak wavelength: {:.1f} nm'.format(unpol['peak_wavelength']),
                'Peak absorption: {:.2f} nm^2'.format(unpol['peak_absorption']),
                'Peak extinction: {:.2f} nm^2'.format(unpol['peak_extinction']),
                'Peak scattering: {:.2f} nm^2'.format(unpol['peak_scattering']),
            ])
        elif analysis_results and 'unpolarized' in analysis_results:
            unpol_info = analysis_results['unpolarized']
            lines.extend([
                '',
                '--- UNPOLARIZED ---',
                'Can calculate: {}'.format(unpol_info['can_calculate']),
                'Reason: {}'.format(unpol_info['reason']),
            ])

        if 'fields' in data and data['fields']:
            lines.extend([
                '',
                '--- FIELD DATA ---',
                'Number of field entries: {}'.format(len(data['fields'])),
            ])
            for idx, field in enumerate(data['fields']):
                pol_idx = field.get('polarization_idx', idx)
                wl = field.get('wavelength', 0)
                enh = field.get('enhancement')
                max_enh = np.nanmax(np.abs(enh)) if enh is not None else 'N/A'
                lines.append('  Field {}: pol{}, lambda={:.1f}nm, max_enhancement={}'.format(
                    idx + 1, pol_idx, wl, max_enh))

        lines.extend(['', '=' * 70])

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        if self.verbose:
            print('  Exported: {}'.format(filename))

        return filepath

    def _create_spectrum_header(self,
            description: str,
            columns: List[str]) -> str:

        header_lines = [
            '# {}'.format(description),
            '# Simulation: {}'.format(self.simulation_name),
            '# Export time: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            '#',
            '# ' + '\t'.join(columns)
        ]
        return '\n'.join(header_lines)

    def _write_txt_file(self,
            filepath: str,
            header: str,
            data_array: np.ndarray,
            fmt: Union[str, List[str]] = '%.6e') -> None:

        with open(filepath, 'w') as f:
            f.write(header + '\n')
            if isinstance(fmt, list):
                for row in data_array:
                    line = '\t'.join(fmt_str % v for fmt_str, v in zip(fmt, row))
                    f.write(line + '\n')
            else:
                np.savetxt(f, data_array, fmt = fmt, delimiter = '\t')

    def _format_polarization_label(self,
            polarizations: Any,
            ipol: int) -> str:

        if ipol < len(polarizations):
            pol = polarizations[ipol]
            if isinstance(pol, (list, np.ndarray)) and len(pol) == 3:
                return '[{:.2g}, {:.2g}, {:.2g}]'.format(pol[0], pol[1], pol[2])
        return 'pol{}'.format(ipol + 1)

    def _format_polarization_short(self,
            polarizations: Any,
            ipol: int) -> str:

        if ipol < len(polarizations):
            pol = polarizations[ipol]
            if isinstance(pol, (list, np.ndarray)) and len(pol) == 3:
                if abs(pol[0]) > 0.9 and abs(pol[1]) < 0.1 and abs(pol[2]) < 0.1:
                    return 'x'
                elif abs(pol[0]) < 0.1 and abs(pol[1]) > 0.9 and abs(pol[2]) < 0.1:
                    return 'y'
                elif abs(pol[0]) < 0.1 and abs(pol[1]) < 0.1 and abs(pol[2]) > 0.9:
                    return 'z'
        return 'pol{}'.format(ipol + 1)

    def _determine_plane_type(self,
            x_grid: Any,
            y_grid: Any,
            z_grid: Any) -> str:

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
