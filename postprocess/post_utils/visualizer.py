import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Circle, Rectangle

from .geometry_cross_section import GeometryCrossSection
from .edge_filter import get_sphere_boundaries_from_config, find_edge_artifacts


class Visualizer(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose
        self.output_dir = os.path.join(config.get('output_dir'), config.get('simulation_name'))
        self.save_plots = config.get('save_plots', True)
        self.plot_format = config.get('plot_format', ['png', 'pdf'])
        self.dpi = config.get('plot_dpi', 300)
        self.polarizations = config.get('polarizations', [])
        self.propagation_dirs = config.get('propagation_dirs', [])
        self.geometry = GeometryCrossSection(config, verbose)

    def _format_vector_label(self,
            vec: Optional[Any]) -> str:

        if vec is None or len(vec) == 0:
            return ''
        vec_rounded = np.round(vec, 3)
        return '[{:.0f} {:.0f} {:.0f}]'.format(vec_rounded[0], vec_rounded[1], vec_rounded[2])

    def _get_polarization_label(self,
            ipol: int) -> str:

        if ipol < len(self.polarizations) and ipol < len(self.propagation_dirs):
            pol_vec = self.polarizations[ipol]
            prop_vec = self.propagation_dirs[ipol]
            pol_str = self._format_vector_label(pol_vec)
            prop_str = self._format_vector_label(prop_vec)
            return 'Pol{} Prop{}'.format(pol_str, prop_str)
        else:
            return 'Polarization {}'.format(ipol + 1)

    def create_all_plots(self,
            data: Dict[str, Any],
            analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:

        plots_created = []

        has_spectrum_data = (
            'wavelength' in data and
            'extinction' in data and
            data['extinction'] is not None and
            data['extinction'].size > 0
        )
        if has_spectrum_data:
            spectrum_file = self.plot_spectrum(data)
            plots_created.append(spectrum_file)

        if has_spectrum_data and data['extinction'].shape[1] > 1:
            pol_file = self.plot_polarization_comparison(data)
            plots_created.append(pol_file)

        if analysis_results and 'unpolarized_spectrum' in analysis_results:
            unpol_files = self.plot_unpolarized_spectrum(data, analysis_results)
            plots_created.extend(unpol_files)

        if 'fields' in data:
            field_files = self.plot_fields(data)
            plots_created.extend(field_files)

            if analysis_results and analysis_results.get('unpolarized', {}).get('can_calculate', False):
                unpol_field_files = self.plot_unpolarized_fields(data, analysis_results)
                plots_created.extend(unpol_field_files)

        if 'surface_charge' in data and data['surface_charge']:
            if self.verbose:
                print('\n  Creating surface charge plots...')
            sc_files = self.plot_surface_charge(data)
            plots_created.extend(sc_files)

        return plots_created

    def plot_spectrum(self,
            data: Dict[str, Any]) -> List[str]:

        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']

        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')

        if xaxis_unit == 'energy':
            # E(eV) = 1239.84 / lambda(nm)
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            xdata = xdata[::-1]
            extinction = extinction[::-1, :]
            scattering = scattering[::-1, :]
            absorption = absorption[::-1, :]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'

        n_pol = extinction.shape[1]
        saved_files_all = []

        for ipol in range(n_pol):
            fig, ax = plt.subplots(figsize = (10, 6))

            ax.plot(xdata, extinction[:, ipol], 'b-', linewidth = 2, label = 'Extinction')
            ax.plot(xdata, scattering[:, ipol], 'r--', linewidth = 2, label = 'Scattering')
            ax.plot(xdata, absorption[:, ipol], 'g:', linewidth = 2, label = 'Absorption')

            ax.set_xlabel(xlabel_text, fontsize = 12)
            ax.set_ylabel('Cross Section (nm²)', fontsize = 12)

            pol_label = self._get_polarization_label(ipol)
            ax.set_title('Optical Spectra - {}'.format(pol_label), fontsize = 14, fontweight = 'bold')
            ax.legend(fontsize = 11)
            ax.grid(True, alpha = 0.3)

            if xaxis_unit == 'energy':
                ax.invert_xaxis()

            plt.tight_layout()

            base_filename = 'simulation_spectrum_pol{}'.format(ipol + 1)
            saved_files = self._save_figure(fig, base_filename)
            if saved_files:
                saved_files_all.extend(saved_files)
            plt.close(fig)

        return saved_files_all

    def plot_polarization_comparison(self,
            data: Dict[str, Any]) -> List[str]:

        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        n_pol = extinction.shape[1]

        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')

        if xaxis_unit == 'energy':
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            xdata = xdata[::-1]
            extinction = extinction[::-1, :]
            scattering = scattering[::-1, :]
            absorption = absorption[::-1, :]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'

        colors = plt.cm.viridis(np.linspace(0, 1, n_pol))
        saved_files_all = []

        # ========== Extinction Comparison ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, extinction[:, i],
                   color = colors[i], linewidth = 2,
                   label = pol_label)

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize = 12)
        ax.set_title('Polarization Comparison - Extinction', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()

        base_filename = 'simulation_polarization_extinction'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)

        # ========== Scattering Comparison ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, scattering[:, i],
                   color = colors[i], linewidth = 2,
                   label = pol_label)

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Scattering Cross Section (nm²)', fontsize = 12)
        ax.set_title('Polarization Comparison - Scattering', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()

        base_filename = 'simulation_polarization_scattering'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)

        # ========== Absorption Comparison ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, absorption[:, i],
                   color = colors[i], linewidth = 2,
                   label = pol_label)

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Absorption Cross Section (nm²)', fontsize = 12)
        ax.set_title('Polarization Comparison - Absorption', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()

        base_filename = 'simulation_polarization_absorption'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)

        return saved_files_all

    def plot_fields(self,
            data: Dict[str, Any]) -> List[str]:

        if 'fields' not in data or not data['fields']:
            return []

        fields = data['fields']
        saved_files = []

        for idx, field_data in enumerate(fields):
            pol_idx = field_data.get('polarization_idx', idx)
            wl_idx = field_data.get('wavelength_idx')
            wavelength = field_data.get('wavelength')

            enhancement_file = self._plot_field_enhancement(field_data, pol_idx, wl_idx)
            if enhancement_file:
                saved_files.extend(enhancement_file)

            if 'intensity' in field_data and field_data['intensity'] is not None:
                intensity_file = self._plot_field_intensity(field_data, pol_idx, wl_idx)
                if intensity_file:
                    saved_files.extend(intensity_file)

            if self._is_2d_slice(field_data):
                vector_file = self._plot_field_vectors(field_data, pol_idx, wl_idx)
                if vector_file:
                    saved_files.extend(vector_file)

        if self.verbose:
            print('\n  Creating separate internal/external field plots...')

        separate_files = self.plot_field_separate_internal_external(fields)
        if separate_files:
            saved_files.extend(separate_files)
            if self.verbose:
                print('  Created {} separate field plot(s)'.format(len(separate_files)))

        return saved_files

    def _plot_field_enhancement(self,
            field_data: Dict[str, Any],
            polarization_idx: int,
            wavelength_idx: Optional[int] = None) -> List[str]:

        enhancement = field_data['enhancement']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']

        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([[enhancement]])
        elif enhancement.ndim == 0:
            enhancement = np.array([[enhancement.item()]])
        elif enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)

        n_unique_x = len(np.unique(x_grid))
        n_unique_y = len(np.unique(y_grid))
        if enhancement.shape == (n_unique_x, n_unique_y):
            enhancement = enhancement.T

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        pol_label = self._get_polarization_label(polarization_idx)

        enhancement_masked = np.ma.masked_invalid(enhancement)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

        valid_data = enhancement_masked.compressed()
        if len(valid_data) > 0:
            vmin_linear = np.percentile(valid_data, 1)
            vmax_linear = np.percentile(valid_data, 99)
        else:
            vmin_linear, vmax_linear = 0, 1

        # Linear scale with percentile clipping
        im1 = ax1.imshow(enhancement_masked, extent = extent, origin = 'lower',
                        cmap = 'hot', aspect = 'auto', vmin = vmin_linear, vmax = vmax_linear)
        ax1.set_xlabel(x_label, fontsize = 11)
        ax1.set_ylabel(y_label, fontsize = 11)
        ax1.set_title('Intensity Enhancement (Linear)\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                     fontsize = 11, fontweight = 'bold')
        cbar1 = plt.colorbar(im1, ax = ax1)
        cbar1.set_label('|E|²/|E₀|²', fontsize = 11)

        z_plane = float(z_grid.flat[0])
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax1, section, plane_type)

        # Log scale
        if len(valid_data) > 0 and np.any(valid_data > 0):
            positive_data = valid_data[valid_data > 0]
            vmin_log = max(np.percentile(positive_data, 5), 0.1)  # At least 0.1
            vmax_log = np.percentile(positive_data, 99.5)

            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 100

            im2 = ax2.imshow(enhancement_masked, extent = extent, origin = 'lower',
                            cmap = 'hot', aspect = 'auto',
                            norm = LogNorm(vmin = vmin_log, vmax = vmax_log))
            ax2.set_xlabel(x_label, fontsize = 11)
            ax2.set_ylabel(y_label, fontsize = 11)
            ax2.set_title('Intensity Enhancement (Log Scale)\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                         fontsize = 11, fontweight = 'bold')
            cbar2 = plt.colorbar(im2, ax = ax2)
            for section in sections:
                self._draw_material_boundary(ax2, section, plane_type)
            cbar2.set_label('|E|²/|E₀|²', fontsize = 11)
        else:
            im2 = ax2.imshow(enhancement_masked, extent = extent, origin = 'lower',
                            cmap = 'hot', aspect = 'auto')
            ax2.set_xlabel(x_label, fontsize = 11)
            ax2.set_ylabel(y_label, fontsize = 11)
            ax2.set_title('Intensity Enhancement\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                         fontsize = 11, fontweight = 'bold')
            cbar2 = plt.colorbar(im2, ax = ax2)
            for section in sections:
                self._draw_material_boundary(ax2, section, plane_type)
            cbar2.set_label('|E|²/|E₀|²', fontsize = 11)

        plt.tight_layout()

        if wavelength_idx is not None:
            base_filename = 'field_enhancement_wl{}_pol{}_{}'.format(wavelength_idx, polarization_idx + 1, plane_type)
        else:
            base_filename = 'field_enhancement_pol{}_{}'.format(polarization_idx + 1, plane_type)
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved_files

    def _plot_field_intensity(self,
            field_data: Dict[str, Any],
            polarization_idx: int,
            wavelength_idx: Optional[int] = None) -> List[str]:

        intensity = field_data['intensity']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']

        if not isinstance(intensity, np.ndarray):
            intensity = np.array([[intensity]])
        elif intensity.ndim == 0:
            intensity = np.array([[intensity.item()]])
        elif intensity.ndim == 1:
            intensity = intensity.reshape(1, -1)

        if np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        n_unique_x = len(np.unique(x_grid))
        n_unique_y = len(np.unique(y_grid))
        if intensity.shape == (n_unique_x, n_unique_y):
            intensity = intensity.T

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        pol_label = self._get_polarization_label(polarization_idx)

        intensity_masked = np.ma.masked_invalid(intensity)

        fig, ax = plt.subplots(figsize = (9, 7))

        valid_data = intensity_masked.compressed()
        if len(valid_data) > 0 and np.any(valid_data > 0):
            positive_data = valid_data[valid_data > 0]
            if len(positive_data) > 0:
                vmin_log = max(np.percentile(positive_data, 2), 1e-10)
                vmax_log = np.percentile(positive_data, 99.5)

                if vmin_log >= vmax_log:
                    vmin_log = vmax_log / 1000

                im = ax.imshow(intensity_masked, extent = extent, origin = 'lower',
                              cmap = 'inferno', aspect = 'auto',
                              norm = LogNorm(vmin = vmin_log, vmax = vmax_log))
            else:
                im = ax.imshow(intensity_masked, extent = extent, origin = 'lower',
                              cmap = 'inferno', aspect = 'auto')
        else:
            im = ax.imshow(intensity_masked, extent = extent, origin = 'lower',
                          cmap = 'inferno', aspect = 'auto')

        ax.set_xlabel(x_label, fontsize = 11)
        ax.set_ylabel(y_label, fontsize = 11)
        ax.set_title('Field Intensity |E|² (Log Scale)\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                    fontsize = 12, fontweight = 'bold')

        cbar = plt.colorbar(im, ax = ax)
        cbar.set_label('|E|² (a.u.)', fontsize = 11)

        z_plane = float(z_grid.flat[0])
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax, section, plane_type)

        plt.tight_layout()

        if wavelength_idx is not None:
            base_filename = 'field_intensity_wl{}_pol{}_{}'.format(wavelength_idx, polarization_idx + 1, plane_type)
        else:
            base_filename = 'field_intensity_pol{}_{}'.format(polarization_idx + 1, plane_type)
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved_files

    def _plot_field_vectors(self,
            field_data: Dict[str, Any],
            polarization_idx: int,
            wavelength_idx: Optional[int] = None) -> List[str]:

        e_total = field_data.get('e_total')
        if e_total is None:
            return []

        if not isinstance(e_total, np.ndarray):
            return []
        if e_total.ndim < 3:
            return []
        if e_total.shape[-1] < 3:
            return []

        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        enhancement = field_data['enhancement']

        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([[x_grid]])
        if not isinstance(y_grid, np.ndarray):
            y_grid = np.array([[y_grid]])
        if not isinstance(z_grid, np.ndarray):
            z_grid = np.array([[z_grid]])

        if x_grid.ndim == 0:
            x_grid = np.array([[x_grid.item()]])
        if y_grid.ndim == 0:
            y_grid = np.array([[y_grid.item()]])
        if z_grid.ndim == 0:
            z_grid = np.array([[z_grid.item()]])

        if x_grid.ndim == 1:
            x_grid = x_grid.reshape(1, -1)
        if y_grid.ndim == 1:
            y_grid = y_grid.reshape(1, -1)
        if z_grid.ndim == 1:
            z_grid = z_grid.reshape(1, -1)

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        pol_label = self._get_polarization_label(polarization_idx)

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)

        if plane_type == 'xz':
            x_coord = x_grid[0, :]
            z_coord = z_grid[:, 0]
            e_x = e_total[:, :, 0].real
            e_z = e_total[:, :, 2].real
            x_plot = x_coord
            y_plot = z_coord
            U = e_x
            V = e_z

        elif plane_type == 'xy':
            x_coord = x_grid[0, :]
            y_coord = y_grid[:, 0]
            e_x = e_total[:, :, 0].real
            e_y = e_total[:, :, 1].real
            x_plot = x_coord
            y_plot = y_coord
            U = e_x
            V = e_y

        elif plane_type == 'yz':
            y_coord = y_grid[:, 0]
            z_coord = z_grid[0, :]
            e_y = e_total[:, :, 1].real
            e_z = e_total[:, :, 2].real
            x_plot = y_coord
            y_plot = z_coord
            U = e_y
            V = e_z

        else:
            return []

        # Downsample for vector plot
        nx, ny = U.shape
        skip_x = max(1, nx // 15)
        skip_y = max(1, ny // 15)

        x_down = x_plot[::skip_y]
        y_down = y_plot[::skip_x]

        X, Y = np.meshgrid(x_down, y_down)

        U_down = U[::skip_x, ::skip_y]
        V_down = V[::skip_x, ::skip_y]

        magnitude = np.sqrt(U_down**2 + V_down**2)
        magnitude_max = np.max(magnitude)

        if magnitude_max > 1e-10:
            U_norm = U_down / (magnitude + 1e-10)
            V_norm = V_down / (magnitude + 1e-10)
        else:
            U_norm = U_down
            V_norm = V_down

        fig, ax = plt.subplots(figsize = (10, 8))

        enhancement_masked = np.ma.masked_invalid(enhancement)

        im = ax.imshow(enhancement_masked, extent = extent, origin = 'lower',
                      cmap = 'hot', aspect = 'auto', alpha = 0.7)

        q = ax.quiver(X, Y, U_norm, V_norm, magnitude,
                      cmap = 'cool', scale = 25, width = 0.004,
                      alpha = 0.9, pivot = 'middle')

        ax.set_xlabel(x_label, fontsize = 11)
        ax.set_ylabel(y_label, fontsize = 11)
        ax.set_title('Electric Field Vectors\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                    fontsize = 12, fontweight = 'bold')

        cbar1 = plt.colorbar(im, ax = ax, pad = 0.12, label = '|E|²/|E₀|²')
        cbar2 = plt.colorbar(q, ax = ax, label = 'Field Magnitude')

        plt.tight_layout()

        if wavelength_idx is not None:
            base_filename = 'field_vectors_wl{}_pol{}_{}'.format(wavelength_idx, polarization_idx + 1, plane_type)
        else:
            base_filename = 'field_vectors_pol{}_{}'.format(polarization_idx + 1, plane_type)
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved_files

    def plot_field_separate_internal_external(self,
            field_data: List[Dict[str, Any]]) -> List[str]:

        if not field_data:
            return []

        saved_files = []

        spheres = get_sphere_boundaries_from_config(self.config)
        if spheres and self.verbose:
            print('  Edge artifact filter: {} sphere(s) detected'.format(len(spheres)))

        for idx, field in enumerate(field_data):
            has_separate = (
                'enhancement_ext' in field and
                'enhancement_int' in field and
                field['enhancement_ext'] is not None and
                field['enhancement_int'] is not None
            )

            if not has_separate:
                continue

            pol_idx = field.get('polarization_idx', idx)
            wl_idx = field.get('wavelength_idx')

            if self.verbose:
                print('  Creating separate int/ext plots for pol {}...'.format(pol_idx + 1))

            artifact_mask = self._compute_artifact_mask(field, spheres)

            sep_files = self._plot_field_separate(field, pol_idx, wl_idx, artifact_mask)
            if sep_files:
                saved_files.extend(sep_files)

            comp_files = self._plot_field_comparison(field, pol_idx, wl_idx, artifact_mask)
            if comp_files:
                saved_files.extend(comp_files)

            overlay_files = self._plot_field_overlay(field, pol_idx, wl_idx, artifact_mask)
            if overlay_files:
                saved_files.extend(overlay_files)

        return saved_files

    def _compute_artifact_mask(self,
            field: Dict[str, Any],
            spheres: Optional[List[Tuple[float, float, float, float]]]) -> Optional[np.ndarray]:

        if spheres is None or len(spheres) == 0:
            return None

        detection_key = None
        for candidate in ['enhancement_int', 'intensity_int', 'e_sq_int']:
            if candidate in field and field[candidate] is not None:
                detection_key = candidate
                break

        if detection_key is None:
            return None

        det_data = np.array(field[detection_key], dtype = float)
        if np.iscomplexobj(det_data):
            det_data = np.abs(det_data)

        x_grid = np.atleast_2d(np.asarray(field['x_grid'], dtype = float))
        y_grid = np.atleast_2d(np.asarray(field['y_grid'], dtype = float))
        z_grid = np.atleast_2d(np.asarray(field['z_grid'], dtype = float))

        if x_grid.shape != det_data.shape and x_grid.size == det_data.size:
            x_grid = x_grid.reshape(det_data.shape)
            y_grid = y_grid.reshape(det_data.shape)
            z_grid = z_grid.reshape(det_data.shape)

        int_mask = ~np.isnan(det_data) & np.isfinite(det_data)

        artifact_mask, n_artifacts = find_edge_artifacts(
            det_data, x_grid, y_grid, z_grid, spheres,
            mask = int_mask,
            edge_threshold = 1.0,
            isolation_ratio = 1.3,
            verbose = self.verbose
        )

        if self.verbose and n_artifacts > 0:
            print('    Edge filter: {} artifact pixels will be removed '
                  'from internal + merged plots (detected from {})'.format(n_artifacts, detection_key))

        return artifact_mask

    def _plot_field_separate(self,
            field_data: Dict[str, Any],
            polarization_idx: int,
            wavelength_idx: Optional[int] = None,
            artifact_mask: Optional[np.ndarray] = None) -> List[str]:

        saved_files = []

        enhancement_ext = np.array(field_data['enhancement_ext'])
        enhancement_int = np.array(field_data['enhancement_int'])
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']

        if np.iscomplexobj(enhancement_ext):
            enhancement_ext = np.abs(enhancement_ext)
        if np.iscomplexobj(enhancement_int):
            enhancement_int = np.abs(enhancement_int)

        if artifact_mask is not None and artifact_mask.shape == enhancement_int.shape:
            enhancement_int = enhancement_int.copy()
            enhancement_int[artifact_mask] = np.nan

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        pol_label = self._get_polarization_label(polarization_idx)

        enh_ext_masked = np.ma.masked_invalid(enhancement_ext)
        enh_int_masked = np.ma.masked_invalid(enhancement_int)

        valid_ext = enh_ext_masked.compressed()
        valid_int = enh_int_masked.compressed()

        if len(valid_ext) > 0 and len(valid_int) > 0:
            vmin = min(np.percentile(valid_ext, 1), np.percentile(valid_int, 1))
            vmax = max(np.percentile(valid_ext, 99), np.percentile(valid_int, 99))
        elif len(valid_ext) > 0:
            vmin = np.percentile(valid_ext, 1)
            vmax = np.percentile(valid_ext, 99)
        elif len(valid_int) > 0:
            vmin = np.percentile(valid_int, 1)
            vmax = np.percentile(valid_int, 99)
        else:
            vmin, vmax = 0, 1

        fig, axes = plt.subplots(1, 2, figsize = (16, 6))

        # Plot 1: External field
        im1 = axes[0].imshow(enh_ext_masked, extent = extent, origin = 'lower',
                            cmap = 'hot', aspect = 'auto', vmin = vmin, vmax = vmax)
        axes[0].set_xlabel(x_label, fontsize = 11)
        axes[0].set_ylabel(y_label, fontsize = 11)
        axes[0].set_title('External Field Only\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                         fontsize = 12, fontweight = 'bold')

        cbar1 = plt.colorbar(im1, ax = axes[0])
        cbar1.set_label('|E|²/|E₀|²', fontsize = 11)

        z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(axes[0], section, plane_type)

        n_valid_ext = np.sum(np.isfinite(enhancement_ext))
        axes[0].text(0.02, 0.98, 'Valid: {} pts'.format(n_valid_ext),
                    transform = axes[0].transAxes, fontsize = 9,
                    verticalalignment = 'top',
                    bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))

        # Plot 2: Internal field
        im2 = axes[1].imshow(enh_int_masked, extent = extent, origin = 'lower',
                            cmap = 'hot', aspect = 'auto', vmin = vmin, vmax = vmax)
        axes[1].set_xlabel(x_label, fontsize = 11)
        axes[1].set_ylabel(y_label, fontsize = 11)
        axes[1].set_title('Internal Field Only\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                         fontsize = 12, fontweight = 'bold')

        cbar2 = plt.colorbar(im2, ax = axes[1])
        cbar2.set_label('|E|²/|E₀|²', fontsize = 11)

        for section in sections:
            self._draw_material_boundary(axes[1], section, plane_type)

        n_valid_int = np.sum(np.isfinite(enhancement_int))
        axes[1].text(0.02, 0.98, 'Valid: {} pts'.format(n_valid_int),
                    transform = axes[1].transAxes, fontsize = 9,
                    verticalalignment = 'top',
                    bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))

        plt.tight_layout()

        if wavelength_idx is not None:
            base_filename = 'field_enhancement_separate_wl{}_pol{}_{}'.format(wavelength_idx, polarization_idx + 1, plane_type)
        else:
            base_filename = 'field_enhancement_separate_pol{}_{}'.format(polarization_idx + 1, plane_type)

        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def _plot_field_comparison(self,
            field_data: Dict[str, Any],
            polarization_idx: int,
            wavelength_idx: Optional[int] = None,
            artifact_mask: Optional[np.ndarray] = None) -> List[str]:

        saved_files = []

        enhancement_ext = np.array(field_data['enhancement_ext'])
        enhancement_int = np.array(field_data['enhancement_int'])
        enhancement_merged = np.array(field_data['enhancement'])
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']

        if np.iscomplexobj(enhancement_ext):
            enhancement_ext = np.abs(enhancement_ext)
        if np.iscomplexobj(enhancement_int):
            enhancement_int = np.abs(enhancement_int)
        if np.iscomplexobj(enhancement_merged):
            enhancement_merged = np.abs(enhancement_merged)

        if artifact_mask is not None:
            if artifact_mask.shape == enhancement_int.shape:
                enhancement_int = enhancement_int.copy()
                enhancement_int[artifact_mask] = np.nan
            if artifact_mask.shape == enhancement_merged.shape:
                enhancement_merged = enhancement_merged.copy()
                enhancement_merged[artifact_mask] = np.nan

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        pol_label = self._get_polarization_label(polarization_idx)

        enh_ext_masked = np.ma.masked_invalid(enhancement_ext)
        enh_int_masked = np.ma.masked_invalid(enhancement_int)
        enh_merged_masked = np.ma.masked_invalid(enhancement_merged)

        valid_data = []
        for enh in [enh_ext_masked, enh_int_masked, enh_merged_masked]:
            compressed = enh.compressed()
            if len(compressed) > 0:
                valid_data.extend(compressed)

        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 1)
            vmax = np.percentile(valid_data, 99)
        else:
            vmin, vmax = 0, 1

        fig, axes = plt.subplots(1, 3, figsize = (22, 6))

        titles = ['External Field Only', 'Internal Field Only', 'Merged (Combined)']
        enhancements = [enh_ext_masked, enh_int_masked, enh_merged_masked]

        z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
        sections = self.geometry.get_cross_section(z_plane)

        for idx, (ax, title, enh) in enumerate(zip(axes, titles, enhancements)):
            im = ax.imshow(enh, extent = extent, origin = 'lower',
                          cmap = 'hot', aspect = 'auto', vmin = vmin, vmax = vmax)

            ax.set_xlabel(x_label, fontsize = 11)
            ax.set_ylabel(y_label, fontsize = 11)
            ax.set_title('{}\nλ = {:.1f} nm, {}'.format(title, wavelength, pol_label),
                        fontsize = 11, fontweight = 'bold')

            cbar = plt.colorbar(im, ax = ax)
            cbar.set_label('|E|²/|E₀|²', fontsize = 10)

            for section in sections:
                self._draw_material_boundary(ax, section, plane_type)

            n_valid = np.sum(np.isfinite(enh))
            ax.text(0.02, 0.98, 'Valid: {} pts'.format(n_valid),
                   transform = ax.transAxes, fontsize = 9,
                   verticalalignment = 'top',
                   bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))

        plt.tight_layout()

        if wavelength_idx is not None:
            base_filename = 'field_enhancement_comparison_wl{}_pol{}_{}'.format(wavelength_idx, polarization_idx + 1, plane_type)
        else:
            base_filename = 'field_enhancement_comparison_pol{}_{}'.format(polarization_idx + 1, plane_type)

        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def _plot_field_overlay(self,
            field_data: Dict[str, Any],
            polarization_idx: int,
            wavelength_idx: Optional[int] = None,
            artifact_mask: Optional[np.ndarray] = None) -> List[str]:

        saved_files = []

        enhancement_ext = np.array(field_data['enhancement_ext'])
        enhancement_int = np.array(field_data['enhancement_int'])
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']

        if np.iscomplexobj(enhancement_ext):
            enhancement_ext = np.abs(enhancement_ext)
        if np.iscomplexobj(enhancement_int):
            enhancement_int = np.abs(enhancement_int)

        if artifact_mask is not None and artifact_mask.shape == enhancement_int.shape:
            enhancement_int = enhancement_int.copy()
            enhancement_int[artifact_mask] = np.nan

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        pol_label = self._get_polarization_label(polarization_idx)

        fig, ax = plt.subplots(figsize = (12, 9))

        enh_ext_masked = np.ma.masked_invalid(enhancement_ext)
        valid_ext = enh_ext_masked.compressed()

        if len(valid_ext) > 0:
            vmin_ext = np.percentile(valid_ext, 1)
            vmax_ext = np.percentile(valid_ext, 99)
        else:
            vmin_ext, vmax_ext = 0, 1

        im_ext = ax.imshow(enh_ext_masked, extent = extent, origin = 'lower',
                          cmap = 'hot', aspect = 'auto',
                          vmin = vmin_ext, vmax = vmax_ext, alpha = 0.7)

        mask_int = np.isfinite(enhancement_int) & (enhancement_int > 0)

        if np.any(mask_int):
            if plane_type == 'xy':
                x_int = x_grid[mask_int]
                y_int = y_grid[mask_int]
            elif plane_type == 'xz':
                x_int = x_grid[mask_int]
                y_int = z_grid[mask_int]
            elif plane_type == 'yz':
                x_int = y_grid[mask_int]
                y_int = z_grid[mask_int]
            else:
                x_int = x_grid[mask_int]
                y_int = y_grid[mask_int]

            values_int = enhancement_int[mask_int]

            vmin_int = np.min(values_int)
            vmax_int = np.max(values_int)

            scatter = ax.scatter(x_int, y_int, c = values_int,
                               cmap = 'viridis', s = 20,
                               vmin = vmin_int, vmax = vmax_int,
                               edgecolors = 'black', linewidth = 0.3, alpha = 0.9,
                               label = 'Internal ({} pts)'.format(len(x_int)))

            cbar_int = plt.colorbar(scatter, ax = ax, pad = 0.12)
            cbar_int.set_label('|E|²/|E₀|² (Internal)', fontsize = 11)

        cbar_ext = plt.colorbar(im_ext, ax = ax)
        cbar_ext.set_label('|E|²/|E₀|² (External)', fontsize = 11)

        z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax, section, plane_type)

        ax.set_xlabel(x_label, fontsize = 11)
        ax.set_ylabel(y_label, fontsize = 11)
        ax.set_title('Internal (scatter) + External (heatmap) Fields\nλ = {:.1f} nm, {}'.format(wavelength, pol_label),
                    fontsize = 12, fontweight = 'bold')

        if np.any(mask_int):
            ax.legend(loc = 'upper right', fontsize = 10)

        plt.tight_layout()

        if wavelength_idx is not None:
            base_filename = 'field_enhancement_overlay_wl{}_pol{}_{}'.format(wavelength_idx, polarization_idx + 1, plane_type)
        else:
            base_filename = 'field_enhancement_overlay_pol{}_{}'.format(polarization_idx + 1, plane_type)

        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def _determine_plane(self,
            x_grid: Any,
            y_grid: Any,
            z_grid: Any) -> Tuple[str, List[float], str, str]:

        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([[x_grid]])
        if not isinstance(y_grid, np.ndarray):
            y_grid = np.array([[y_grid]])
        if not isinstance(z_grid, np.ndarray):
            z_grid = np.array([[z_grid]])

        if x_grid.ndim == 0:
            x_grid = np.array([[x_grid.item()]])
        if y_grid.ndim == 0:
            y_grid = np.array([[y_grid.item()]])
        if z_grid.ndim == 0:
            z_grid = np.array([[z_grid.item()]])

        if x_grid.ndim == 1:
            x_grid = x_grid.reshape(1, -1)
        if y_grid.ndim == 1:
            y_grid = y_grid.reshape(1, -1)
        if z_grid.ndim == 1:
            z_grid = z_grid.reshape(1, -1)

        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1

        if y_constant:
            plane_type = 'xz'
            x_min, x_max = x_grid.min(), x_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if z_min == z_max:
                z_min -= 0.5
                z_max += 0.5
            extent = [x_min, x_max, z_min, z_max]
            x_label = 'x (nm)'
            y_label = 'z (nm)'
        elif z_constant:
            plane_type = 'xy'
            x_min, x_max = x_grid.min(), x_grid.max()
            y_min, y_max = y_grid.min(), y_grid.max()
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            extent = [x_min, x_max, y_min, y_max]
            x_label = 'x (nm)'
            y_label = 'y (nm)'
        elif x_constant:
            plane_type = 'yz'
            y_min, y_max = y_grid.min(), y_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            if z_min == z_max:
                z_min -= 0.5
                z_max += 0.5
            extent = [y_min, y_max, z_min, z_max]
            x_label = 'y (nm)'
            y_label = 'z (nm)'
        else:
            plane_type = '3d'
            x_min, x_max = x_grid.min(), x_grid.max()
            y_min, y_max = y_grid.min(), y_grid.max()
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            extent = [x_min, x_max, y_min, y_max]
            x_label = 'x (nm)'
            y_label = 'y (nm)'

        return plane_type, extent, x_label, y_label

    def _is_2d_slice(self,
            field_data: Dict[str, Any]) -> bool:

        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']

        if not isinstance(x_grid, np.ndarray):
            return False

        if x_grid.ndim == 0:
            return False

        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1

        return sum([x_constant, y_constant, z_constant]) == 1

    def _save_figure(self,
            fig: Any,
            base_filename: str) -> List[str]:

        saved_files = []

        for fmt in self.plot_format:
            filepath = os.path.join(self.output_dir, '{}.{}'.format(base_filename, fmt))
            fig.savefig(filepath, dpi = self.dpi, bbox_inches = 'tight')
            saved_files.append(filepath)

            if self.verbose:
                print('  Saved: {}'.format(filepath))

        return saved_files

    def _draw_material_boundary(self,
            ax: Any,
            section: Dict[str, Any],
            plane_type: str) -> None:

        if section['type'] == 'circle':
            center = section['center']
            radius = section['radius']

            circle = Circle(
                center,
                radius,
                fill = False,
                edgecolor = 'gray',
                linestyle = '--',
                linewidth = 2,
                label = section.get('label', 'Material boundary')
            )
            ax.add_patch(circle)

        elif section['type'] == 'rectangle':
            bounds = section['bounds']  # [x_min, x_max, y_min, y_max]
            x_min, x_max, y_min, y_max = bounds

            width = x_max - x_min
            height = y_max - y_min

            rectangle = Rectangle(
                (x_min, y_min),
                width,
                height,
                fill = False,
                edgecolor = 'gray',
                linestyle = '--',
                linewidth = 2,
                label = section.get('label', 'Material boundary')
            )
            ax.add_patch(rectangle)

    # ========================================================================
    # UNPOLARIZED PLOTS
    # ========================================================================

    def plot_unpolarized_spectrum(self,
            data: Dict[str, Any],
            analysis_results: Dict[str, Any]) -> List[str]:

        saved_files = []
        unpol = analysis_results['unpolarized_spectrum']

        wavelength = unpol['wavelength']
        unpol_ext = unpol['extinction']
        unpol_sca = unpol['scattering']
        unpol_abs = unpol['absorption']

        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')

        if xaxis_unit == 'energy':
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            xdata = xdata[::-1]
            unpol_ext = unpol_ext[::-1]
            unpol_sca = unpol_sca[::-1]
            unpol_abs = unpol_abs[::-1]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'

        # ========== 1. Standalone Unpolarized Spectrum ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        ax.plot(xdata, unpol_ext, 'b-', linewidth = 2, label = 'Extinction')
        ax.plot(xdata, unpol_sca, 'r--', linewidth = 2, label = 'Scattering')
        ax.plot(xdata, unpol_abs, 'g:', linewidth = 2, label = 'Absorption')

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Cross Section (nm²)', fontsize = 12)
        ax.set_title('Unpolarized Spectrum\n(FDTD-style incoherent average, {} polarizations)'.format(unpol['n_averaged']),
                    fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 11)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()

        base_filename = 'simulation_spectrum_unpolarized'
        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== 2. Comparison: All polarizations + Unpolarized ==========
        comparison_files = self._plot_spectrum_comparison_with_unpolarized(data, analysis_results)
        saved_files.extend(comparison_files)

        return saved_files

    def _plot_spectrum_comparison_with_unpolarized(self,
            data: Dict[str, Any],
            analysis_results: Dict[str, Any]) -> List[str]:

        saved_files = []

        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        n_pol = extinction.shape[1]

        unpol = analysis_results['unpolarized_spectrum']

        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')
        if xaxis_unit == 'energy':
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            xdata = xdata[::-1]
            extinction = extinction[::-1, :]
            scattering = scattering[::-1, :]
            absorption = absorption[::-1, :]
            unpol_ext = unpol['extinction'][::-1]
            unpol_sca = unpol['scattering'][::-1]
            unpol_abs = unpol['absorption'][::-1]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'
            unpol_ext = unpol['extinction']
            unpol_sca = unpol['scattering']
            unpol_abs = unpol['absorption']

        colors = plt.cm.tab10(np.linspace(0, 0.7, n_pol))

        # ========== Extinction Comparison ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, extinction[:, i], color = colors[i], linewidth = 1.5,
                   linestyle = '--', alpha = 0.7, label = pol_label)

        ax.plot(xdata, unpol_ext, 'k-', linewidth = 2.5, label = 'Unpolarized')

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize = 12)
        ax.set_title('Extinction: Polarizations vs Unpolarized', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()
        files = self._save_figure(fig, 'simulation_comparison_extinction_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== Scattering Comparison ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, scattering[:, i], color = colors[i], linewidth = 1.5,
                   linestyle = '--', alpha = 0.7, label = pol_label)

        ax.plot(xdata, unpol_sca, 'k-', linewidth = 2.5, label = 'Unpolarized')

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Scattering Cross Section (nm²)', fontsize = 12)
        ax.set_title('Scattering: Polarizations vs Unpolarized', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()
        files = self._save_figure(fig, 'simulation_comparison_scattering_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== Absorption Comparison ==========
        fig, ax = plt.subplots(figsize = (10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, absorption[:, i], color = colors[i], linewidth = 1.5,
                   linestyle = '--', alpha = 0.7, label = pol_label)

        ax.plot(xdata, unpol_abs, 'k-', linewidth = 2.5, label = 'Unpolarized')

        ax.set_xlabel(xlabel_text, fontsize = 12)
        ax.set_ylabel('Absorption Cross Section (nm²)', fontsize = 12)
        ax.set_title('Absorption: Polarizations vs Unpolarized', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()
        files = self._save_figure(fig, 'simulation_comparison_absorption_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== All-in-one Comparison (3 subplots) ==========
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            axes[0].plot(xdata, extinction[:, i], color = colors[i], linewidth = 1.5,
                        linestyle = '--', alpha = 0.7, label = pol_label)
            axes[1].plot(xdata, scattering[:, i], color = colors[i], linewidth = 1.5,
                        linestyle = '--', alpha = 0.7, label = pol_label)
            axes[2].plot(xdata, absorption[:, i], color = colors[i], linewidth = 1.5,
                        linestyle = '--', alpha = 0.7, label = pol_label)

        axes[0].plot(xdata, unpol_ext, 'k-', linewidth = 2.5, label = 'Unpolarized')
        axes[1].plot(xdata, unpol_sca, 'k-', linewidth = 2.5, label = 'Unpolarized')
        axes[2].plot(xdata, unpol_abs, 'k-', linewidth = 2.5, label = 'Unpolarized')

        titles = ['Extinction', 'Scattering', 'Absorption']
        for idx, ax in enumerate(axes):
            ax.set_xlabel(xlabel_text, fontsize = 11)
            ax.set_ylabel('Cross Section (nm²)', fontsize = 11)
            ax.set_title(titles[idx], fontsize = 12, fontweight = 'bold')
            ax.legend(fontsize = 8)
            ax.grid(True, alpha = 0.3)
            if xaxis_unit == 'energy':
                ax.invert_xaxis()

        plt.suptitle('Polarizations vs Unpolarized Comparison', fontsize = 14, fontweight = 'bold')
        plt.tight_layout()

        files = self._save_figure(fig, 'simulation_comparison_all_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def plot_unpolarized_fields(self,
            data: Dict[str, Any],
            analysis_results: Dict[str, Any]) -> List[str]:

        saved_files = []
        fields = data.get('fields', [])

        if not fields:
            return saved_files

        unpol_info = analysis_results.get('unpolarized', {})
        method = unpol_info.get('method', '')

        if method == 'orthogonal_2pol_average':
            expected_n_pol = 2
        elif method == 'orthogonal_3dir_average':
            expected_n_pol = 3
        else:
            return saved_files

        # Group fields by wavelength
        fields_by_wavelength = {}
        for field in fields:
            wl = field.get('wavelength', 0)
            wl_key = '{:.1f}'.format(wl)
            if wl_key not in fields_by_wavelength:
                fields_by_wavelength[wl_key] = []
            fields_by_wavelength[wl_key].append(field)

        for wl_key, wl_fields in fields_by_wavelength.items():
            if len(wl_fields) != expected_n_pol:
                continue

            wl_fields_sorted = sorted(wl_fields, key = lambda f: f.get('polarization_idx', 0))

            ref_field = wl_fields_sorted[0]
            x_grid = ref_field.get('x_grid')
            y_grid = ref_field.get('y_grid')
            z_grid = ref_field.get('z_grid')
            wavelength = ref_field.get('wavelength', 0)

            # Calculate unpolarized (incoherent average)
            # NOTE: MATLAB Universal Reference method stores INTENSITY enhancement (|E|^2/|E0|^2),
            # not field enhancement (|E|/|E0|). For intensity enhancement, we use arithmetic mean.
            enhancements = []
            intensities = []

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

            # Incoherent average for intensity enhancement (|E|^2/|E0|^2)
            unpol_enh = np.mean(enhancements, axis = 0)
            unpol_intensity = np.mean(intensities, axis = 0) if len(intensities) == expected_n_pol else unpol_enh

            unpol_field = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid,
                'enhancement': unpol_enh,
                'intensity': unpol_intensity,
                'wavelength': wavelength
            }

            unpol_files = self._plot_unpolarized_field_enhancement(
                unpol_field, wavelength, expected_n_pol
            )
            saved_files.extend(unpol_files)

            unpol_intensity_files = self._plot_unpolarized_field_intensity(
                unpol_field, wavelength, expected_n_pol
            )
            saved_files.extend(unpol_intensity_files)

            comparison_files = self._plot_field_comparison_with_unpolarized(
                wl_fields_sorted, unpol_field, wavelength
            )
            saved_files.extend(comparison_files)

        return saved_files

    def _plot_unpolarized_field_enhancement(self,
            unpol_field: Dict[str, Any],
            wavelength: float,
            n_pol: int) -> List[str]:

        saved_files = []

        enhancement = unpol_field['enhancement']
        x_grid = unpol_field['x_grid']
        y_grid = unpol_field['y_grid']
        z_grid = unpol_field['z_grid']

        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([[enhancement]])
        elif enhancement.ndim == 0:
            enhancement = np.array([[enhancement.item()]])
        elif enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)

        n_unique_x = len(np.unique(x_grid))
        n_unique_y = len(np.unique(y_grid))
        if enhancement.shape == (n_unique_x, n_unique_y):
            enhancement = enhancement.T

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        enhancement_masked = np.ma.masked_invalid(enhancement)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

        valid_data = enhancement_masked.compressed()
        if len(valid_data) > 0:
            vmin_linear = np.percentile(valid_data, 1)
            vmax_linear = np.percentile(valid_data, 99)
        else:
            vmin_linear, vmax_linear = 0, 1

        # Linear scale
        im1 = ax1.imshow(enhancement_masked, extent = extent, origin = 'lower',
                        cmap = 'hot', aspect = 'auto', vmin = vmin_linear, vmax = vmax_linear)
        ax1.set_xlabel(x_label, fontsize = 11)
        ax1.set_ylabel(y_label, fontsize = 11)
        ax1.set_title('Unpolarized Intensity Enhancement (Linear)\n'
                     'λ = {:.1f} nm, avg of {} pols'.format(wavelength, n_pol),
                     fontsize = 11, fontweight = 'bold')
        cbar1 = plt.colorbar(im1, ax = ax1)
        cbar1.set_label('|E|²/|E₀|²', fontsize = 11)

        z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax1, section, plane_type)

        # Log scale
        if len(valid_data) > 0 and np.any(valid_data > 0):
            positive_data = valid_data[valid_data > 0]
            vmin_log = max(np.percentile(positive_data, 5), 0.1)
            vmax_log = np.percentile(positive_data, 99.5)
            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 100

            im2 = ax2.imshow(enhancement_masked, extent = extent, origin = 'lower',
                            cmap = 'hot', aspect = 'auto',
                            norm = LogNorm(vmin = vmin_log, vmax = vmax_log))
        else:
            im2 = ax2.imshow(enhancement_masked, extent = extent, origin = 'lower',
                            cmap = 'hot', aspect = 'auto')

        ax2.set_xlabel(x_label, fontsize = 11)
        ax2.set_ylabel(y_label, fontsize = 11)
        ax2.set_title('Unpolarized Intensity Enhancement (Log)\n'
                     'λ = {:.1f} nm, avg of {} pols'.format(wavelength, n_pol),
                     fontsize = 11, fontweight = 'bold')
        cbar2 = plt.colorbar(im2, ax = ax2)
        cbar2.set_label('|E|²/|E₀|²', fontsize = 11)

        for section in sections:
            self._draw_material_boundary(ax2, section, plane_type)

        plt.tight_layout()

        base_filename = 'field_enhancement_unpolarized_{}'.format(plane_type)
        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def _plot_unpolarized_field_intensity(self,
            unpol_field: Dict[str, Any],
            wavelength: float,
            n_pol: int) -> List[str]:

        saved_files = []

        intensity = unpol_field['intensity']
        x_grid = unpol_field['x_grid']
        y_grid = unpol_field['y_grid']
        z_grid = unpol_field['z_grid']

        if not isinstance(intensity, np.ndarray):
            intensity = np.array([[intensity]])
        elif intensity.ndim == 0:
            intensity = np.array([[intensity.item()]])
        elif intensity.ndim == 1:
            intensity = intensity.reshape(1, -1)

        if np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        n_unique_x = len(np.unique(x_grid))
        n_unique_y = len(np.unique(y_grid))
        if intensity.shape == (n_unique_x, n_unique_y):
            intensity = intensity.T

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        fig, ax = plt.subplots(figsize = (9, 7))

        intensity_log = np.maximum(intensity, 1e-10)
        int_max = intensity.max()
        int_min = intensity_log[intensity_log > 0].min() if np.any(intensity_log > 0) else 1e-10

        if int_max > int_min and int_max > 0:
            vmin_log = max(int_min, int_max / 1e6)
            vmax_log = int_max

            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 10

            im = ax.imshow(intensity_log, extent = extent, origin = 'lower',
                          cmap = 'viridis', aspect = 'auto',
                          norm = LogNorm(vmin = vmin_log, vmax = vmax_log))
        else:
            im = ax.imshow(intensity, extent = extent, origin = 'lower',
                          cmap = 'viridis', aspect = 'auto')

        ax.set_xlabel(x_label, fontsize = 11)
        ax.set_ylabel(y_label, fontsize = 11)
        ax.set_title('Unpolarized Field Intensity |E|² (Log Scale)\n'
                     'λ = {:.1f} nm, avg of {} pols'.format(wavelength, n_pol),
                     fontsize = 12, fontweight = 'bold')

        cbar = plt.colorbar(im, ax = ax)
        cbar.set_label('|E|² (a.u.)', fontsize = 11)

        z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax, section, plane_type)

        plt.tight_layout()

        base_filename = 'field_intensity_unpolarized_{}'.format(plane_type)
        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def _plot_field_comparison_with_unpolarized(self,
            pol_fields: List[Dict[str, Any]],
            unpol_field: Dict[str, Any],
            wavelength: float) -> List[str]:

        saved_files = []
        n_pol = len(pol_fields)

        ref_field = pol_fields[0]
        x_grid = ref_field['x_grid']
        y_grid = ref_field['y_grid']
        z_grid = ref_field['z_grid']

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        n_cols = n_pol + 1
        fig, axes = plt.subplots(1, n_cols, figsize = (5 * n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        # Find global vmin/vmax for consistent colorbar
        # NOTE: Using list + extend instead of np.concatenate per convention
        all_data = []
        for field in pol_fields:
            enh = field.get('enhancement')
            if enh is not None:
                if np.iscomplexobj(enh):
                    enh = np.abs(enh)
                all_data.append(enh.flatten())

        unpol_enh = unpol_field['enhancement']
        if np.iscomplexobj(unpol_enh):
            unpol_enh = np.abs(unpol_enh)
        all_data.append(unpol_enh.flatten())

        # Build concatenated array without np.concatenate
        total_len = sum(arr.shape[0] for arr in all_data)
        all_data_flat = np.empty(total_len, dtype = all_data[0].dtype)
        offset = 0
        for arr in all_data:
            all_data_flat[offset:offset + arr.shape[0]] = arr
            offset += arr.shape[0]

        valid_data = all_data_flat[~np.isnan(all_data_flat)]

        if len(valid_data) > 0:
            # Intensity enhancement is |E|^2/|E0|^2, must be non-negative
            vmin = max(0, np.percentile(valid_data, 1))
            vmax = np.percentile(valid_data, 99)

            if vmax <= vmin or vmax < 0.1:
                vmin = 0
                vmax = max(np.max(valid_data), 1.0)
        else:
            vmin, vmax = 0, 1

        # Plot each polarization
        for idx, field in enumerate(pol_fields):
            ax = axes[idx]
            enh = field.get('enhancement')
            pol_idx = field.get('polarization_idx', idx)

            if np.iscomplexobj(enh):
                enh = np.abs(enh)

            enh_masked = np.ma.masked_invalid(enh)

            im = ax.imshow(enh_masked, extent = extent, origin = 'lower',
                          cmap = 'hot', aspect = 'auto', vmin = vmin, vmax = vmax)

            pol_label = self._get_polarization_label(pol_idx)
            ax.set_title('{}'.format(pol_label), fontsize = 10, fontweight = 'bold')
            ax.set_xlabel(x_label, fontsize = 9)
            ax.set_ylabel(y_label, fontsize = 9)

            z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
            sections = self.geometry.get_cross_section(z_plane)
            for section in sections:
                self._draw_material_boundary(ax, section, plane_type)

        # Plot unpolarized
        ax = axes[-1]
        unpol_enh_masked = np.ma.masked_invalid(unpol_enh)

        im = ax.imshow(unpol_enh_masked, extent = extent, origin = 'lower',
                      cmap = 'hot', aspect = 'auto', vmin = vmin, vmax = vmax)

        ax.set_title('Unpolarized', fontsize = 10, fontweight = 'bold')
        ax.set_xlabel(x_label, fontsize = 9)
        ax.set_ylabel(y_label, fontsize = 9)

        for section in sections:
            self._draw_material_boundary(ax, section, plane_type)

        fig.subplots_adjust(right = 0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax = cbar_ax)
        cbar.set_label('|E|²/|E₀|²', fontsize = 11)

        plt.suptitle('Intensity Enhancement Comparison (λ = {:.1f} nm)'.format(wavelength),
                    fontsize = 12, fontweight = 'bold')

        base_filename = 'field_comparison_unpolarized_{}'.format(plane_type)
        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    # ========================================================================
    # SURFACE CHARGE VISUALIZATION
    # ========================================================================

    def plot_surface_charge(self,
            data: Dict[str, Any]) -> List[str]:

        if 'surface_charge' not in data or not data['surface_charge']:
            if self.verbose:
                print('  No surface charge data available')
            return []

        saved_files = []

        for sc_data in data['surface_charge']:
            wavelength = sc_data['wavelength']
            polarization = sc_data['polarization']
            pol_idx = sc_data.get('polarization_idx', 1)
            vertices = sc_data['vertices']
            faces = sc_data['faces']
            centroids = sc_data['centroids']
            normals = sc_data.get('normals')
            charge = sc_data['charge']

            if wavelength is None or charge is None or centroids is None:
                if self.verbose:
                    print('  Skipping incomplete surface charge data (pol={})'.format(pol_idx))
                continue

            if self.verbose:
                print('  Processing surface charge: lambda={:.1f}nm, pol={}'.format(wavelength, pol_idx))

            moments = self._calculate_moments(centroids, charge, sc_data.get('areas'))

            gap_faces = self._detect_gap_facing_faces(centroids, normals)

            for norm_method in ['linear', 'percentile', 'power']:
                files_3d = self._plot_surface_charge_3d(
                    vertices, faces, charge, wavelength, polarization,
                    pol_idx, moments, norm_method
                )
                saved_files.extend(files_3d)

            for norm_method in ['linear', 'percentile', 'power']:
                files_2d = self._plot_surface_charge_2d_8views(
                    centroids, normals, charge, wavelength, polarization,
                    pol_idx, moments, gap_faces, norm_method
                )
                saved_files.extend(files_2d)

        return saved_files

    def _calculate_moments(self,
            centroids: np.ndarray,
            charge: np.ndarray,
            areas: Optional[np.ndarray] = None) -> Dict[str, Any]:

        # Convert complex charge to real (MNPBEM returns complex surface charge)
        if np.iscomplexobj(charge):
            charge_real = np.real(charge)
        else:
            charge_real = charge

        if areas is not None:
            weights = np.abs(charge_real) * areas
        else:
            weights = np.abs(charge_real)

        total_weight = np.sum(weights) + 1e-30  # Avoid division by zero
        center = np.sum(centroids * weights[:, None], axis = 0) / total_weight

        # Dipole moment: p = integral r * sigma dS
        if areas is not None:
            dipole = np.sum(centroids * (charge_real * areas)[:, None], axis = 0)
        else:
            dipole = np.sum(centroids * charge_real[:, None], axis = 0)

        dipole_mag = np.linalg.norm(dipole)

        # Quadrupole trace (simplified): Tr(Q) = integral (3z^2 - r^2) sigma dS
        r_from_center = centroids - center
        r_sq = np.sum(r_from_center**2, axis = 1)
        z_sq = r_from_center[:, 2]**2

        if areas is not None:
            q_trace = np.sum((3 * z_sq - r_sq) * charge_real * areas)
        else:
            q_trace = np.sum((3 * z_sq - r_sq) * charge_real)

        return {
            'dipole': dipole,
            'dipole_mag': dipole_mag,
            'quadrupole_trace': q_trace,
            'center': center,
        }

    def _plot_surface_charge_3d(self,
            vertices: np.ndarray,
            faces: np.ndarray,
            charge: np.ndarray,
            wavelength: float,
            polarization: Any,
            pol_idx: int,
            moments: Dict[str, Any],
            norm_method: str = 'linear') -> List[str]:

        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize = (12, 10))
        ax = fig.add_subplot(111, projection = '3d')

        faces_clean = self._process_faces_for_plotting(faces)

        verts_tri = vertices[faces_clean]

        charge_plot = self._replicate_charge_for_split_faces(faces, charge, faces_clean)

        if np.iscomplexobj(charge_plot):
            charge_plot = np.real(charge_plot)

        charge_normalized, norm, vmin, vmax = self._normalize_charge(charge_plot, norm_method)

        poly = Poly3DCollection(verts_tri, alpha = 0.9, edgecolor = 'k', linewidth = 0.3)
        poly.set_array(charge_normalized)
        poly.set_cmap('RdBu_r')  # Red = positive, Blue = negative

        if norm is not None:
            poly.set_norm(norm)
        else:
            poly.set_clim(vmin, vmax)

        ax.add_collection3d(poly)

        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_xlabel('x (nm)', fontsize = 11)
        ax.set_ylabel('y (nm)', fontsize = 11)
        ax.set_zlabel('z (nm)', fontsize = 11)

        pol_str = self._format_vector_label(polarization)
        norm_label = {'linear': 'Linear', 'percentile': 'Percentile (95%)', 'power': 'Power (gamma=0.2)'}
        ax.set_title('Surface Charge Distribution (Plasmon Mode)\n'
                     'λ = {:.1f} nm, Pol = {}, Norm: {}'.format(wavelength, pol_str, norm_label[norm_method]),
                     fontsize = 12, fontweight = 'bold')

        cbar = plt.colorbar(poly, ax = ax, pad = 0.1, shrink = 0.8)
        cbar.set_label('Normalized Surface Charge', fontsize = 11)

        moment_text = 'Dipole: |p| = {:.2e} e·nm\n'.format(moments['dipole_mag'])
        moment_text += 'Q trace: {:.2e}'.format(moments['quadrupole_trace'])
        ax.text2D(0.02, 0.98, moment_text, transform = ax.transAxes,
                 fontsize = 10, verticalalignment = 'top',
                 bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5))

        x_range = vertices[:, 0].max() - vertices[:, 0].min()
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        z_range = vertices[:, 2].max() - vertices[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])

        plt.tight_layout()

        base_filename = 'surface_charge_3d_pol{}_lambda{:.0f}nm_{}'.format(pol_idx, wavelength, norm_method)
        saved = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved

    def _plot_surface_charge_2d_8views(self,
            centroids: np.ndarray,
            normals: Optional[np.ndarray],
            charge: np.ndarray,
            wavelength: float,
            polarization: Any,
            pol_idx: int,
            moments: Dict[str, Any],
            gap_faces: Dict[str, Any],
            norm_method: str = 'linear') -> List[str]:

        fig, axes = plt.subplots(2, 4, figsize = (24, 12))

        if np.iscomplexobj(charge):
            charge = np.real(charge)

        charge_normalized, norm, vmin, vmax = self._normalize_charge(charge, norm_method)

        standard_views = [
            ('+X view', 'x+', (1, 2), 'y (nm)', 'z (nm)'),
            ('-X view', 'x-', (1, 2), 'y (nm)', 'z (nm)'),
            ('+Y view', 'y+', (0, 2), 'x (nm)', 'z (nm)'),
            ('-Y view', 'y-', (0, 2), 'x (nm)', 'z (nm)'),
            ('+Z view', 'z+', (0, 1), 'x (nm)', 'y (nm)'),
            ('-Z view', 'z-', (0, 1), 'x (nm)', 'y (nm)'),
        ]

        scatter = None

        for idx, (view_name, direction, axes_idx, xlabel, ylabel) in enumerate(standard_views):
            ax = axes.flat[idx]

            outer_indices = self._detect_outer_surface_faces(centroids, normals, direction)

            if len(outer_indices) > 0:
                filtered_centroids = centroids[outer_indices]
                filtered_charge = charge_normalized[outer_indices]

                x_proj = filtered_centroids[:, axes_idx[0]]
                y_proj = filtered_centroids[:, axes_idx[1]]

                if norm is not None:
                    scatter = ax.scatter(x_proj, y_proj, c = filtered_charge, cmap = 'RdBu_r',
                                        s = 50, norm = norm, edgecolors = 'k', linewidth = 0.3)
                else:
                    scatter = ax.scatter(x_proj, y_proj, c = filtered_charge, cmap = 'RdBu_r',
                                        s = 50, vmin = vmin, vmax = vmax, edgecolors = 'k', linewidth = 0.3)
            else:
                ax.text(0.5, 0.5, 'No outer surface\nfaces detected', ha = 'center', va = 'center',
                       transform = ax.transAxes, fontsize = 12, color = 'gray')

            ax.set_xlabel(xlabel, fontsize = 10)
            ax.set_ylabel(ylabel, fontsize = 10)
            ax.set_title(view_name, fontsize = 11, fontweight = 'bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha = 0.3)

        # Plot Gap views (indices 6 and 7)
        gap_view_configs = [
            ('Gap+ (P1→Gap)', gap_faces.get('particle1', []), (1, 2), 'y (nm)', 'z (nm)'),
            ('Gap- (P2→Gap)', gap_faces.get('particle2', []), (1, 2), 'y (nm)', 'z (nm)'),
        ]

        for idx, (view_name, face_indices, axes_idx, xlabel, ylabel) in enumerate(gap_view_configs):
            ax = axes.flat[6 + idx]

            if len(face_indices) > 0:
                gap_centroids = centroids[face_indices]
                gap_charge = charge_normalized[face_indices]

                x_proj = gap_centroids[:, axes_idx[0]]
                y_proj = gap_centroids[:, axes_idx[1]]

                if norm is not None:
                    scatter = ax.scatter(x_proj, y_proj, c = gap_charge, cmap = 'RdBu_r',
                                        s = 80, norm = norm, edgecolors = 'k', linewidth = 0.3)
                else:
                    scatter = ax.scatter(x_proj, y_proj, c = gap_charge, cmap = 'RdBu_r',
                                        s = 80, vmin = vmin, vmax = vmax, edgecolors = 'k', linewidth = 0.3)
            else:
                ax.text(0.5, 0.5, 'No gap faces\ndetected', ha = 'center', va = 'center',
                       transform = ax.transAxes, fontsize = 12, color = 'gray')

            ax.set_xlabel(xlabel, fontsize = 10)
            ax.set_ylabel(ylabel, fontsize = 10)
            ax.set_title(view_name, fontsize = 11, fontweight = 'bold', color = 'darkred')
            ax.set_aspect('equal')
            ax.grid(True, alpha = 0.3)

        if scatter is not None:
            fig.subplots_adjust(right = 0.92)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
            cbar = fig.colorbar(scatter, cax = cbar_ax)
            cbar.set_label('Normalized Charge', fontsize = 11)

        pol_str = self._format_vector_label(polarization)
        norm_label = {'linear': 'Linear', 'percentile': 'Percentile (95%)', 'power': 'Power (gamma=0.2)'}
        fig.suptitle('Surface Charge Distribution - 8 Views\n'
                     'λ = {:.1f} nm, Pol = {}, Norm: {}'.format(wavelength, pol_str, norm_label[norm_method]),
                     fontsize = 13, fontweight = 'bold')

        moment_text = 'Dipole: p = [{:.2e}, {:.2e}, {:.2e}] e·nm\n'.format(
            moments['dipole'][0], moments['dipole'][1], moments['dipole'][2])
        moment_text += '|p| = {:.2e} e·nm\n'.format(moments['dipole_mag'])
        moment_text += 'Quadrupole trace: {:.2e}'.format(moments['quadrupole_trace'])
        fig.text(0.02, 0.02, moment_text, fontsize = 10,
                bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.7))

        plt.tight_layout(rect = [0, 0.05, 0.92, 0.96])

        base_filename = 'surface_charge_8views_pol{}_lambda{:.0f}nm_{}'.format(pol_idx, wavelength, norm_method)
        saved = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved

    def _normalize_charge(self,
            charge: np.ndarray,
            method: str = 'linear') -> Tuple[np.ndarray, Optional[Any], float, float]:

        if method == 'linear':
            charge_max = np.max(np.abs(charge))
            charge_normalized = charge / (charge_max + 1e-10)
            return charge_normalized, None, -1, 1

        elif method == 'percentile':
            percentile_95 = np.percentile(np.abs(charge), 95)
            if percentile_95 < 1e-10:
                percentile_95 = np.max(np.abs(charge))
            charge_normalized = charge / (percentile_95 + 1e-10)
            charge_normalized = np.clip(charge_normalized, -1, 1)
            return charge_normalized, None, -1, 1

        elif method == 'power':
            charge_max = np.max(np.abs(charge))
            charge_normalized = charge / (charge_max + 1e-10)
            gamma = 0.2
            charge_power = np.sign(charge_normalized) * np.abs(charge_normalized) ** gamma
            return charge_power, None, -1, 1

        else:
            raise ValueError('[error] Unknown normalization method: <{}>'.format(method))

    def _detect_gap_facing_faces(self,
            centroids: np.ndarray,
            normals: Optional[np.ndarray]) -> Dict[str, Any]:

        if normals is None:
            return {'particle1': [], 'particle2': []}

        x_coords = centroids[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        gap_center_x = (x_min + x_max) / 2

        normal_threshold = 0.5  # cos(60 deg)

        gap_dir_p1 = np.array([1, 0, 0])   # Particle 1 faces +x (towards gap)
        gap_dir_p2 = np.array([-1, 0, 0])  # Particle 2 faces -x (towards gap)

        normals_normalized = normals / (np.linalg.norm(normals, axis = 1, keepdims = True) + 1e-10)

        p1_mask = (x_coords < gap_center_x) & (np.dot(normals_normalized, gap_dir_p1) > normal_threshold)
        p2_mask = (x_coords > gap_center_x) & (np.dot(normals_normalized, gap_dir_p2) > normal_threshold)

        p1_indices = np.where(p1_mask)[0]
        p2_indices = np.where(p2_mask)[0]

        return {'particle1': p1_indices, 'particle2': p2_indices}

    def _detect_outer_surface_faces(self,
            centroids: np.ndarray,
            normals: Optional[np.ndarray],
            direction: str,
            position_threshold: Optional[float] = None) -> np.ndarray:

        if position_threshold is None:
            extent = centroids.max(axis = 0) - centroids.min(axis = 0)
            position_threshold = min(extent) * 0.15  # 15% of minimum extent
            position_threshold = max(position_threshold, 1.0)

        if normals is None:
            return self._detect_outer_surface_position_only(centroids, direction, position_threshold)

        direction_config = {
            'x+': (0, np.array([1, 0, 0]), 1),
            'x-': (0, np.array([-1, 0, 0]), -1),
            'y+': (1, np.array([0, 1, 0]), 1),
            'y-': (1, np.array([0, -1, 0]), -1),
            'z+': (2, np.array([0, 0, 1]), 1),
            'z-': (2, np.array([0, 0, -1]), -1),
        }

        if direction not in direction_config:
            raise ValueError('[error] Invalid <direction>: {}'.format(direction))

        axis, normal_dir, sign = direction_config[direction]
        coords = centroids[:, axis]

        normal_threshold = 0.5  # cos(60 deg)

        normals_normalized = normals / (np.linalg.norm(normals, axis = 1, keepdims = True) + 1e-10)

        if sign > 0:
            coord_max = coords.max()
            position_mask = coords >= (coord_max - position_threshold)
        else:
            coord_min = coords.min()
            position_mask = coords <= (coord_min + position_threshold)

        normal_mask = np.dot(normals_normalized, normal_dir) > normal_threshold

        final_mask = position_mask & normal_mask
        indices = np.where(final_mask)[0]

        return indices

    def _detect_outer_surface_position_only(self,
            centroids: np.ndarray,
            direction: str,
            position_threshold: Optional[float] = None) -> np.ndarray:

        if position_threshold is None:
            extent = centroids.max(axis = 0) - centroids.min(axis = 0)
            position_threshold = min(extent) * 0.15
            position_threshold = max(position_threshold, 1.0)

        direction_to_axis = {
            'x+': (0, 1), 'x-': (0, -1),
            'y+': (1, 1), 'y-': (1, -1),
            'z+': (2, 1), 'z-': (2, -1),
        }

        axis, sign = direction_to_axis[direction]
        coords = centroids[:, axis]

        if sign > 0:
            coord_extreme = coords.max()
            mask = coords >= (coord_extreme - position_threshold)
        else:
            coord_extreme = coords.min()
            mask = coords <= (coord_extreme + position_threshold)

        return np.where(mask)[0]

    def _process_faces_for_plotting(self,
            faces: np.ndarray) -> np.ndarray:

        faces_clean = []

        # MATLAB uses 1-based indexing, convert to 0-based
        faces_0based = faces - 1

        for face in faces_0based:
            if faces.shape[1] == 4:
                if not np.isnan(face[3]):
                    # Quadrilateral - split into two triangles
                    faces_clean.append([int(face[0]), int(face[1]), int(face[2])])
                    faces_clean.append([int(face[0]), int(face[2]), int(face[3])])
                else:
                    # Triangle
                    faces_clean.append([int(face[0]), int(face[1]), int(face[2])])
            else:
                # Pure triangular mesh
                faces_clean.append([int(face[0]), int(face[1]), int(face[2])])

        return np.array(faces_clean)

    def _replicate_charge_for_split_faces(self,
            faces: np.ndarray,
            charge: np.ndarray,
            faces_clean: np.ndarray) -> np.ndarray:

        if len(faces_clean) == len(charge):
            return charge

        charge_plot = []
        face_idx = 0

        for orig_face in faces:
            charge_plot.append(charge[face_idx])

            if faces.shape[1] == 4 and not np.isnan(orig_face[3]):
                charge_plot.append(charge[face_idx])

            face_idx += 1

        return np.array(charge_plot)
