import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
from .geometry_cross_section import GeometryCrossSection

class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = os.path.join(config.get('output_dir'), config.get('simulation_name'))
        self.save_plots = config.get('save_plots', True)
        self.plot_format = config.get('plot_format', ['png', 'pdf'])
        self.dpi = config.get('plot_dpi', 300)
        self.polarizations = config.get('polarizations', [])
        self.propagation_dirs = config.get('propagation_dirs', [])
        self.geometry = GeometryCrossSection(config, verbose)
    
    def _format_vector_label(self, vec):
        """Format vector as compact string."""
        if vec is None or len(vec) == 0:
            return ""
        vec_rounded = np.round(vec, 3)
        return f"[{vec_rounded[0]:.0f} {vec_rounded[1]:.0f} {vec_rounded[2]:.0f}]"

    def _get_polarization_label(self, ipol):
        """Get descriptive label for polarization."""
        if ipol < len(self.polarizations) and ipol < len(self.propagation_dirs):
            pol_vec = self.polarizations[ipol]
            prop_vec = self.propagation_dirs[ipol]
            pol_str = self._format_vector_label(pol_vec)
            prop_str = self._format_vector_label(prop_vec)
            return f"Pol{pol_str} Prop{prop_str}"
        else:
            return f"Polarization {ipol+1}"
    
    def create_all_plots(self, data, analysis_results=None):
        """Create all visualization plots."""
        plots_created = []

        # Spectrum plots
        if 'wavelength' in data and 'extinction' in data:
            spectrum_file = self.plot_spectrum(data)
            plots_created.append(spectrum_file)

        # Polarization comparison
        if 'wavelength' in data and data['extinction'].shape[1] > 1:
            pol_file = self.plot_polarization_comparison(data)
            plots_created.append(pol_file)

        # Unpolarized spectrum plots (if available)
        if analysis_results and 'unpolarized_spectrum' in analysis_results:
            unpol_files = self.plot_unpolarized_spectrum(data, analysis_results)
            plots_created.extend(unpol_files)

        # Field plots
        if 'fields' in data:
            field_files = self.plot_fields(data)
            plots_created.extend(field_files)

            # Unpolarized field plots (if conditions met)
            if analysis_results and analysis_results.get('unpolarized', {}).get('can_calculate', False):
                unpol_field_files = self.plot_unpolarized_fields(data, analysis_results)
                plots_created.extend(unpol_field_files)

        return plots_created
    
    def plot_spectrum(self, data):
        """Plot extinction, scattering, and absorption spectra for each polarization."""
        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        
        # Check x-axis unit preference
        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')
        
        # Convert to energy if requested
        if xaxis_unit == 'energy':
            # E(eV) = 1239.84 / λ(nm)
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            # Reverse order for energy (high energy = short wavelength)
            xdata = xdata[::-1]
            extinction = extinction[::-1, :]
            scattering = scattering[::-1, :]
            absorption = absorption[::-1, :]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'
        
        n_pol = extinction.shape[1]
        saved_files_all = []
        
        # Create separate plot for each polarization
        for ipol in range(n_pol):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(xdata, extinction[:, ipol], 'b-', linewidth=2, label='Extinction')
            ax.plot(xdata, scattering[:, ipol], 'r--', linewidth=2, label='Scattering')
            ax.plot(xdata, absorption[:, ipol], 'g:', linewidth=2, label='Absorption')
            
            ax.set_xlabel(xlabel_text, fontsize=12)
            ax.set_ylabel('Cross Section (nm²)', fontsize=12)

            # ✅ FIX: Don't duplicate "Polarization"
            pol_label = self._get_polarization_label(ipol)
            ax.set_title(f'Optical Spectra - {pol_label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Reverse x-axis for energy to show high energy on left
            if xaxis_unit == 'energy':
                ax.invert_xaxis()
            
            plt.tight_layout()
            
            # Save plot with polarization index
            base_filename = f'simulation_spectrum_pol{ipol+1}'
            saved_files = self._save_figure(fig, base_filename)
            if saved_files:
                saved_files_all.extend(saved_files)
            plt.close(fig)
        
        return saved_files_all
    
    def plot_polarization_comparison(self, data):
        """Plot comparison of different polarizations for all cross sections."""
        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        n_pol = extinction.shape[1]
        
        # Check x-axis unit preference
        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')
        
        # Convert to energy if requested
        if xaxis_unit == 'energy':
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            # Reverse order for energy
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
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_pol):
            # ✅ FIX: Use polarization label
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, extinction[:, i], 
                   color=colors[i], linewidth=2, 
                   label=pol_label)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison - Extinction', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)  # Smaller for long labels
        ax.grid(True, alpha=0.3)
        
        if xaxis_unit == 'energy':
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        base_filename = 'simulation_polarization_extinction'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)
        
        # ========== Scattering Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, scattering[:, i], 
                   color=colors[i], linewidth=2, 
                   label=pol_label)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Scattering Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison - Scattering', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if xaxis_unit == 'energy':
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        base_filename = 'simulation_polarization_scattering'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)
        
        # ========== Absorption Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, absorption[:, i], 
                   color=colors[i], linewidth=2, 
                   label=pol_label)
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Absorption Cross Section (nm²)', fontsize=12)
        ax.set_title('Polarization Comparison - Absorption', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if xaxis_unit == 'energy':
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        base_filename = 'simulation_polarization_absorption'
        saved_files = self._save_figure(fig, base_filename)
        if saved_files:
            saved_files_all.extend(saved_files)
        plt.close(fig)
        
        return saved_files_all
    
    def plot_fields(self, data):
        """Plot electromagnetic field distributions."""
        if 'fields' not in data or not data['fields']:
            return []

        fields = data['fields']
        saved_files = []

        # Plot for each field entry (may have multiple wavelengths and polarizations)
        for idx, field_data in enumerate(fields):
            # Get polarization and wavelength indices from field_data
            # Fall back to enumerate index for backwards compatibility
            pol_idx = field_data.get('polarization_idx', idx)
            wl_idx = field_data.get('wavelength_idx')
            wavelength = field_data.get('wavelength')

            # Enhancement plot
            enhancement_file = self._plot_field_enhancement(field_data, pol_idx, wl_idx)
            if enhancement_file:
                saved_files.extend(enhancement_file)

            # Intensity plot
            intensity_file = self._plot_field_intensity(field_data, pol_idx, wl_idx)
            if intensity_file:
                saved_files.extend(intensity_file)

            # Vector field plot (optional, for 2D slices)
            if self._is_2d_slice(field_data):
                vector_file = self._plot_field_vectors(field_data, pol_idx, wl_idx)
                if vector_file:
                    saved_files.extend(vector_file)

        return saved_files
    
    def _plot_field_enhancement(self, field_data, polarization_idx, wavelength_idx=None):
        """Plot field enhancement |E|/|E0|."""
        enhancement = field_data['enhancement']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        
        # FIX: Handle scalar enhancement
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([[enhancement]])
        elif enhancement.ndim == 0:
            enhancement = np.array([[enhancement.item()]])
        elif enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)

        # FIX: Convert complex to real magnitude for plotting
        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)

        # Determine plane type
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        # FIX: Use polarization label
        pol_label = self._get_polarization_label(polarization_idx)

        # FIX: Use masked array to make NaN transparent (instead of black)
        enhancement_masked = np.ma.masked_invalid(enhancement)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Calculate percentile-based limits for better visualization
        valid_data = enhancement_masked.compressed()  # Get non-masked values
        if len(valid_data) > 0:
            vmin_linear = np.percentile(valid_data, 1)
            vmax_linear = np.percentile(valid_data, 99)
        else:
            vmin_linear, vmax_linear = 0, 1

        # Linear scale with percentile clipping
        im1 = ax1.imshow(enhancement_masked, extent=extent, origin='lower',
                        cmap='hot', aspect='auto', vmin=vmin_linear, vmax=vmax_linear)
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.set_title(f'Field Enhancement (Linear)\nλ = {wavelength:.1f} nm, {pol_label}',
                     fontsize=11, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('|E|/|E₀|', fontsize=11)

        z_plane = float(z_grid.flat[0])  # Extract z-coordinate
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax1, section, plane_type)

        # Log scale
        if len(valid_data) > 0 and np.any(valid_data > 0):
            # Use percentile for log scale limits
            positive_data = valid_data[valid_data > 0]
            vmin_log = max(np.percentile(positive_data, 5), 0.1)  # At least 0.1
            vmax_log = np.percentile(positive_data, 99.5)

            if vmin_log >= vmax_log:
                vmin_log = vmax_log / 100

            im2 = ax2.imshow(enhancement_masked, extent=extent, origin='lower',
                            cmap='hot', aspect='auto',
                            norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
            ax2.set_xlabel(x_label, fontsize=11)
            ax2.set_ylabel(y_label, fontsize=11)
            ax2.set_title(f'Field Enhancement (Log Scale)\nλ = {wavelength:.1f} nm, {pol_label}',
                         fontsize=11, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2)
            for section in sections:
                self._draw_material_boundary(ax2, section, plane_type)
            cbar2.set_label('|E|/|E₀|', fontsize=11)
        else:
            im2 = ax2.imshow(enhancement_masked, extent=extent, origin='lower',
                            cmap='hot', aspect='auto')
            ax2.set_xlabel(x_label, fontsize=11)
            ax2.set_ylabel(y_label, fontsize=11)
            ax2.set_title(f'Field Enhancement\nλ = {wavelength:.1f} nm, {pol_label}',
                         fontsize=11, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2)
            for section in sections:
                self._draw_material_boundary(ax2, section, plane_type)
            cbar2.set_label('|E|/|E₀|', fontsize=11)

        plt.tight_layout()

        # Save - include wavelength info if multiple wavelengths
        if wavelength_idx is not None:
            base_filename = f'field_enhancement_wl{wavelength_idx}_pol{polarization_idx+1}_{plane_type}'
        else:
            base_filename = f'field_enhancement_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved_files

    def _plot_field_intensity(self, field_data, polarization_idx, wavelength_idx=None):
        """Plot field intensity |E|²."""
        intensity = field_data['intensity']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        
        # FIX: Handle scalar intensity
        if not isinstance(intensity, np.ndarray):
            intensity = np.array([[intensity]])
        elif intensity.ndim == 0:
            intensity = np.array([[intensity.item()]])
        elif intensity.ndim == 1:
            intensity = intensity.reshape(1, -1)

        # FIX: Convert complex to real magnitude for plotting
        if np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        # Determine plane type
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        # FIX: Use polarization label
        pol_label = self._get_polarization_label(polarization_idx)

        # FIX: Use masked array to make NaN transparent
        intensity_masked = np.ma.masked_invalid(intensity)

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 7))

        # Use percentile-based scaling for better visualization of hotspots
        valid_data = intensity_masked.compressed()
        if len(valid_data) > 0 and np.any(valid_data > 0):
            positive_data = valid_data[valid_data > 0]
            if len(positive_data) > 0:
                # Use percentile to avoid extreme outliers dominating colorscale
                vmin_log = max(np.percentile(positive_data, 2), 1e-10)
                vmax_log = np.percentile(positive_data, 99.5)

                if vmin_log >= vmax_log:
                    vmin_log = vmax_log / 1000

                im = ax.imshow(intensity_masked, extent=extent, origin='lower',
                              cmap='inferno', aspect='auto',
                              norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
            else:
                im = ax.imshow(intensity_masked, extent=extent, origin='lower',
                              cmap='inferno', aspect='auto')
        else:
            im = ax.imshow(intensity_masked, extent=extent, origin='lower',
                          cmap='inferno', aspect='auto')

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Field Intensity |E|² (Log Scale)\nλ = {wavelength:.1f} nm, {pol_label}',
                    fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|E|² (a.u.)', fontsize=11)

        z_plane = float(z_grid.flat[0])
        sections = self.geometry.get_cross_section(z_plane)
        for section in sections:
            self._draw_material_boundary(ax, section, plane_type)
        
        plt.tight_layout()

        # Save - include wavelength info if multiple wavelengths
        if wavelength_idx is not None:
            base_filename = f'field_intensity_wl{wavelength_idx}_pol{polarization_idx+1}_{plane_type}'
        else:
            base_filename = f'field_intensity_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved_files

    def _plot_field_vectors(self, field_data, polarization_idx, wavelength_idx=None):
        """Plot electric field vector arrows (for 2D slices)."""
        e_total = field_data['e_total']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        wavelength = field_data['wavelength']
        enhancement = field_data['enhancement']
        
        # Determine plane and extract relevant components
        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        # FIX: Use polarization label
        pol_label = self._get_polarization_label(polarization_idx)

        # FIX: Convert complex enhancement to real magnitude for plotting
        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)

        # Properly extract coordinates and field components
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # FIX: Use masked array for NaN transparency
        enhancement_masked = np.ma.masked_invalid(enhancement)

        im = ax.imshow(enhancement_masked, extent=extent, origin='lower',
                      cmap='hot', aspect='auto', alpha=0.7)
        
        q = ax.quiver(X, Y, U_norm, V_norm, magnitude,
                      cmap='cool', scale=25, width=0.004, 
                      alpha=0.9, pivot='middle')
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Electric Field Vectors\nλ = {wavelength:.1f} nm, {pol_label}', 
                    fontsize=12, fontweight='bold')
        
        cbar1 = plt.colorbar(im, ax=ax, pad=0.12, label='|E|/|E₀|')
        cbar2 = plt.colorbar(q, ax=ax, label='Field Magnitude')

        plt.tight_layout()

        # Save - include wavelength info if multiple wavelengths
        if wavelength_idx is not None:
            base_filename = f'field_vectors_wl{wavelength_idx}_pol{polarization_idx+1}_{plane_type}'
        else:
            base_filename = f'field_vectors_pol{polarization_idx+1}_{plane_type}'
        saved_files = self._save_figure(fig, base_filename)
        plt.close(fig)

        return saved_files
    
    def _determine_plane(self, x_grid, y_grid, z_grid):
        """Determine which 2D plane is being plotted."""
        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([[x_grid]])
            y_grid = np.array([[y_grid]])
            z_grid = np.array([[z_grid]])
        
        if x_grid.ndim == 0:
            x_grid = np.array([[x_grid.item()]])
            y_grid = np.array([[y_grid.item()]])
            z_grid = np.array([[z_grid.item()]])
        
        if x_grid.ndim == 1:
            x_grid = x_grid.reshape(1, -1)
            y_grid = y_grid.reshape(1, -1)
            z_grid = z_grid.reshape(1, -1)
        
        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1
        
        if y_constant:
            plane_type = 'xz'
            x_min, x_max = x_grid.min(), x_grid.max()
            z_min, z_max = z_grid.min(), z_grid.max()
            # ✅ FIX: Extend extent if single point
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
            # ✅ FIX: Extend extent if single point
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
            # ✅ FIX: Extend extent if single point
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

    def _is_2d_slice(self, field_data):
        """Check if field data is a 2D slice (for vector plots)."""
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']
        
        # ✅ FIX: Handle scalars
        if not isinstance(x_grid, np.ndarray):
            return False  # Single point, not a slice
        
        if x_grid.ndim == 0:
            return False  # Single point
        
        x_constant = len(np.unique(x_grid)) == 1
        y_constant = len(np.unique(y_grid)) == 1
        z_constant = len(np.unique(z_grid)) == 1
        
        return sum([x_constant, y_constant, z_constant]) == 1
    
    def _save_figure(self, fig, base_filename):
        """Save figure in specified formats."""
        saved_files = []
        
        for fmt in self.plot_format:
            filepath = os.path.join(self.output_dir, f'{base_filename}.{fmt}')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(filepath)
            
            if self.verbose:
                print(f"  Saved: {filepath}")
        
        return saved_files


    def _draw_material_boundary(self, ax, section, plane_type):
        """
        Draw material boundary on field plot.
        
        Parameters
        ----------
        ax : matplotlib axis
            Axis to draw on
        section : dict
            Cross-section information from GeometryCrossSection
        plane_type : str
            Type of plane ('xy', 'xz', 'yz')
        """
        if section['type'] == 'circle':
            # Draw circle
            center = section['center']
            radius = section['radius']
            
            circle = Circle(
                center,
                radius,
                fill=False,
                edgecolor='gray',
                linestyle='--',
                linewidth=2,
                label=section.get('label', 'Material boundary')
            )
            ax.add_patch(circle)
        
        elif section['type'] == 'rectangle':
            # Draw rectangle
            bounds = section['bounds']  # [x_min, x_max, y_min, y_max]
            x_min, x_max, y_min, y_max = bounds
            
            width = x_max - x_min
            height = y_max - y_min
            
            rectangle = Rectangle(
                (x_min, y_min),
                width,
                height,
                fill=False,
                edgecolor='gray',
                linestyle='--',
                linewidth=2,
                label=section.get('label', 'Material boundary')
            )
            ax.add_patch(rectangle)

    # ========================================================================
    # UNPOLARIZED PLOTS
    # ========================================================================

    def plot_unpolarized_spectrum(self, data, analysis_results):
        """
        Plot unpolarized spectrum (FDTD-style incoherent average).

        Creates:
        1. Standalone unpolarized spectrum plot
        2. Comparison plot: all polarizations + unpolarized
        """
        saved_files = []
        unpol = analysis_results['unpolarized_spectrum']

        wavelength = unpol['wavelength']
        unpol_ext = unpol['extinction']
        unpol_sca = unpol['scattering']
        unpol_abs = unpol['absorption']

        # Check x-axis unit preference
        xaxis_unit = self.config.get('spectrum_xaxis', 'wavelength')

        if xaxis_unit == 'energy':
            xdata = 1239.84 / wavelength
            xlabel_text = 'Energy (eV)'
            # Reverse for energy
            xdata = xdata[::-1]
            unpol_ext = unpol_ext[::-1]
            unpol_sca = unpol_sca[::-1]
            unpol_abs = unpol_abs[::-1]
        else:
            xdata = wavelength
            xlabel_text = 'Wavelength (nm)'

        # ========== 1. Standalone Unpolarized Spectrum ==========
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(xdata, unpol_ext, 'b-', linewidth=2, label='Extinction')
        ax.plot(xdata, unpol_sca, 'r--', linewidth=2, label='Scattering')
        ax.plot(xdata, unpol_abs, 'g:', linewidth=2, label='Absorption')

        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Cross Section (nm²)', fontsize=12)
        ax.set_title(f'Unpolarized Spectrum\n(FDTD-style incoherent average, {unpol["n_averaged"]} polarizations)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

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

    def _plot_spectrum_comparison_with_unpolarized(self, data, analysis_results):
        """Plot comparison of all polarizations including unpolarized."""
        saved_files = []

        wavelength = data['wavelength']
        extinction = data['extinction']
        scattering = data['scattering']
        absorption = data['absorption']
        n_pol = extinction.shape[1]

        unpol = analysis_results['unpolarized_spectrum']

        # X-axis
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

        # Colors for polarizations + black for unpolarized
        colors = plt.cm.tab10(np.linspace(0, 0.7, n_pol))

        # ========== Extinction Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, extinction[:, i], color=colors[i], linewidth=1.5,
                   linestyle='--', alpha=0.7, label=pol_label)

        ax.plot(xdata, unpol_ext, 'k-', linewidth=2.5, label='Unpolarized')

        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize=12)
        ax.set_title('Extinction: Polarizations vs Unpolarized', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()
        files = self._save_figure(fig, 'simulation_comparison_extinction_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== Scattering Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, scattering[:, i], color=colors[i], linewidth=1.5,
                   linestyle='--', alpha=0.7, label=pol_label)

        ax.plot(xdata, unpol_sca, 'k-', linewidth=2.5, label='Unpolarized')

        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Scattering Cross Section (nm²)', fontsize=12)
        ax.set_title('Scattering: Polarizations vs Unpolarized', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()
        files = self._save_figure(fig, 'simulation_comparison_scattering_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== Absorption Comparison ==========
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            ax.plot(xdata, absorption[:, i], color=colors[i], linewidth=1.5,
                   linestyle='--', alpha=0.7, label=pol_label)

        ax.plot(xdata, unpol_abs, 'k-', linewidth=2.5, label='Unpolarized')

        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel('Absorption Cross Section (nm²)', fontsize=12)
        ax.set_title('Absorption: Polarizations vs Unpolarized', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if xaxis_unit == 'energy':
            ax.invert_xaxis()

        plt.tight_layout()
        files = self._save_figure(fig, 'simulation_comparison_absorption_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        # ========== All-in-one Comparison (3 subplots) ==========
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i in range(n_pol):
            pol_label = self._get_polarization_label(i)
            axes[0].plot(xdata, extinction[:, i], color=colors[i], linewidth=1.5,
                        linestyle='--', alpha=0.7, label=pol_label)
            axes[1].plot(xdata, scattering[:, i], color=colors[i], linewidth=1.5,
                        linestyle='--', alpha=0.7, label=pol_label)
            axes[2].plot(xdata, absorption[:, i], color=colors[i], linewidth=1.5,
                        linestyle='--', alpha=0.7, label=pol_label)

        axes[0].plot(xdata, unpol_ext, 'k-', linewidth=2.5, label='Unpolarized')
        axes[1].plot(xdata, unpol_sca, 'k-', linewidth=2.5, label='Unpolarized')
        axes[2].plot(xdata, unpol_abs, 'k-', linewidth=2.5, label='Unpolarized')

        titles = ['Extinction', 'Scattering', 'Absorption']
        for idx, ax in enumerate(axes):
            ax.set_xlabel(xlabel_text, fontsize=11)
            ax.set_ylabel('Cross Section (nm²)', fontsize=11)
            ax.set_title(titles[idx], fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if xaxis_unit == 'energy':
                ax.invert_xaxis()

        plt.suptitle('Polarizations vs Unpolarized Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        files = self._save_figure(fig, 'simulation_comparison_all_unpolarized')
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def plot_unpolarized_fields(self, data, analysis_results):
        """
        Plot unpolarized field distributions.

        Uses FDTD-style incoherent averaging:
        - Intensity: I_unpol = mean(I_pol1, I_pol2, ...)
        - Enhancement: enh_unpol = sqrt(mean(enh1^2, enh2^2, ...))
        """
        saved_files = []
        fields = data.get('fields', [])

        if not fields:
            return saved_files

        unpol_info = analysis_results.get('unpolarized', {})
        method = unpol_info.get('method', '')

        # Determine expected number of polarizations
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
            wl_key = f"{wl:.1f}"
            if wl_key not in fields_by_wavelength:
                fields_by_wavelength[wl_key] = []
            fields_by_wavelength[wl_key].append(field)

        for wl_key, wl_fields in fields_by_wavelength.items():
            # Check if we have all required polarizations
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

            # Calculate unpolarized (incoherent average)
            enhancements_sq = []
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

                enhancements_sq.append(enh ** 2)
                if inten is not None:
                    intensities.append(inten)

            if len(enhancements_sq) != expected_n_pol:
                continue

            # Incoherent average
            unpol_enh = np.sqrt(np.mean(enhancements_sq, axis=0))
            unpol_intensity = np.mean(intensities, axis=0) if len(intensities) == expected_n_pol else unpol_enh ** 2

            # Create unpolarized field dict
            unpol_field = {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid,
                'enhancement': unpol_enh,
                'intensity': unpol_intensity,
                'wavelength': wavelength
            }

            # Plot unpolarized enhancement
            unpol_files = self._plot_unpolarized_field_enhancement(
                unpol_field, wavelength, expected_n_pol
            )
            saved_files.extend(unpol_files)

            # Plot field comparison (all polarizations + unpolarized)
            comparison_files = self._plot_field_comparison_with_unpolarized(
                wl_fields_sorted, unpol_field, wavelength
            )
            saved_files.extend(comparison_files)

        return saved_files

    def _plot_unpolarized_field_enhancement(self, unpol_field, wavelength, n_pol):
        """Plot unpolarized field enhancement."""
        saved_files = []

        enhancement = unpol_field['enhancement']
        x_grid = unpol_field['x_grid']
        y_grid = unpol_field['y_grid']
        z_grid = unpol_field['z_grid']

        # Handle scalar/1D arrays
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([[enhancement]])
        elif enhancement.ndim == 0:
            enhancement = np.array([[enhancement.item()]])
        elif enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)
        enhancement_masked = np.ma.masked_invalid(enhancement)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        valid_data = enhancement_masked.compressed()
        if len(valid_data) > 0:
            vmin_linear = np.percentile(valid_data, 1)
            vmax_linear = np.percentile(valid_data, 99)
        else:
            vmin_linear, vmax_linear = 0, 1

        # Linear scale
        im1 = ax1.imshow(enhancement_masked, extent=extent, origin='lower',
                        cmap='hot', aspect='auto', vmin=vmin_linear, vmax=vmax_linear)
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.set_title(f'Unpolarized Field Enhancement (Linear)\n'
                     f'λ = {wavelength:.1f} nm, avg of {n_pol} pols',
                     fontsize=11, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('|E|/|E₀|', fontsize=11)

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

            im2 = ax2.imshow(enhancement_masked, extent=extent, origin='lower',
                            cmap='hot', aspect='auto',
                            norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
        else:
            im2 = ax2.imshow(enhancement_masked, extent=extent, origin='lower',
                            cmap='hot', aspect='auto')

        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel(y_label, fontsize=11)
        ax2.set_title(f'Unpolarized Field Enhancement (Log)\n'
                     f'λ = {wavelength:.1f} nm, avg of {n_pol} pols',
                     fontsize=11, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('|E|/|E₀|', fontsize=11)

        for section in sections:
            self._draw_material_boundary(ax2, section, plane_type)

        plt.tight_layout()

        base_filename = f'field_enhancement_unpolarized_{plane_type}'
        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files

    def _plot_field_comparison_with_unpolarized(self, pol_fields, unpol_field, wavelength):
        """Plot comparison of field enhancement: all polarizations + unpolarized."""
        saved_files = []
        n_pol = len(pol_fields)

        # Get grid info
        ref_field = pol_fields[0]
        x_grid = ref_field['x_grid']
        y_grid = ref_field['y_grid']
        z_grid = ref_field['z_grid']

        plane_type, extent, x_label, y_label = self._determine_plane(x_grid, y_grid, z_grid)

        # Create figure with n_pol + 1 subplots
        n_cols = n_pol + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        # Find global vmin/vmax for consistent colorbar
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

        all_data_flat = np.concatenate(all_data)
        valid_data = all_data_flat[~np.isnan(all_data_flat)]

        if len(valid_data) > 0:
            # Enhancement is |E|/|E₀|, must be non-negative
            vmin = max(0, np.percentile(valid_data, 1))
            vmax = np.percentile(valid_data, 99)

            # If vmax is very small or zero, use reasonable defaults
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

            im = ax.imshow(enh_masked, extent=extent, origin='lower',
                          cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)

            pol_label = self._get_polarization_label(pol_idx)
            ax.set_title(f'{pol_label}', fontsize=10, fontweight='bold')
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)

            z_plane = float(z_grid.flat[0]) if isinstance(z_grid, np.ndarray) else float(z_grid)
            sections = self.geometry.get_cross_section(z_plane)
            for section in sections:
                self._draw_material_boundary(ax, section, plane_type)

        # Plot unpolarized
        ax = axes[-1]
        unpol_enh_masked = np.ma.masked_invalid(unpol_enh)

        im = ax.imshow(unpol_enh_masked, extent=extent, origin='lower',
                      cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)

        ax.set_title(f'Unpolarized', fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)

        for section in sections:
            self._draw_material_boundary(ax, section, plane_type)

        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('|E|/|E₀|', fontsize=11)

        plt.suptitle(f'Field Enhancement Comparison (λ = {wavelength:.1f} nm)',
                    fontsize=12, fontweight='bold')

        base_filename = f'field_comparison_unpolarized_{plane_type}'
        files = self._save_figure(fig, base_filename)
        if files:
            saved_files.extend(files)
        plt.close(fig)

        return saved_files
