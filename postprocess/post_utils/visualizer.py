"""
Visualizer for pyMNPBEM simulation results.

Provides matplotlib-based visualization of:
- Optical spectra (extinction, scattering, absorption)
- Electric field distributions (2D heatmaps)
- Polarization comparisons
- Unpolarized spectra and fields
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Any, Optional, Tuple, Union
import os


class Visualizer:
    """
    Creates visualizations for plasmonic simulation results.

    Supports:
    - Spectrum plots (single and multi-polarization)
    - Field enhancement maps (2D)
    - Geometry overlay on field plots
    - Export to multiple formats (PNG, PDF, SVG, EPS)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualizer.

        Args:
            config: Configuration dictionary with plot settings
        """
        self.config = config or {}

        # Default plot settings
        self.dpi = self.config.get('plot_dpi', 300)
        self.formats = self.config.get('plot_format', ['png'])
        self.x_axis = self.config.get('spectrum_xaxis', 'wavelength')  # 'wavelength' or 'energy'

        # Color schemes
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Set default style
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0

    def plot_spectrum(self, spectrum_data: Dict[str, np.ndarray],
                      pol_idx: int = 0,
                      spectrum_types: List[str] = ['extinction', 'scattering', 'absorption'],
                      title: Optional[str] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optical spectrum for a single polarization.

        Args:
            spectrum_data: Dictionary with wavelengths and cross-sections
            pol_idx: Polarization index
            spectrum_types: Types of spectra to plot
            title: Plot title
            save_path: Path to save figure (without extension)

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        wavelengths = spectrum_data.get('wavelengths')
        if wavelengths is None or len(wavelengths) == 0:
            ax.text(0.5, 0.5, 'No wavelength data available',
                    transform=ax.transAxes, ha='center', va='center')
            return fig

        # Convert to energy if requested
        if self.x_axis == 'energy':
            x_values = 1239.84 / wavelengths  # eV
            x_label = 'Photon Energy (eV)'
        else:
            x_values = wavelengths
            x_label = 'Wavelength (nm)'

        colors = {'extinction': '#1f77b4', 'scattering': '#ff7f0e', 'absorption': '#2ca02c'}
        labels = {'extinction': 'Extinction', 'scattering': 'Scattering', 'absorption': 'Absorption'}

        for spec_type in spectrum_types:
            if spec_type in spectrum_data:
                data = spectrum_data[spec_type]
                if data.ndim > 1:
                    y_values = data[:, pol_idx]
                else:
                    y_values = data
                ax.plot(x_values, y_values, color=colors.get(spec_type, 'black'),
                        label=labels.get(spec_type, spec_type), linewidth=2)

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Cross Section (nm²)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Polarization {pol_idx + 1}', fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_spectrum_comparison(self, spectrum_data: Dict[str, np.ndarray],
                                  spectrum_type: str = 'extinction',
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spectrum comparison across all polarizations.

        Args:
            spectrum_data: Dictionary with wavelengths and cross-sections
            spectrum_type: Type of spectrum to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        wavelengths = spectrum_data.get('wavelengths')
        data = spectrum_data.get(spectrum_type)

        if wavelengths is None or len(wavelengths) == 0 or data is None:
            ax.text(0.5, 0.5, 'No spectrum data available',
                    transform=ax.transAxes, ha='center', va='center')
            return fig

        if self.x_axis == 'energy':
            x_values = 1239.84 / wavelengths
            x_label = 'Photon Energy (eV)'
        else:
            x_values = wavelengths
            x_label = 'Wavelength (nm)'

        n_pol = data.shape[1] if data.ndim > 1 else 1

        for i in range(n_pol):
            y_values = data[:, i] if data.ndim > 1 else data
            ax.plot(x_values, y_values, color=self.colors[i % len(self.colors)],
                    label=f'Polarization {i + 1}', linewidth=2)

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(f'{spectrum_type.capitalize()} Cross Section (nm²)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_spectrum_unpolarized(self, spectrum_data: Dict[str, np.ndarray],
                                   unpolarized_data: Optional[Dict[str, np.ndarray]] = None,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot unpolarized spectrum along with individual polarizations.

        Args:
            spectrum_data: Dictionary with polarization-resolved data
            unpolarized_data: Dictionary with unpolarized spectra
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        wavelengths = spectrum_data.get('wavelengths')
        if wavelengths is None or len(wavelengths) == 0:
            ax.text(0.5, 0.5, 'No wavelength data available',
                    transform=ax.transAxes, ha='center', va='center')
            return fig

        if self.x_axis == 'energy':
            x_values = 1239.84 / wavelengths
            x_label = 'Photon Energy (eV)'
        else:
            x_values = wavelengths
            x_label = 'Wavelength (nm)'

        # Plot individual polarizations (faded)
        ext = spectrum_data.get('extinction')
        if ext is not None:
            n_pol = ext.shape[1] if ext.ndim > 1 else 1
            for i in range(n_pol):
                y_values = ext[:, i] if ext.ndim > 1 else ext
                ax.plot(x_values, y_values, color=self.colors[i % len(self.colors)],
                        alpha=0.5, linewidth=1, linestyle='--',
                        label=f'Pol {i + 1}')

        # Plot unpolarized (bold)
        if unpolarized_data and 'extinction_unpolarized' in unpolarized_data:
            ax.plot(x_values, unpolarized_data['extinction_unpolarized'],
                    color='black', linewidth=2.5, label='Unpolarized')

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Extinction Cross Section (nm²)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_field_enhancement(self, field_data: Dict[str, np.ndarray],
                                log_scale: bool = True,
                                vmin: Optional[float] = None,
                                vmax: Optional[float] = None,
                                geometry_overlay: Optional[List[Dict]] = None,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D field enhancement map.

        Args:
            field_data: Dictionary with enhancement and coordinates
            log_scale: Whether to use logarithmic color scale
            vmin, vmax: Color scale limits
            geometry_overlay: List of geometry elements to overlay
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        enhancement = field_data.get('enhancement')
        x_coords = field_data.get('x')

        # Check for missing data
        if enhancement is None or x_coords is None or len(x_coords) == 0:
            ax.text(0.5, 0.5, 'No field data available',
                    transform=ax.transAxes, ha='center', va='center')
            if title:
                ax.set_title(title, fontsize=14)
            return fig

        y_coords_arr = field_data.get('y', np.array([0.0]))
        z_coords_arr = field_data.get('z', np.array([0.0]))

        # Determine plane orientation and get/create meshgrid arrays
        if len(y_coords_arr) == 1:
            # XZ plane
            y_label = 'z (nm)'
            y_coords = z_coords_arr
            if 'X' in field_data and 'Z' in field_data:
                X = field_data['X']
                Y = field_data['Z']
            else:
                X, Y = np.meshgrid(x_coords, z_coords_arr)
        elif len(z_coords_arr) == 1:
            # XY plane
            y_label = 'y (nm)'
            y_coords = y_coords_arr
            if 'X' in field_data and 'Y' in field_data:
                X = field_data['X']
                Y = field_data['Y']
            else:
                X, Y = np.meshgrid(x_coords, y_coords_arr)
        else:
            y_label = 'y (nm)'
            y_coords = y_coords_arr
            if 'X' in field_data and 'Y' in field_data:
                X = field_data['X']
                Y = field_data['Y']
            else:
                X, Y = np.meshgrid(x_coords, y_coords_arr)

        # Set color scale
        if vmax is None:
            positive_values = enhancement[enhancement > 0]
            vmax = np.max(positive_values) if len(positive_values) > 0 else 1.0
        if vmin is None:
            vmin = 1.0 if log_scale else 0.0

        # Plot
        if log_scale:
            norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
        else:
            norm = None
            enhancement = np.clip(enhancement, vmin, vmax)

        # Ensure shapes are compatible for pcolormesh
        # enhancement should be transposed if X, Y are from meshgrid
        try:
            if X.shape != enhancement.shape:
                enhancement = enhancement.T
            im = ax.pcolormesh(X, Y, enhancement, cmap='hot', norm=norm, shading='auto')
        except Exception as e:
            # Fallback: create simple meshgrid from coordinates
            X_new, Y_new = np.meshgrid(x_coords, y_coords)
            if X_new.shape != enhancement.shape:
                enhancement = enhancement.T
            im = ax.pcolormesh(X_new, Y_new, enhancement, cmap='hot', norm=norm, shading='auto')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('|E|²/|E₀|²', fontsize=12)

        # Add geometry overlay
        if geometry_overlay:
            self._add_geometry_overlay(ax, geometry_overlay)

        ax.set_xlabel('x (nm)', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_aspect('equal')

        if title:
            ax.set_title(title, fontsize=14)
        elif 'wavelength' in field_data:
            ax.set_title(f'Field Enhancement at λ = {field_data["wavelength"]:.1f} nm', fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_field_comparison(self, field_data: Dict[int, Dict[str, np.ndarray]],
                               log_scale: bool = True,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot field enhancement for multiple polarizations side by side.

        Args:
            field_data: Dictionary mapping polarization index to field data
            log_scale: Whether to use logarithmic color scale
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        n_pol = len(field_data)
        if n_pol == 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.text(0.5, 0.5, 'No field data available', ha='center', va='center')
            return fig

        fig, axes = plt.subplots(1, n_pol, figsize=(6 * n_pol, 5))

        if n_pol == 1:
            axes = [axes]

        # Find global vmax
        def safe_max(data):
            enh = data.get('enhancement')
            if enh is None:
                return 1.0
            positive = enh[enh > 0]
            return np.max(positive) if len(positive) > 0 else 1.0
        vmax = max((safe_max(data) for data in field_data.values()), default=1.0)

        for i, (pol_idx, data) in enumerate(sorted(field_data.items())):
            ax = axes[i]

            enhancement = data.get('enhancement')
            x_coords = data.get('x')

            # Check for missing data
            if enhancement is None or x_coords is None or len(x_coords) == 0:
                ax.text(0.5, 0.5, 'No field data',
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'Polarization {pol_idx + 1}', fontsize=12)
                continue

            y_coords_arr = data.get('y', np.array([0.0]))
            z_coords_arr = data.get('z', np.array([0.0]))

            # Get or create meshgrid arrays
            if len(y_coords_arr) == 1:
                # XZ plane
                if 'X' in data and 'Z' in data:
                    X, Y = data['X'], data['Z']
                else:
                    X, Y = np.meshgrid(x_coords, z_coords_arr)
            else:
                # XY plane
                if 'X' in data and 'Y' in data:
                    X, Y = data['X'], data['Y']
                else:
                    X, Y = np.meshgrid(x_coords, y_coords_arr)

            if log_scale:
                norm = LogNorm(vmin=1.0, vmax=vmax)
            else:
                norm = None

            # Ensure shapes are compatible for pcolormesh
            try:
                if X.shape != enhancement.shape:
                    enhancement = enhancement.T
                im = ax.pcolormesh(X, Y, enhancement, cmap='hot', norm=norm, shading='auto')
            except Exception:
                X_new, Y_new = np.meshgrid(x_coords, z_coords_arr if len(y_coords_arr) == 1 else y_coords_arr)
                if X_new.shape != enhancement.shape:
                    enhancement = enhancement.T
                im = ax.pcolormesh(X_new, Y_new, enhancement, cmap='hot', norm=norm, shading='auto')
            ax.set_xlabel('x (nm)', fontsize=12)
            if i == 0:
                ax.set_ylabel('z (nm)' if len(y_coords_arr) == 1 else 'y (nm)', fontsize=12)
            ax.set_aspect('equal')
            ax.set_title(f'Polarization {pol_idx + 1}', fontsize=12)

            plt.colorbar(im, ax=ax, label='|E|²/|E₀|²')

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_all_spectra(self, spectrum_data: Dict[str, np.ndarray],
                         save_dir: str) -> List[plt.Figure]:
        """
        Generate all standard spectrum plots.

        Args:
            spectrum_data: Dictionary with wavelengths and cross-sections
            save_dir: Directory to save plots

        Returns:
            List of figure objects
        """
        figures = []
        os.makedirs(save_dir, exist_ok=True)

        ext = spectrum_data.get('extinction')
        if ext is None:
            return figures
        n_pol = ext.shape[1] if ext.ndim > 1 else 1

        # Individual polarization plots
        for pol_idx in range(n_pol):
            fig = self.plot_spectrum(
                spectrum_data, pol_idx,
                save_path=os.path.join(save_dir, f'spectrum_pol{pol_idx + 1}')
            )
            figures.append(fig)

        # Comparison plot
        fig = self.plot_spectrum_comparison(
            spectrum_data, 'extinction',
            title='Extinction Comparison',
            save_path=os.path.join(save_dir, 'spectrum_comparison')
        )
        figures.append(fig)

        return figures

    def _add_geometry_overlay(self, ax: plt.Axes, elements: List[Dict]):
        """Add geometry overlay to a plot."""
        for elem in elements:
            if elem['type'] == 'circle':
                circle = Circle(
                    (elem['x'], elem['y']),
                    elem['radius'],
                    fill=False,
                    edgecolor=elem.get('color', 'white'),
                    linewidth=elem.get('linewidth', 2),
                    linestyle=elem.get('linestyle', '-')
                )
                ax.add_patch(circle)
            elif elem['type'] == 'rectangle':
                rect = Rectangle(
                    (elem['x'] - elem['width']/2, elem['y'] - elem['height']/2),
                    elem['width'],
                    elem['height'],
                    fill=False,
                    edgecolor=elem.get('color', 'white'),
                    linewidth=elem.get('linewidth', 2),
                    linestyle=elem.get('linestyle', '-')
                )
                ax.add_patch(rect)

    def _save_figure(self, fig: plt.Figure, base_path: str):
        """Save figure in configured formats."""
        formats = self.formats if isinstance(self.formats, list) else [self.formats]

        for fmt in formats:
            filepath = f"{base_path}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')

    def close_all(self):
        """Close all matplotlib figures."""
        plt.close('all')
