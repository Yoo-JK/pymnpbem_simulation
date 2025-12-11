"""
Surface Charge Visualizer for pyMNPBEM simulation results.

Provides 3D visualization of surface charge distributions for:
- Mode identification (dipolar, quadrupolar, etc.)
- Plasmon mode analysis
- Publication-quality figures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Any, Optional, Tuple
import os


class SurfaceChargeVisualizer:
    """
    Creates 3D visualizations of surface charge distributions.

    Features:
    - 3D surface plots with charge coloring
    - Multiple viewing angles
    - Mode classification visualization
    - Cross-section charge plots
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualizer.

        Args:
            config: Configuration dictionary with plot settings
        """
        self.config = config or {}
        self.dpi = self.config.get('plot_dpi', 300)
        self.formats = self.config.get('plot_format', ['png'])

    def plot_surface_charges_3d(self, charge_data: Dict[str, np.ndarray],
                                 component: str = 'real',
                                 view_angles: Tuple[float, float] = (30, 45),
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D surface with charge distribution coloring.

        Args:
            charge_data: Dictionary with vertices, faces, and charges
            component: 'real', 'imag', 'magnitude', or 'phase'
            view_angles: (elevation, azimuth) viewing angles
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        vertices = charge_data['vertices']
        faces = charge_data['faces']

        # Get charge values to plot
        if component == 'real':
            values = charge_data['charge_real']
            label = 'Re(σ)'
        elif component == 'imag':
            values = charge_data['charge_imag']
            label = 'Im(σ)'
        elif component == 'magnitude':
            values = charge_data['charge_magnitude']
            label = '|σ|'
        elif component == 'phase':
            values = charge_data['charge_phase']
            label = 'Phase(σ)'
        else:
            values = charge_data['charge_real']
            label = 'Re(σ)'

        # Create polygons
        polys = []
        face_values = []

        for i, face in enumerate(faces):
            if len(face) >= 3:
                # Handle both triangles and quads
                if len(face) == 3:
                    verts = vertices[face]
                    polys.append(verts)
                    face_values.append(values[i] if i < len(values) else 0)
                elif len(face) == 4:
                    # Split quad into two triangles
                    for tri_indices in [[0, 1, 2], [0, 2, 3]]:
                        verts = vertices[face[tri_indices]]
                        polys.append(verts)
                        face_values.append(values[i] if i < len(values) else 0)

        face_values = np.array(face_values)

        # Handle empty face_values array
        if len(face_values) == 0:
            ax.text2D(0.5, 0.5, 'No face data available',
                      transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xlabel('x (nm)', fontsize=12)
            ax.set_ylabel('y (nm)', fontsize=12)
            ax.set_zlabel('z (nm)', fontsize=12)
            if title:
                ax.set_title(title, fontsize=14)
            plt.tight_layout()
            if save_path:
                self._save_figure(fig, save_path)
            return fig

        # Normalize colors
        vmax = np.max(np.abs(face_values))
        if vmax == 0:
            vmax = 1.0  # Prevent division by zero in normalization
        if component in ['real', 'imag']:
            # Symmetric colormap centered at 0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = plt.cm.hot

        # Create colors
        colors = cmap(norm(face_values))

        # Create 3D collection
        poly_collection = Poly3DCollection(polys, facecolors=colors,
                                            edgecolors='gray', linewidths=0.1,
                                            alpha=0.9)
        ax.add_collection3d(poly_collection)

        # Set axis limits
        x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
        y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
        z_range = [vertices[:, 2].min(), vertices[:, 2].max()]

        max_range = max(x_range[1] - x_range[0],
                        y_range[1] - y_range[0],
                        z_range[1] - z_range[0]) / 2

        mid_x = (x_range[0] + x_range[1]) / 2
        mid_y = (y_range[0] + y_range[1]) / 2
        mid_z = (z_range[0] + z_range[1]) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set viewing angle
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

        # Labels
        ax.set_xlabel('x (nm)', fontsize=12)
        ax.set_ylabel('y (nm)', fontsize=12)
        ax.set_zlabel('z (nm)', fontsize=12)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label(f'Surface Charge {label}', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        elif 'wavelength' in charge_data:
            ax.set_title(f'Surface Charge at λ = {charge_data["wavelength"]:.1f} nm', fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_charge_multiview(self, charge_data: Dict[str, np.ndarray],
                               component: str = 'real',
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot surface charges from multiple viewing angles.

        Args:
            charge_data: Dictionary with vertices, faces, and charges
            component: Which charge component to plot
            title: Overall title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        views = [
            (0, 0, 'Front (xy)'),
            (0, 90, 'Side (xz)'),
            (90, 0, 'Top (yz)'),
            (30, 45, 'Isometric')
        ]

        fig = plt.figure(figsize=(14, 12))

        for i, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            self._plot_surface_charges_on_axis(ax, charge_data, component,
                                                view_angles=(elev, azim))
            ax.set_title(view_name, fontsize=12)

        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_mode_analysis(self, charge_data: Dict[str, np.ndarray],
                            mode_info: Dict[str, Any],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive mode analysis figure.

        Args:
            charge_data: Dictionary with surface charge data
            mode_info: Dictionary with mode identification results
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(14, 6))

        # 3D charge plot
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_surface_charges_on_axis(ax1, charge_data, 'real',
                                            view_angles=(30, 45))
        ax1.set_title('Surface Charge Distribution', fontsize=12)

        # Mode strength bar chart
        ax2 = fig.add_subplot(122)
        modes = list(mode_info['mode_strengths'].keys())
        strengths = [mode_info['mode_strengths'][m] for m in modes]

        colors = ['#2ecc71' if m == mode_info['dominant_mode'] else '#3498db'
                  for m in modes]

        bars = ax2.bar(modes, strengths, color=colors, edgecolor='black')
        ax2.set_ylabel('Relative Strength', fontsize=12)
        ax2.set_title(f"Mode Analysis: {mode_info['dominant_mode'].capitalize()}", fontsize=12)

        # Add text annotation
        info_text = (f"Dipole: {mode_info['dipole_strength']:.2e}\n"
                     f"Quadrupole: {mode_info['quadrupole_strength']:.2e}\n"
                     f"Charge asymmetry: {mode_info['charge_asymmetry']:.2f}")
        ax2.text(0.95, 0.95, info_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_charge_cross_section(self, charge_data: Dict[str, np.ndarray],
                                   plane: str = 'xy',
                                   position: float = 0.0,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot charge distribution on a 2D cross-section plane.

        Args:
            charge_data: Dictionary with positions and charges
            plane: 'xy', 'xz', or 'yz'
            position: Position of the plane along the perpendicular axis
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        positions = charge_data['positions']
        charges = charge_data['charge_real']

        # Select faces near the plane
        tolerance = 5.0  # nm

        if plane == 'xy':
            mask = np.abs(positions[:, 2] - position) < tolerance
            x, y = positions[mask, 0], positions[mask, 1]
            xlabel, ylabel = 'x (nm)', 'y (nm)'
        elif plane == 'xz':
            mask = np.abs(positions[:, 1] - position) < tolerance
            x, y = positions[mask, 0], positions[mask, 2]
            xlabel, ylabel = 'x (nm)', 'z (nm)'
        else:  # yz
            mask = np.abs(positions[:, 0] - position) < tolerance
            x, y = positions[mask, 1], positions[mask, 2]
            xlabel, ylabel = 'y (nm)', 'z (nm)'

        values = charges[mask]

        if len(x) == 0:
            ax.text(0.5, 0.5, 'No data in selected plane',
                    transform=ax.transAxes, ha='center')
        else:
            vmax = np.max(np.abs(values))
            scatter = ax.scatter(x, y, c=values, cmap='RdBu_r',
                                 vmin=-vmax, vmax=vmax, s=50)
            plt.colorbar(scatter, ax=ax, label='Surface Charge')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_aspect('equal')

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Charge Distribution ({plane} plane at {position:.1f} nm)', fontsize=14)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_charge_comparison(self, charge_data_dict: Dict[int, Dict[str, np.ndarray]],
                                component: str = 'real',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare charge distributions for multiple polarizations.

        Args:
            charge_data_dict: Dictionary mapping polarization index to charge data
            component: Which charge component to plot
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        n_pol = len(charge_data_dict)
        fig = plt.figure(figsize=(6 * n_pol, 5))

        for i, (pol_idx, charge_data) in enumerate(sorted(charge_data_dict.items())):
            ax = fig.add_subplot(1, n_pol, i + 1, projection='3d')
            self._plot_surface_charges_on_axis(ax, charge_data, component,
                                                view_angles=(30, 45))
            ax.set_title(f'Polarization {pol_idx + 1}', fontsize=12)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _plot_surface_charges_on_axis(self, ax: plt.Axes,
                                       charge_data: Dict[str, np.ndarray],
                                       component: str = 'real',
                                       view_angles: Tuple[float, float] = (30, 45)):
        """Helper method to plot surface charges on a given axis."""
        vertices = charge_data['vertices']
        faces = charge_data['faces']

        if component == 'real':
            values = charge_data['charge_real']
        elif component == 'imag':
            values = charge_data['charge_imag']
        elif component == 'magnitude':
            values = charge_data['charge_magnitude']
        else:
            values = charge_data['charge_real']

        # Create polygons
        polys = []
        face_values = []

        for i, face in enumerate(faces):
            if len(face) >= 3:
                verts = vertices[face[:3]]  # Take first 3 vertices for triangle
                polys.append(verts)
                face_values.append(values[i] if i < len(values) else 0)

        face_values = np.array(face_values)

        # Handle empty face_values array
        if len(face_values) == 0:
            ax.text2D(0.5, 0.5, 'No face data available',
                      transform=ax.transAxes, ha='center', va='center', fontsize=10)
            return

        vmax = np.max(np.abs(face_values))
        if vmax == 0:
            vmax = 1.0  # Prevent division by zero in normalization
        if component in ['real', 'imag']:
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = plt.cm.hot

        colors = cmap(norm(face_values))

        poly_collection = Poly3DCollection(polys, facecolors=colors,
                                            edgecolors='gray', linewidths=0.1,
                                            alpha=0.9)
        ax.add_collection3d(poly_collection)

        # Set limits
        x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
        y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
        z_range = [vertices[:, 2].min(), vertices[:, 2].max()]

        max_range = max(x_range[1] - x_range[0],
                        y_range[1] - y_range[0],
                        z_range[1] - z_range[0]) / 2

        mid_x = (x_range[0] + x_range[1]) / 2
        mid_y = (y_range[0] + y_range[1]) / 2
        mid_z = (z_range[0] + z_range[1]) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')

    def _save_figure(self, fig: plt.Figure, base_path: str):
        """Save figure in configured formats."""
        formats = self.formats if isinstance(self.formats, list) else [self.formats]

        for fmt in formats:
            filepath = f"{base_path}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
