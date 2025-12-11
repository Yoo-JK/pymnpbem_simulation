"""
Field Calculator for pyMNPBEM-based simulations.

Computes electric field distributions at arbitrary points:
- 2D slices (e.g., xy-plane, xz-plane)
- 3D grids
- Field enhancement |E|^2/|E0|^2
- Near-field intensity distributions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
import sys


class FieldCalculator:
    """
    Electric field calculator using pyMNPBEM.

    Supports:
    - Arbitrary point calculations
    - 2D and 3D grid calculations
    - Field enhancement computation
    - Automatic grid generation around particles
    """

    def __init__(self, sim_config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize the field calculator.

        Args:
            sim_config: Dictionary containing simulation parameters
            pymnpbem_path: Path to pyMNPBEM installation (optional)
        """
        self.config = sim_config

        # Import pyMNPBEM modules
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        try:
            from mnpbem.particles import Point
            self.Point = Point
            self._pymnpbem_available = True
        except ImportError as e:
            print(f"Warning: Could not import pyMNPBEM modules: {e}")
            self._pymnpbem_available = False

        # Field calculation parameters
        self.mindist = self.config.get('field_mindist', 0.5)

    def compute_field_grid(self, bem_solver: Any, wavelength: float,
                           excitation_idx: int = 0,
                           show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute electric field on a grid.

        Args:
            bem_solver: BEMSolver object with setup complete
            wavelength: Wavelength in nm
            excitation_idx: Index of excitation
            show_progress: Whether to show progress

        Returns:
            Dictionary with:
                - 'x', 'y', 'z': Grid coordinate arrays
                - 'X', 'Y', 'Z': Meshgrid arrays
                - 'Ex', 'Ey', 'Ez': Electric field components (complex)
                - 'E_magnitude': |E| field magnitude
                - 'enhancement': |E|^2/|E0|^2 field enhancement
        """
        # Get field region from config
        field_region = self.config.get('field_region', {
            'x_range': [-50, 50, 101],
            'y_range': [0, 0, 1],
            'z_range': [-50, 50, 101]
        })

        # Create grid
        x = np.linspace(field_region['x_range'][0],
                        field_region['x_range'][1],
                        int(field_region['x_range'][2]))
        y = np.linspace(field_region['y_range'][0],
                        field_region['y_range'][1],
                        int(field_region['y_range'][2]))
        z = np.linspace(field_region['z_range'][0],
                        field_region['z_range'][1],
                        int(field_region['z_range'][2]))

        # Create meshgrid
        if len(y) == 1:
            # XZ plane (y constant)
            X, Z = np.meshgrid(x, z)
            Y = np.full_like(X, y[0])
        elif len(z) == 1:
            # XY plane (z constant)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, z[0])
        elif len(x) == 1:
            # YZ plane (x constant)
            Y, Z = np.meshgrid(y, z)
            X = np.full_like(Y, x[0])
        else:
            # Full 3D grid
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Flatten for computation
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Filter points inside particle (keep only points outside)
        mask = self._filter_points_outside_particle(points, bem_solver.particle)

        # Compute field
        Ex, Ey, Ez = self._compute_field_at_points(
            bem_solver, points[mask], wavelength, excitation_idx, show_progress
        )

        # Create full arrays with zeros inside particle
        Ex_full = np.zeros(len(points), dtype=complex)
        Ey_full = np.zeros(len(points), dtype=complex)
        Ez_full = np.zeros(len(points), dtype=complex)

        Ex_full[mask] = Ex
        Ey_full[mask] = Ey
        Ez_full[mask] = Ez

        # Reshape to grid
        shape = X.shape
        Ex_grid = Ex_full.reshape(shape)
        Ey_grid = Ey_full.reshape(shape)
        Ez_grid = Ez_full.reshape(shape)

        # Compute derived quantities
        E_magnitude = np.sqrt(np.abs(Ex_grid)**2 + np.abs(Ey_grid)**2 + np.abs(Ez_grid)**2)

        # Enhancement: |E|^2 / |E0|^2
        # For plane wave, |E0| = 1 (normalized)
        enhancement = np.abs(Ex_grid)**2 + np.abs(Ey_grid)**2 + np.abs(Ez_grid)**2

        return {
            'x': x,
            'y': y,
            'z': z,
            'X': X,
            'Y': Y,
            'Z': Z,
            'Ex': Ex_grid,
            'Ey': Ey_grid,
            'Ez': Ez_grid,
            'E_magnitude': E_magnitude,
            'enhancement': enhancement,
            'wavelength': wavelength,
            'excitation_idx': excitation_idx,
        }

    def _compute_field_at_points(self, bem_solver: Any, points: np.ndarray,
                                  wavelength: float, excitation_idx: int,
                                  show_progress: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute electric field at arbitrary points.

        Args:
            bem_solver: BEMSolver object
            points: Array of positions (n_points, 3)
            wavelength: Wavelength in nm
            excitation_idx: Index of excitation

        Returns:
            Tuple of (Ex, Ey, Ez) arrays
        """
        if len(points) == 0:
            return np.array([]), np.array([]), np.array([])

        # Solve BEM at this wavelength
        sig = bem_solver.solve_at_wavelength(wavelength, excitation_idx)

        # Create Point object
        pts = self.Point(points)

        # Compute field using BEM
        field = bem_solver.bem.field(sig, pts)

        # field shape is (n_points, 3) for [Ex, Ey, Ez]
        if field.ndim == 2 and field.shape[1] == 3:
            Ex = field[:, 0]
            Ey = field[:, 1]
            Ez = field[:, 2]
        else:
            # Handle different output shapes
            Ex = field.flatten()
            Ey = np.zeros_like(Ex)
            Ez = np.zeros_like(Ex)

        # Add incident field for plane wave excitation
        exc = bem_solver.excitations[excitation_idx]
        if hasattr(exc, 'pol'):
            pol = np.array(exc.pol)
            # Handle 2D polarization array (n_pol, 3) - take first polarization
            if pol.ndim == 2:
                pol = pol[0]
            # Plane wave: E_inc = E0 * pol (normalized to 1)
            Ex = Ex + pol[0]
            Ey = Ey + pol[1]
            Ez = Ez + pol[2]

        return Ex, Ey, Ez

    def _filter_points_outside_particle(self, points: np.ndarray,
                                         particle: Any) -> np.ndarray:
        """
        Create mask for points outside the particle.

        Uses minimum distance to particle surface.
        """
        # Get particle centroid positions
        particle_pos = particle.pos  # Face centroids

        # For each point, find minimum distance to any particle face
        n_points = len(points)
        mask = np.ones(n_points, dtype=bool)

        # Use vectorized distance calculation
        for i, pt in enumerate(points):
            distances = np.linalg.norm(particle_pos - pt, axis=1)
            if np.min(distances) < self.mindist:
                mask[i] = False

        return mask

    def compute_field_at_peaks(self, bem_solver: Any, spectrum_data: Dict[str, np.ndarray],
                                show_progress: bool = True) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Compute field at peak wavelength for each excitation.

        Args:
            bem_solver: BEMSolver object
            spectrum_data: Dictionary from bem_solver.compute_spectrum()
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping excitation_idx to field data
        """
        field_data = {}

        field_wl_idx = self.config.get('field_wavelength_idx', 'peak')

        n_exc = spectrum_data['scattering'].shape[1]

        for exc_idx in range(n_exc):
            if show_progress:
                print(f"Computing field for excitation {exc_idx+1}/{n_exc}")

            # Determine wavelength
            if field_wl_idx == 'peak' or field_wl_idx == 'peak_abs':
                wl = bem_solver.get_peak_wavelength(spectrum_data, exc_idx, 'absorption')
            elif field_wl_idx == 'peak_ext':
                wl = bem_solver.get_peak_wavelength(spectrum_data, exc_idx, 'extinction')
            elif field_wl_idx == 'peak_sca':
                wl = bem_solver.get_peak_wavelength(spectrum_data, exc_idx, 'scattering')
            elif field_wl_idx == 'middle':
                wavelengths = spectrum_data['wavelengths']
                wl = wavelengths[len(wavelengths)//2]
            elif isinstance(field_wl_idx, int):
                wl = spectrum_data['wavelengths'][field_wl_idx]
            else:
                wl = bem_solver.get_peak_wavelength(spectrum_data, exc_idx, 'extinction')

            # Compute field
            field = self.compute_field_grid(bem_solver, wl, exc_idx, show_progress)
            field_data[exc_idx] = field

        return field_data

    def compute_unpolarized_field(self, field_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Compute unpolarized field enhancement from orthogonal polarizations.

        Uses FDTD convention: I_unpol = (I_pol1 + I_pol2) / 2

        Args:
            field_data: Dictionary of field data for each excitation

        Returns:
            Dictionary with unpolarized field enhancement
        """
        if len(field_data) < 2:
            # Only one polarization - return as is
            return field_data[0]

        # Get reference grid
        ref = field_data[0]

        # Sum intensities (enhancement = |E|^2/|E0|^2)
        total_enhancement = np.zeros_like(ref['enhancement'])

        for exc_idx, data in field_data.items():
            total_enhancement += data['enhancement']

        # Average
        unpol_enhancement = total_enhancement / len(field_data)

        return {
            'x': ref['x'],
            'y': ref['y'],
            'z': ref['z'],
            'X': ref['X'],
            'Y': ref['Y'],
            'Z': ref['Z'],
            'enhancement': np.sqrt(unpol_enhancement),  # |E|/|E0|
            'intensity': unpol_enhancement,  # |E|^2/|E0|^2
            'wavelength': 'unpolarized',
        }

    def find_hotspots(self, field_data: Dict[str, np.ndarray],
                       n_hotspots: int = 10,
                       min_distance: int = 3) -> List[Dict[str, Any]]:
        """
        Find field hotspot locations.

        Args:
            field_data: Dictionary from compute_field_grid()
            n_hotspots: Number of hotspots to find
            min_distance: Minimum grid-point distance between hotspots

        Returns:
            List of hotspot dictionaries with position and enhancement
        """
        from scipy.ndimage import maximum_filter

        enhancement = field_data['enhancement']

        # Find local maxima
        local_max = maximum_filter(enhancement, size=min_distance)
        is_max = (enhancement == local_max)

        # Get positions of maxima
        max_positions = np.where(is_max)
        max_values = enhancement[is_max]

        # Sort by value
        sort_idx = np.argsort(max_values)[::-1]

        hotspots = []
        for i in range(min(n_hotspots, len(sort_idx))):
            idx = sort_idx[i]

            # Get grid indices
            if enhancement.ndim == 2:
                gi, gj = max_positions[0][idx], max_positions[1][idx]
                x = field_data['X'][gi, gj]
                y = field_data['Y'][gi, gj]
                z = field_data['Z'][gi, gj]
            else:
                gi, gj, gk = max_positions[0][idx], max_positions[1][idx], max_positions[2][idx]
                x = field_data['X'][gi, gj, gk]
                y = field_data['Y'][gi, gj, gk]
                z = field_data['Z'][gi, gj, gk]

            hotspots.append({
                'position': [float(x), float(y), float(z)],
                'enhancement': float(max_values[idx]),
                'grid_index': [int(gi), int(gj)] if enhancement.ndim == 2 else [int(gi), int(gj), int(gk)]
            })

        return hotspots

    def get_field_statistics(self, field_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics of field enhancement.

        Args:
            field_data: Dictionary from compute_field_grid()

        Returns:
            Dictionary with statistics
        """
        enhancement = field_data['enhancement']

        # Mask out zeros (inside particle)
        mask = enhancement > 0

        if not np.any(mask):
            return {
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
            }

        valid = enhancement[mask]

        return {
            'max': float(np.max(valid)),
            'min': float(np.min(valid)),
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid)),
            'median': float(np.median(valid)),
        }
