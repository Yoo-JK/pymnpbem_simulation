"""
Surface Charge Calculator for pyMNPBEM-based simulations.

Computes and analyzes surface charge distributions for:
- Mode identification (dipolar, quadrupolar, etc.)
- Charge density visualization
- Plasmon mode analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import sys


class SurfaceChargeCalculator:
    """
    Surface charge distribution calculator using pyMNPBEM.

    Supports:
    - Surface charge extraction from BEM solutions
    - Charge density normalization
    - Mode classification (dipolar, quadrupolar, etc.)
    - Data preparation for 3D visualization
    """

    def __init__(self, sim_config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize the surface charge calculator.

        Args:
            sim_config: Dictionary containing simulation parameters
            pymnpbem_path: Path to pyMNPBEM installation (optional)
        """
        self.config = sim_config

        # Import pyMNPBEM modules
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        self._pymnpbem_available = True

    def compute_surface_charges(self, bem_solver: Any, wavelength: float,
                                 excitation_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Compute surface charge distribution at a specific wavelength.

        Args:
            bem_solver: BEMSolver object with setup complete
            wavelength: Wavelength in nm
            excitation_idx: Index of excitation

        Returns:
            Dictionary with:
                - 'positions': Face centroid positions (n_faces, 3)
                - 'normals': Face normal vectors (n_faces, 3)
                - 'areas': Face areas (n_faces,)
                - 'charges': Complex surface charges (n_faces,)
                - 'charge_real': Real part of charges
                - 'charge_imag': Imaginary part of charges
                - 'charge_magnitude': |charge|
                - 'charge_phase': Phase of complex charge
                - 'vertices': All mesh vertices
                - 'faces': Face connectivity
        """
        # Solve BEM at this wavelength
        sig = bem_solver.solve_at_wavelength(wavelength, excitation_idx)

        # Get particle information
        particle = bem_solver.particle

        # Extract surface data
        positions = particle.pos  # Face centroids
        normals = particle.nvec  # Normal vectors
        areas = particle.area    # Face areas
        charges = sig.sig.flatten()  # Surface charges

        # Get mesh data for visualization
        # Combine all particles if multiple
        all_verts = []
        all_faces = []
        vert_offset = 0

        for p in particle.particles:
            all_verts.append(p.verts)
            all_faces.append(p.faces + vert_offset)
            vert_offset += len(p.verts)

        vertices = np.vstack(all_verts)
        faces = np.vstack(all_faces)

        # Compute derived quantities
        charge_real = np.real(charges)
        charge_imag = np.imag(charges)
        charge_magnitude = np.abs(charges)
        charge_phase = np.angle(charges)

        return {
            'positions': positions,
            'normals': normals,
            'areas': areas,
            'charges': charges,
            'charge_real': charge_real,
            'charge_imag': charge_imag,
            'charge_magnitude': charge_magnitude,
            'charge_phase': charge_phase,
            'vertices': vertices,
            'faces': faces,
            'wavelength': wavelength,
            'excitation_idx': excitation_idx,
        }

    def compute_charges_at_peaks(self, bem_solver: Any,
                                  spectrum_data: Dict[str, np.ndarray],
                                  show_progress: bool = True) -> Dict[int, Dict[str, Any]]:
        """
        Compute surface charges at peak wavelengths for mode analysis.

        Args:
            bem_solver: BEMSolver object
            spectrum_data: Dictionary from bem_solver.compute_spectrum()
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping excitation_idx to surface charge data
        """
        charge_data = {}

        n_exc = spectrum_data['scattering'].shape[1]

        for exc_idx in range(n_exc):
            if show_progress:
                print(f"Computing surface charges for excitation {exc_idx+1}/{n_exc}")

            # Find peak wavelength
            peak_wl = bem_solver.get_peak_wavelength(spectrum_data, exc_idx, 'extinction')

            # Compute charges
            charges = self.compute_surface_charges(bem_solver, peak_wl, exc_idx)
            charge_data[exc_idx] = charges

        return charge_data

    def identify_mode(self, charge_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Attempt to identify plasmon mode from charge distribution.

        Uses multipole expansion analysis:
        - Dipolar: charge distribution has one node (positive/negative split)
        - Quadrupolar: charge distribution has two nodes
        - Hexapolar: three nodes
        - etc.

        Args:
            charge_data: Dictionary from compute_surface_charges()

        Returns:
            Dictionary with mode identification information
        """
        charges = charge_data['charge_real']
        positions = charge_data['positions']
        areas = charge_data['areas']

        # Compute dipole moment
        dipole = self._compute_dipole_moment(positions, charges, areas)
        dipole_strength = np.linalg.norm(dipole)

        # Compute quadrupole tensor
        quadrupole = self._compute_quadrupole_tensor(positions, charges, areas)
        quadrupole_strength = np.sqrt(np.sum(quadrupole**2))

        # Compute octupole (simplified)
        octupole_strength = self._compute_octupole_strength(positions, charges, areas)

        # Determine dominant mode
        strengths = {
            'dipolar': dipole_strength,
            'quadrupolar': quadrupole_strength,
            'octupolar': octupole_strength
        }

        # Normalize
        total = sum(strengths.values()) + 1e-10
        normalized = {k: v/total for k, v in strengths.items()}

        dominant_mode = max(normalized, key=normalized.get)

        # Count charge sign changes (nodes)
        n_positive = np.sum(charges > 0)
        n_negative = np.sum(charges < 0)
        asymmetry = abs(n_positive - n_negative) / (n_positive + n_negative + 1e-10)

        return {
            'dominant_mode': dominant_mode,
            'mode_strengths': normalized,
            'dipole_moment': dipole.tolist(),
            'dipole_strength': float(dipole_strength),
            'quadrupole_strength': float(quadrupole_strength),
            'octupole_strength': float(octupole_strength),
            'charge_asymmetry': float(asymmetry),
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
        }

    def _compute_dipole_moment(self, positions: np.ndarray, charges: np.ndarray,
                                areas: np.ndarray) -> np.ndarray:
        """Compute electric dipole moment."""
        # p = integral(r * sigma * dA)
        weighted_charges = charges * areas
        dipole = np.sum(positions * weighted_charges[:, np.newaxis], axis=0)
        return dipole

    def _compute_quadrupole_tensor(self, positions: np.ndarray, charges: np.ndarray,
                                    areas: np.ndarray) -> np.ndarray:
        """Compute quadrupole tensor."""
        # Q_ij = integral((3*r_i*r_j - r^2*delta_ij) * sigma * dA)
        weighted_charges = charges * areas
        r2 = np.sum(positions**2, axis=1)

        Q = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                Q[i, j] = np.sum((3 * positions[:, i] * positions[:, j] - r2 * delta_ij) *
                                 weighted_charges)
        return Q

    def _compute_octupole_strength(self, positions: np.ndarray, charges: np.ndarray,
                                    areas: np.ndarray) -> float:
        """Compute simplified octupole strength."""
        weighted_charges = charges * areas
        r = np.linalg.norm(positions, axis=1)
        r3 = r**3

        # Simplified octupole: sum of r^3 * charge
        octupole = np.sum(r3 * np.abs(weighted_charges))
        return float(octupole)

    def get_charge_statistics(self, charge_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics of surface charge distribution.

        Args:
            charge_data: Dictionary from compute_surface_charges()

        Returns:
            Dictionary with statistics
        """
        charges = charge_data['charges']
        charge_real = charge_data['charge_real']
        charge_magnitude = charge_data['charge_magnitude']
        areas = charge_data['areas']

        # Total charge (should be ~0 for closed surface)
        total_charge = np.sum(charge_real * areas)

        # RMS charge
        rms_charge = np.sqrt(np.mean(charge_magnitude**2))

        # Maximum values
        max_positive = np.max(charge_real)
        max_negative = np.min(charge_real)

        # Charge contrast
        contrast = (max_positive - max_negative) / (max_positive + np.abs(max_negative) + 1e-10)

        return {
            'total_charge': float(total_charge),
            'rms_charge': float(rms_charge),
            'max_positive': float(max_positive),
            'max_negative': float(max_negative),
            'max_magnitude': float(np.max(charge_magnitude)),
            'mean_magnitude': float(np.mean(charge_magnitude)),
            'charge_contrast': float(contrast),
        }

    def prepare_visualization_data(self, charge_data: Dict[str, np.ndarray],
                                    component: str = 'real') -> Dict[str, Any]:
        """
        Prepare data for 3D visualization of surface charges.

        Args:
            charge_data: Dictionary from compute_surface_charges()
            component: 'real', 'imag', 'magnitude', or 'phase'

        Returns:
            Dictionary suitable for matplotlib 3D plotting or export
        """
        vertices = charge_data['vertices']
        faces = charge_data['faces']

        if component == 'real':
            values = charge_data['charge_real']
        elif component == 'imag':
            values = charge_data['charge_imag']
        elif component == 'magnitude':
            values = charge_data['charge_magnitude']
        elif component == 'phase':
            values = charge_data['charge_phase']
        else:
            values = charge_data['charge_real']

        # Normalize for colormap
        vmax = np.max(np.abs(values))
        if vmax > 0:
            normalized = values / vmax
        else:
            normalized = values

        return {
            'vertices': vertices,
            'faces': faces,
            'values': values,
            'values_normalized': normalized,
            'value_range': [-float(vmax), float(vmax)],
            'component': component,
            'wavelength': charge_data['wavelength'],
        }

    def compute_mode_spectrum(self, bem_solver: Any,
                               wavelengths: Optional[np.ndarray] = None,
                               excitation_idx: int = 0,
                               show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute mode decomposition as function of wavelength.

        This tracks how multipole contributions vary across the spectrum.

        Args:
            bem_solver: BEMSolver object
            wavelengths: Array of wavelengths (uses solver wavelengths if None)
            excitation_idx: Index of excitation
            show_progress: Whether to show progress

        Returns:
            Dictionary with wavelength-dependent mode information
        """
        if wavelengths is None:
            wavelengths = bem_solver.wavelengths

        n_wl = len(wavelengths)

        dipole_strength = np.zeros(n_wl)
        quadrupole_strength = np.zeros(n_wl)
        octupole_strength = np.zeros(n_wl)

        iterator = enumerate(wavelengths)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Computing mode spectrum")

        for i, wl in iterator:
            charge_data = self.compute_surface_charges(bem_solver, wl, excitation_idx)
            mode_info = self.identify_mode(charge_data)

            dipole_strength[i] = mode_info['dipole_strength']
            quadrupole_strength[i] = mode_info['quadrupole_strength']
            octupole_strength[i] = mode_info['octupole_strength']

        return {
            'wavelengths': wavelengths,
            'dipole_strength': dipole_strength,
            'quadrupole_strength': quadrupole_strength,
            'octupole_strength': octupole_strength,
        }
