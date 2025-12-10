"""
BEM Solver wrapper for pyMNPBEM-based simulations.

Provides a high-level interface to:
- BEM solver setup (quasistatic, retarded, with/without substrate)
- Excitation generation (plane wave, dipole, EELS)
- Spectrum calculation (scattering, absorption, extinction)
- Solution extraction (surface charges, currents)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
import sys


class BEMSolver:
    """
    High-level BEM solver wrapper for plasmonic simulations.

    Supports:
    - Quasistatic (small particles) and retarded (large particles) simulations
    - Plane wave, dipole, and EELS excitations
    - Multi-polarization spectrum calculations
    - Substrate effects (layer BEM)
    """

    def __init__(self, sim_config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize the BEM solver.

        Args:
            sim_config: Dictionary containing simulation parameters
            pymnpbem_path: Path to pyMNPBEM installation (optional)
        """
        self.config = sim_config

        # Import pyMNPBEM modules
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        self._import_modules()

        # Store solver state
        self.particle = None
        self.bem = None
        self.excitations = []
        self.wavelengths = None

    def _import_modules(self):
        """Import required pyMNPBEM modules."""
        try:
            from mnpbem import ComParticle, bemoptions
            from mnpbem.bem import BEMStat, BEMRet
            from mnpbem.simulation import (
                PlaneWaveStat, PlaneWaveRet,
                DipoleStat, DipoleRet,
                SpectrumStat, SpectrumRet
            )

            self.ComParticle = ComParticle
            self.bemoptions = bemoptions
            self.BEMStat = BEMStat
            self.BEMRet = BEMRet
            self.PlaneWaveStat = PlaneWaveStat
            self.PlaneWaveRet = PlaneWaveRet
            self.DipoleStat = DipoleStat
            self.DipoleRet = DipoleRet
            self.SpectrumStat = SpectrumStat
            self.SpectrumRet = SpectrumRet
            self._pymnpbem_available = True

            # Try to import layer solvers
            try:
                from mnpbem.bem import BEMStatLayer, BEMRetLayer
                self.BEMStatLayer = BEMStatLayer
                self.BEMRetLayer = BEMRetLayer
                self._layer_available = True
            except ImportError:
                self._layer_available = False

            # Try to import EELS
            try:
                from mnpbem.simulation import EELSStat, EELSRet
                self.EELSStat = EELSStat
                self.EELSRet = EELSRet
                self._eels_available = True
            except ImportError:
                self._eels_available = False

        except ImportError as e:
            print(f"Warning: Could not import pyMNPBEM modules: {e}")
            self._pymnpbem_available = False

    def setup(self, particles: List[Any], inout: List[List[int]], epstab: List[Any],
              substrate: Optional[Dict] = None):
        """
        Set up the BEM solver with particles and materials.

        Args:
            particles: List of Particle objects
            inout: List of [inside, outside] material indices
            epstab: List of dielectric function objects
            substrate: Optional substrate configuration
        """
        # Create ComParticle
        self.particle = self.ComParticle(epstab, particles, inout, closed=1)

        # Get simulation options
        sim_type = self.config.get('simulation_type', 'stat')
        interp = self.config.get('interp', 'curv')
        waitbar = self.config.get('waitbar', 0)

        # Create options
        self.options = self.bemoptions(sim=sim_type, interp=interp, waitbar=waitbar)

        # Select appropriate BEM solver
        use_substrate = substrate is not None

        if sim_type == 'stat':
            if use_substrate and self._layer_available:
                self.bem = self.BEMStatLayer(self.particle, substrate['position'])
            else:
                self.bem = self.BEMStat(self.particle)
        else:  # 'ret'
            if use_substrate and self._layer_available:
                self.bem = self.BEMRetLayer(self.particle, substrate['position'])
            else:
                self.bem = self.BEMRet(self.particle)

        # Set up wavelengths
        wl_range = self.config.get('wavelength_range', [400, 800, 100])
        self.wavelengths = np.linspace(wl_range[0], wl_range[1], wl_range[2])

        # Set up excitations
        self._setup_excitations()

    def _setup_excitations(self):
        """Set up excitation objects based on configuration."""
        exc_type = self.config.get('excitation_type', 'planewave')
        sim_type = self.config.get('simulation_type', 'stat')

        if exc_type == 'planewave':
            self._setup_planewave_excitations(sim_type)
        elif exc_type == 'dipole':
            self._setup_dipole_excitations(sim_type)
        elif exc_type == 'eels':
            self._setup_eels_excitations(sim_type)
        else:
            raise ValueError(f"Unknown excitation type: {exc_type}")

    def _setup_planewave_excitations(self, sim_type: str):
        """Set up plane wave excitations."""
        polarizations = self.config.get('polarizations', [[1, 0, 0]])
        propagation_dirs = self.config.get('propagation_dirs', [[0, 0, 1]])

        # Ensure same length
        if len(polarizations) != len(propagation_dirs):
            if len(propagation_dirs) == 1:
                propagation_dirs = propagation_dirs * len(polarizations)
            else:
                raise ValueError("Number of polarizations and propagation directions must match")

        self.excitations = []
        PlaneWave = self.PlaneWaveStat if sim_type == 'stat' else self.PlaneWaveRet

        for pol, prop_dir in zip(polarizations, propagation_dirs):
            if sim_type == 'stat':
                # Quasistatic: only polarization matters
                exc = PlaneWave(pol=pol)
            else:
                # Retarded: both polarization and direction matter
                exc = PlaneWave(pol=pol, direction=prop_dir)
            self.excitations.append(exc)

    def _setup_dipole_excitations(self, sim_type: str):
        """Set up dipole excitations."""
        position = self.config.get('dipole_position', [0, 0, 15])
        moment = self.config.get('dipole_moment', [0, 0, 1])

        Dipole = self.DipoleStat if sim_type == 'stat' else self.DipoleRet

        pt = np.array([position])
        dip = np.array([moment])

        exc = Dipole(pt=pt, dip=dip)
        self.excitations = [exc]

    def _setup_eels_excitations(self, sim_type: str):
        """Set up EELS excitations."""
        if not self._eels_available:
            raise ImportError("EELS excitation not available in this pyMNPBEM version")

        impact = self.config.get('impact_parameter', [10, 0])
        beam_energy = self.config.get('beam_energy', 200e3)
        beam_width = self.config.get('beam_width', 0.2)

        EELS = self.EELSStat if sim_type == 'stat' else self.EELSRet

        exc = EELS(impact=impact, beam_energy=beam_energy, width=beam_width)
        self.excitations = [exc]

    def compute_spectrum(self, show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute optical cross-section spectra.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with:
                - 'wavelengths': Wavelength array (nm)
                - 'scattering': Scattering cross-section (n_wavelengths, n_excitations)
                - 'absorption': Absorption cross-section (n_wavelengths, n_excitations)
                - 'extinction': Extinction cross-section (n_wavelengths, n_excitations)
        """
        sim_type = self.config.get('simulation_type', 'stat')
        Spectrum = self.SpectrumStat if sim_type == 'stat' else self.SpectrumRet

        n_wl = len(self.wavelengths)
        n_exc = len(self.excitations)

        scattering = np.zeros((n_wl, n_exc))
        absorption = np.zeros((n_wl, n_exc))
        extinction = np.zeros((n_wl, n_exc))

        # Store solutions for field calculation
        self.solutions = {}

        for i, exc in enumerate(self.excitations):
            if show_progress:
                print(f"Computing spectrum for excitation {i+1}/{n_exc}")

            # Create spectrum calculator
            spec = Spectrum(self.bem, exc, self.wavelengths, show_progress=show_progress)

            # Compute
            result = spec.compute()

            if isinstance(result, tuple):
                sca, ext = result
                scattering[:, i] = sca.flatten()
                extinction[:, i] = ext.flatten()
                absorption[:, i] = ext.flatten() - sca.flatten()
            elif isinstance(result, dict):
                scattering[:, i] = result.get('sca', np.zeros(n_wl)).flatten()
                extinction[:, i] = result.get('ext', np.zeros(n_wl)).flatten()
                absorption[:, i] = result.get('abs', np.zeros(n_wl)).flatten()

            # Store spectrum object for later use
            self.solutions[i] = spec

        return {
            'wavelengths': self.wavelengths,
            'scattering': scattering,
            'absorption': absorption,
            'extinction': extinction,
        }

    def solve_at_wavelength(self, wavelength: float, excitation_idx: int = 0) -> Any:
        """
        Solve BEM equations at a specific wavelength.

        Args:
            wavelength: Wavelength in nm
            excitation_idx: Index of excitation to use

        Returns:
            BEM solution object (sig)
        """
        exc = self.excitations[excitation_idx]

        # Generate excitation at this wavelength
        exc_struct = exc(self.particle, wavelength)

        # Solve
        sig = self.bem.solve(exc_struct)

        return sig

    def get_surface_charges(self, wavelength: float, excitation_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get surface charge distribution at a specific wavelength.

        Args:
            wavelength: Wavelength in nm
            excitation_idx: Index of excitation

        Returns:
            Tuple of (positions, charges):
                - positions: Face centroid positions (n_faces, 3)
                - charges: Complex surface charges (n_faces,)
        """
        sig = self.solve_at_wavelength(wavelength, excitation_idx)

        # Get face positions
        positions = self.particle.pos  # Face centroids
        charges = sig.sig.flatten()

        return positions, charges

    def get_peak_wavelength(self, spectrum_data: Dict[str, np.ndarray],
                            excitation_idx: int = 0,
                            spectrum_type: str = 'extinction') -> float:
        """
        Find the peak wavelength from spectrum data.

        Args:
            spectrum_data: Dictionary from compute_spectrum()
            excitation_idx: Which excitation to analyze
            spectrum_type: 'extinction', 'absorption', or 'scattering'

        Returns:
            Peak wavelength in nm
        """
        wavelengths = spectrum_data['wavelengths']
        spectrum = spectrum_data[spectrum_type][:, excitation_idx]

        peak_idx = np.argmax(spectrum)
        return wavelengths[peak_idx]

    def compute_unpolarized(self, spectrum_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute unpolarized spectrum from orthogonal polarizations.

        Args:
            spectrum_data: Dictionary from compute_spectrum()

        Returns:
            Dictionary with unpolarized spectra
        """
        n_exc = spectrum_data['scattering'].shape[1]

        if n_exc < 2:
            return spectrum_data

        # Average over polarizations (incoherent averaging)
        if n_exc == 2:
            # Two orthogonal polarizations
            factor = 0.5
        elif n_exc == 3:
            # Three orthogonal dipole moments
            factor = 1.0 / 3.0
        else:
            factor = 1.0 / n_exc

        unpol_sca = np.mean(spectrum_data['scattering'], axis=1)
        unpol_abs = np.mean(spectrum_data['absorption'], axis=1)
        unpol_ext = np.mean(spectrum_data['extinction'], axis=1)

        return {
            'wavelengths': spectrum_data['wavelengths'],
            'scattering_unpolarized': unpol_sca,
            'absorption_unpolarized': unpol_abs,
            'extinction_unpolarized': unpol_ext,
        }

    def get_particle_info(self) -> Dict[str, Any]:
        """Get information about the particle."""
        if self.particle is None:
            return {}

        return {
            'n_faces': self.particle.n_faces,
            'n_particles': len(self.particle.particles) if hasattr(self.particle, 'particles') else 1,
        }

    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about the solver setup."""
        return {
            'simulation_type': self.config.get('simulation_type', 'stat'),
            'excitation_type': self.config.get('excitation_type', 'planewave'),
            'n_wavelengths': len(self.wavelengths) if self.wavelengths is not None else 0,
            'n_excitations': len(self.excitations),
            'wavelength_range': [float(self.wavelengths[0]), float(self.wavelengths[-1])] if self.wavelengths is not None else None,
        }
