"""
BEM Solver wrapper for pyMNPBEM-based simulations.

Provides a high-level interface to:
- BEM solver setup (quasistatic, retarded, with/without substrate)
- Excitation generation (plane wave, dipole, EELS)
- Spectrum calculation (scattering, absorption, extinction)
- Solution extraction (surface charges, currents)
- Parallel wavelength computation (ThreadPoolExecutor)
- H-matrix compression and iterative solvers for large structures
"""

import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import sys


class BEMSolver:
    """
    High-level BEM solver wrapper for plasmonic simulations.

    Supports:
    - Quasistatic (small particles) and retarded (large particles) simulations
    - Plane wave, dipole, and EELS excitations
    - Multi-polarization spectrum calculations
    - Substrate effects (layer BEM)
    - H-matrix compression for large structures
    - Iterative solvers (GMRES, BiCGSTAB)
    - Parallel wavelength computation
    """

    def __init__(self, sim_config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize the BEM solver.

        Args:
            sim_config: Dictionary containing simulation parameters
            pymnpbem_path: Path to pyMNPBEM installation (optional)
        """
        self.config = sim_config
        self.pymnpbem_path = pymnpbem_path

        # Import pyMNPBEM modules
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        # Set up threading environment BEFORE importing numpy-dependent modules
        self._setup_threading_env()

        self._import_modules()

        # Store solver state
        self.particle = None
        self.bem = None
        self.excitations = []
        self.wavelengths = None
        self.epstab = None

        # Parallel computing settings
        self.num_cores = sim_config.get('num_cores', 1)

    def _setup_threading_env(self):
        """Set up threading environment variables for BLAS/LAPACK."""
        max_threads = str(self.config.get('max_comp_threads', 1))

        # Set all common threading environment variables
        os.environ['OMP_NUM_THREADS'] = max_threads
        os.environ['MKL_NUM_THREADS'] = max_threads
        os.environ['OPENBLAS_NUM_THREADS'] = max_threads
        os.environ['NUMEXPR_NUM_THREADS'] = max_threads
        os.environ['VECLIB_MAXIMUM_THREADS'] = max_threads

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
                from mnpbem.particles import LayerStructure
                self.BEMStatLayer = BEMStatLayer
                self.BEMRetLayer = BEMRetLayer
                self.LayerStructure = LayerStructure
                self._layer_available = True
            except ImportError:
                self._layer_available = False

            # Try to import iterative solvers
            try:
                from mnpbem.bem import BEMStatIter, BEMRetIter
                self.BEMStatIter = BEMStatIter
                self.BEMRetIter = BEMRetIter
                self._iter_available = True
            except ImportError:
                self._iter_available = False

            # Try to import H-matrix modules
            try:
                from mnpbem.greenfun.hmatrix import HMatrix, ClusterTree
                from mnpbem.greenfun.aca import aca, ACAMatrix
                self.HMatrix = HMatrix
                self.ClusterTree = ClusterTree
                self.aca = aca
                self.ACAMatrix = ACAMatrix
                self._hmatrix_available = True
            except ImportError:
                self._hmatrix_available = False

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
        # Store epstab for later use
        self.epstab = epstab

        # Create ComParticle
        self.particle = self.ComParticle(epstab, particles, inout, closed=1)

        # Get simulation options
        sim_type = self.config.get('simulation_type', 'stat')
        interp = self.config.get('interp', 'curv')
        waitbar = self.config.get('waitbar', 0)

        # Get solver options
        use_iterative = self.config.get('use_iterative_solver', False)
        use_h2 = self.config.get('use_h2_compression', False)

        # Create options
        self.options = self.bemoptions(sim=sim_type, interp=interp, waitbar=waitbar)

        # Select appropriate BEM solver
        use_substrate = substrate is not None
        self.substrate = substrate

        # Create the BEM solver based on options
        self.bem = self._create_bem_solver(sim_type, use_substrate, use_iterative, use_h2)

        # Set up wavelengths
        wl_range = self.config.get('wavelength_range', [400, 800, 100])
        self.wavelengths = np.linspace(wl_range[0], wl_range[1], wl_range[2])

        # Set up excitations
        self._setup_excitations()

    def _create_bem_solver(self, sim_type: str, use_substrate: bool,
                           use_iterative: bool, use_h2: bool) -> Any:
        """
        Create the appropriate BEM solver based on configuration.

        Args:
            sim_type: 'stat' or 'ret'
            use_substrate: Whether substrate is used
            use_iterative: Whether to use iterative solver
            use_h2: Whether to use H-matrix compression

        Returns:
            BEM solver object
        """
        # Get iterative solver parameters
        iter_tol = self.config.get('iter_tolerance', 1e-6)
        iter_maxiter = self.config.get('iter_maxiter', 1000)
        h2_tol = self.config.get('h2_tolerance', 1e-4)

        # Check availability
        if use_iterative and not self._iter_available:
            print("Warning: Iterative solver requested but not available. Using direct solver.")
            use_iterative = False

        if use_h2 and not self._hmatrix_available:
            print("Warning: H-matrix compression requested but not available. Using dense matrices.")
            use_h2 = False

        # Build solver kwargs
        solver_kwargs = {}
        if use_iterative:
            solver_kwargs['tol'] = iter_tol
            solver_kwargs['maxiter'] = iter_maxiter
        if use_h2:
            solver_kwargs['use_hmatrix'] = True
            solver_kwargs['hmatrix_tol'] = h2_tol

        # Select solver class
        if sim_type == 'stat':
            if use_substrate and self._layer_available:
                # Create LayerStructure from substrate config
                layer = self._create_layer_structure()
                # Layer solver (substrate)
                if use_iterative and hasattr(self, 'BEMStatLayerIter'):
                    return self.BEMStatLayerIter(self.particle, layer, **solver_kwargs)
                return self.BEMStatLayer(self.particle, layer)
            elif use_iterative:
                return self.BEMStatIter(self.particle, **solver_kwargs)
            else:
                return self.BEMStat(self.particle)
        else:  # 'ret'
            if use_substrate and self._layer_available:
                # Create LayerStructure from substrate config
                layer = self._create_layer_structure()
                if use_iterative and hasattr(self, 'BEMRetLayerIter'):
                    return self.BEMRetLayerIter(self.particle, layer, **solver_kwargs)
                return self.BEMRetLayer(self.particle, layer)
            elif use_iterative:
                return self.BEMRetIter(self.particle, **solver_kwargs)
            else:
                return self.BEMRet(self.particle)

    def _create_layer_structure(self):
        """
        Create a LayerStructure from substrate configuration.

        Returns:
            LayerStructure object for BEM layer solvers
        """
        if not self.substrate:
            raise ValueError("No substrate configuration provided")

        # Get substrate material dielectric function
        # epstab[0] is typically air/vacuum, epstab[1] is particle material
        # For substrate, we need: [medium above interface, substrate below]

        # Get the interface position (z-coordinate)
        z_interface = self.substrate.get('position', 0)

        # Get substrate dielectric function
        substrate_eps = self.substrate.get('eps')
        if substrate_eps is None:
            # If eps not directly provided, try to get from material name
            material = self.substrate.get('material', 'glass')
            if material == 'glass':
                from mnpbem import EpsConst
                substrate_eps = EpsConst(2.25)  # n=1.5 for glass
            elif material == 'silicon':
                from mnpbem import EpsTable
                substrate_eps = EpsTable('si')
            else:
                raise ValueError(f"Unknown substrate material: {material}")

        # Create LayerStructure: [medium, substrate]
        # Using the ambient medium (first in epstab) and substrate
        layer_eps = [self.epstab[0], substrate_eps]
        layer = self.LayerStructure(layer_eps, z_interface=np.array([z_interface]))

        return layer

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
            # Ensure 2D array shape (1, 3) for pyMNPBEM
            pol_arr = np.atleast_2d(pol)
            dir_arr = np.atleast_2d(prop_dir)

            if sim_type == 'stat':
                # Quasistatic: only polarization matters
                exc = PlaneWave(pol=pol_arr)
            else:
                # Retarded: both polarization and direction matter
                exc = PlaneWave(pol=pol_arr, dir=dir_arr)
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

    def _solve_single_wavelength(self, wl_idx: int, exc_idx: int) -> Tuple[int, int, float, float, float]:
        """
        Solve BEM at a single wavelength for a single excitation.

        This method is designed to be called in parallel.

        Args:
            wl_idx: Wavelength index
            exc_idx: Excitation index

        Returns:
            Tuple of (wl_idx, exc_idx, scattering, absorption, extinction)
        """
        wavelength = self.wavelengths[wl_idx]
        exc = self.excitations[exc_idx]

        # Generate excitation at this wavelength (also updates exc internal state)
        exc_struct = exc(self.particle, wavelength)

        # Solve BEM equations
        sig = self.bem.solve(exc_struct)

        # Compute cross sections using excitation object's methods
        # Note: sca/ext return arrays with shape (n_pol,), we take first element
        sca_arr = exc.sca(sig)
        ext_arr = exc.ext(sig)

        # Handle both scalar and array returns
        sca = float(np.sum(sca_arr)) if hasattr(sca_arr, '__len__') else float(sca_arr)
        ext = float(np.sum(ext_arr)) if hasattr(ext_arr, '__len__') else float(ext_arr)
        abs_val = ext - sca

        return (wl_idx, exc_idx, sca, abs_val, ext)

    def compute_spectrum(self, show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute optical cross-section spectra.

        Uses parallel computation if num_cores > 1.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with:
                - 'wavelengths': Wavelength array (nm)
                - 'scattering': Scattering cross-section (n_wavelengths, n_excitations)
                - 'absorption': Absorption cross-section (n_wavelengths, n_excitations)
                - 'extinction': Extinction cross-section (n_wavelengths, n_excitations)
        """
        n_wl = len(self.wavelengths)
        n_exc = len(self.excitations)

        scattering = np.zeros((n_wl, n_exc))
        absorption = np.zeros((n_wl, n_exc))
        extinction = np.zeros((n_wl, n_exc))

        # Store solutions for field calculation
        self.solutions = {}

        if self.num_cores > 1:
            # Parallel computation
            return self._compute_spectrum_parallel(show_progress)
        else:
            # Serial computation
            return self._compute_spectrum_serial(show_progress)

    def _compute_spectrum_serial(self, show_progress: bool = True) -> Dict[str, np.ndarray]:
        """Compute spectrum using serial execution."""
        sim_type = self.config.get('simulation_type', 'stat')

        n_wl = len(self.wavelengths)
        n_exc = len(self.excitations)

        scattering = np.zeros((n_wl, n_exc))
        absorption = np.zeros((n_wl, n_exc))
        extinction = np.zeros((n_wl, n_exc))

        for i, exc in enumerate(self.excitations):
            if show_progress:
                print(f"Computing spectrum for excitation {i+1}/{n_exc}")

            if sim_type == 'stat':
                # SpectrumStat API: __init__(bem, exc, wavelengths, show_progress)
                spec = self.SpectrumStat(self.bem, exc, self.wavelengths, show_progress=show_progress)
                result = spec.compute()

                # SpectrumStat.compute() returns tuple (sca, ext)
                if isinstance(result, tuple):
                    sca, ext = result
                    scattering[:, i] = sca.flatten()
                    extinction[:, i] = ext.flatten()
                    absorption[:, i] = ext.flatten() - sca.flatten()
            else:
                # SpectrumRet API: __init__(excitation, particle, bem, options)
                spec = self.SpectrumRet(exc, self.particle, self.bem, self.options)
                result = spec.compute(self.wavelengths)

                # SpectrumRet.compute() returns dict {'sca': ..., 'ext': ..., 'abs': ...}
                if isinstance(result, dict):
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

    def _compute_spectrum_parallel(self, show_progress: bool = True) -> Dict[str, np.ndarray]:
        """Compute spectrum using parallel ThreadPoolExecutor."""
        n_wl = len(self.wavelengths)
        n_exc = len(self.excitations)

        scattering = np.zeros((n_wl, n_exc))
        absorption = np.zeros((n_wl, n_exc))
        extinction = np.zeros((n_wl, n_exc))

        # Create list of all (wavelength, excitation) pairs to compute
        tasks = [(wl_idx, exc_idx) for exc_idx in range(n_exc) for wl_idx in range(n_wl)]
        total_tasks = len(tasks)

        if show_progress:
            print(f"Computing spectrum with {self.num_cores} cores ({total_tasks} calculations)")

        # Use ThreadPoolExecutor for parallel computation
        completed = 0
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._solve_single_wavelength, wl_idx, exc_idx): (wl_idx, exc_idx)
                for wl_idx, exc_idx in tasks
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    wl_idx, exc_idx, sca, abs_val, ext = future.result()
                    scattering[wl_idx, exc_idx] = sca
                    absorption[wl_idx, exc_idx] = abs_val
                    extinction[wl_idx, exc_idx] = ext

                    completed += 1
                    if show_progress and completed % max(1, total_tasks // 20) == 0:
                        print(f"  Progress: {completed}/{total_tasks} ({100*completed//total_tasks}%)")

                except Exception as e:
                    wl_idx, exc_idx = futures[future]
                    print(f"Error at wavelength {self.wavelengths[wl_idx]:.1f} nm, excitation {exc_idx}: {e}")

        if show_progress:
            print(f"  Completed: {completed}/{total_tasks}")

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
        info = {
            'simulation_type': self.config.get('simulation_type', 'stat'),
            'excitation_type': self.config.get('excitation_type', 'planewave'),
            'n_wavelengths': len(self.wavelengths) if self.wavelengths is not None else 0,
            'n_excitations': len(self.excitations),
            'wavelength_range': [float(self.wavelengths[0]), float(self.wavelengths[-1])] if self.wavelengths is not None else None,
            'num_cores': self.num_cores,
            'max_comp_threads': self.config.get('max_comp_threads', 1),
            'use_iterative_solver': self.config.get('use_iterative_solver', False),
            'use_h2_compression': self.config.get('use_h2_compression', False),
        }

        # Add availability info
        info['iterative_available'] = self._iter_available if hasattr(self, '_iter_available') else False
        info['hmatrix_available'] = self._hmatrix_available if hasattr(self, '_hmatrix_available') else False

        return info
