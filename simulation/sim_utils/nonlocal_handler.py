"""
Nonlocal Handler for pyMNPBEM simulations.

Implements quantum corrections for plasmonic nanoparticles using
the artificial cover layer approach (Luo et al., PRL 111, 093901, 2013).

This module provides:
- Nonlocal dielectric function generation
- Cover layer creation for particles
- Material modification for quantum effects
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
import sys


class NonlocalHandler:
    """
    Handles nonlocal quantum corrections for plasmonic simulations.

    Uses the artificial cover layer approach to model nonlocal effects
    in metals with sub-nanometer features (gaps < 1nm).

    Reference: Luo et al., PRL 111, 093901 (2013)
    """

    # Default Drude parameters for common metals
    DRUDE_PARAMS = {
        'gold': {
            'omega_p': 9.02,      # Plasma frequency (eV)
            'gamma': 0.071,       # Damping rate (eV)
            'v_f': 1.39e6,        # Fermi velocity (m/s)
            'eps_inf': 9.84,      # Background permittivity
        },
        'silver': {
            'omega_p': 9.17,      # Plasma frequency (eV)
            'gamma': 0.021,       # Damping rate (eV)
            'v_f': 1.39e6,        # Fermi velocity (m/s)
            'eps_inf': 3.7,       # Background permittivity
        },
        'aluminum': {
            'omega_p': 14.98,     # Plasma frequency (eV)
            'gamma': 0.047,       # Damping rate (eV)
            'v_f': 2.03e6,        # Fermi velocity (m/s)
            'eps_inf': 1.0,       # Background permittivity
        },
        'copper': {
            'omega_p': 10.83,     # Plasma frequency (eV)
            'gamma': 0.073,       # Damping rate (eV)
            'v_f': 1.57e6,        # Fermi velocity (m/s)
            'eps_inf': 1.0,       # Background permittivity
        }
    }

    # Physical constants
    HBAR = 6.582119569e-16  # eV*s
    C = 2.998e8  # m/s

    def __init__(self, config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize nonlocal handler.

        Args:
            config: Configuration dictionary with nonlocal settings
            pymnpbem_path: Path to pyMNPBEM installation
        """
        self.config = config
        self.enabled = config.get('use_nonlocality', False)

        # Nonlocal parameters from config or defaults
        self.cover_thickness = config.get('nonlocal_cover_thickness', 0.05)  # nm
        self.model = config.get('nonlocal_model', 'hydrodynamic')  # 'hydrodynamic' or 'qcm'

        # Custom Drude parameters (optional override)
        self.custom_drude = config.get('nonlocal_drude_params', {})

        # Import pyMNPBEM coverlayer module
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        self._import_coverlayer()

    def _import_coverlayer(self):
        """Import pyMNPBEM coverlayer module."""
        try:
            from mnpbem.greenfun.coverlayer import (
                CoverLayer, GreenStatCover, GreenRetCover,
                shift, coverlayer
            )
            self.CoverLayer = CoverLayer
            self.GreenStatCover = GreenStatCover
            self.GreenRetCover = GreenRetCover
            self.shift_func = shift
            self.coverlayer_factory = coverlayer
            self._coverlayer_available = True
        except ImportError:
            # Fallback if coverlayer not in greenfun
            try:
                from mnpbem.greenfun import coverlayer as cl_module
                self.CoverLayer = cl_module.CoverLayer
                self.shift_func = cl_module.shift
                self.coverlayer_factory = cl_module.coverlayer
                self._coverlayer_available = True
            except ImportError:
                print("Warning: pyMNPBEM coverlayer module not available")
                self._coverlayer_available = False

    def is_needed(self) -> bool:
        """
        Check if nonlocal corrections should be applied.

        Returns:
            True if nonlocal corrections are needed
        """
        if not self.enabled:
            return False

        if not self._coverlayer_available:
            print("Warning: Nonlocal requested but coverlayer module not available")
            return False

        # Check gap size - warn if large
        gap = self.config.get('gap', float('inf'))
        if gap >= 2.0:
            print(f"Note: Gap = {gap} nm is relatively large. "
                  f"Nonlocal effects may be small.")

        return True

    def get_drude_params(self, material: str) -> Dict[str, float]:
        """
        Get Drude model parameters for a material.

        Args:
            material: Material name ('gold', 'silver', etc.)

        Returns:
            Dictionary with omega_p, gamma, v_f, eps_inf
        """
        mat_lower = material.lower()

        # Check custom parameters first
        if mat_lower in self.custom_drude:
            return self.custom_drude[mat_lower]

        # Map common names
        material_map = {
            'au': 'gold',
            'ag': 'silver',
            'al': 'aluminum',
            'cu': 'copper',
        }
        mat_key = material_map.get(mat_lower, mat_lower)

        if mat_key in self.DRUDE_PARAMS:
            return self.DRUDE_PARAMS[mat_key].copy()

        # Default to gold parameters
        print(f"Warning: No Drude parameters for '{material}', using gold defaults")
        return self.DRUDE_PARAMS['gold'].copy()

    def compute_nonlocal_length(self, material: str, wavelength: float) -> float:
        """
        Compute the nonlocal length scale (Thomas-Fermi screening length).

        Args:
            material: Material name
            wavelength: Wavelength in nm

        Returns:
            Nonlocal length in nm
        """
        params = self.get_drude_params(material)
        omega_p = params['omega_p']
        v_f = params['v_f']

        # Convert wavelength to angular frequency
        omega = 2 * np.pi * self.C / (wavelength * 1e-9)  # rad/s
        omega_eV = self.HBAR * omega  # eV

        # Thomas-Fermi screening length
        # delta = v_f / omega_p (in natural units)
        # For hydrodynamic model: delta ~ beta / omega_p where beta = sqrt(3/5) * v_f
        beta = np.sqrt(3/5) * v_f  # m/s

        # Nonlocal length scale
        delta_nm = beta / (omega_p / self.HBAR) * 1e9  # nm

        return delta_nm

    def create_artificial_epsilon(self, material: str, eps_medium: complex = 1.0
                                   ) -> Callable[[float], complex]:
        """
        Create artificial dielectric function for nonlocal cover layer.

        Based on Luo et al., PRL 111, 093901 (2013).

        Args:
            material: Material name ('gold', 'silver', etc.)
            eps_medium: Dielectric constant of surrounding medium

        Returns:
            Function eps(wavelength_nm) -> complex
        """
        params = self.get_drude_params(material)
        omega_p = params['omega_p']
        gamma = params['gamma']
        v_f = params['v_f']
        eps_inf = params['eps_inf']

        d = self.cover_thickness
        beta = np.sqrt(3/5) * v_f  # Hydrodynamic beta

        def eps_nonlocal(wavelength_nm: float) -> complex:
            """
            Compute artificial nonlocal dielectric function.

            Args:
                wavelength_nm: Wavelength in nm

            Returns:
                Complex dielectric function
            """
            # Convert wavelength to energy
            energy_eV = 1239.84 / wavelength_nm
            omega = energy_eV  # in eV

            # Local Drude dielectric function
            eps_drude = eps_inf - omega_p**2 / (omega * (omega + 1j * gamma))

            # Longitudinal wavevector
            # q_l = sqrt(omega_p^2/eps_inf - omega*(omega + i*gamma)) / beta
            numerator = omega_p**2 / eps_inf - omega * (omega + 1j * gamma)
            q_l = np.sqrt(numerator + 0j) / (beta * self.HBAR * 1e9)  # 1/nm

            # Artificial dielectric function for cover layer
            # eps_art = eps_drude * eps_medium / (eps_drude - eps_medium) * q_l * d
            if abs(eps_drude - eps_medium) < 1e-10:
                return eps_drude

            eps_artificial = (eps_drude * eps_medium /
                            (eps_drude - eps_medium) * q_l * d)

            return eps_artificial

        return eps_nonlocal

    def create_cover_layer(self, particle: Any, thickness: Optional[float] = None
                          ) -> Tuple[Any, Any]:
        """
        Create a cover layer structure for a particle.

        Args:
            particle: pyMNPBEM Particle object
            thickness: Layer thickness in nm (uses default if None)

        Returns:
            Tuple of (inner_particle, outer_particle)
        """
        if not self._coverlayer_available:
            raise RuntimeError("Coverlayer module not available")

        d = thickness if thickness is not None else self.cover_thickness

        # Create inner boundary (original)
        inner = particle

        # Create outer boundary (shifted outward)
        outer = self.shift_func(particle, d)

        return inner, outer

    def modify_particles_for_nonlocal(self, particles: List[Any],
                                       inout: List[List[int]],
                                       materials: List[str]
                                       ) -> Tuple[List[Any], List[List[int]], List[Any]]:
        """
        Modify particle list to include nonlocal cover layers.

        For each metal particle, creates an additional thin cover layer
        with the artificial nonlocal dielectric function.

        Args:
            particles: List of Particle objects
            inout: Original inout list [[inside, outside], ...]
            materials: List of material names

        Returns:
            Tuple of (new_particles, new_inout, new_epstab_entries)
        """
        if not self.is_needed():
            return particles, inout, []

        metals = ['gold', 'silver', 'aluminum', 'copper', 'au', 'ag', 'al', 'cu']

        new_particles = []
        new_inout = []
        extra_eps = []

        # Track material index offset due to added artificial materials
        mat_offset = 0

        for i, (particle, io) in enumerate(zip(particles, inout)):
            inside_idx = io[0]
            outside_idx = io[1]

            # Get material for this particle (inside material)
            # Material indices: 1=medium, 2=first material, etc.
            if inside_idx > 1 and inside_idx - 2 < len(materials):
                mat_name = materials[inside_idx - 2]
                mat_lower = mat_name.lower() if isinstance(mat_name, str) else ''
                is_metal = any(m in mat_lower for m in metals)
            else:
                is_metal = False

            if is_metal:
                # Create cover layer
                inner, outer = self.create_cover_layer(particle)

                # Create artificial epsilon for this metal
                eps_art = self.create_artificial_epsilon(mat_name)
                extra_eps.append(eps_art)

                # New material index for artificial layer
                art_mat_idx = len(materials) + 2 + mat_offset
                mat_offset += 1

                # Inner particle: metal inside, artificial layer outside
                new_particles.append(inner)
                new_inout.append([inside_idx, art_mat_idx])

                # Outer particle (cover layer): artificial inside, medium outside
                new_particles.append(outer)
                new_inout.append([art_mat_idx, outside_idx])
            else:
                # Non-metal: keep as is
                new_particles.append(particle)
                new_inout.append(io)

        return new_particles, new_inout, extra_eps

    def get_green_function_class(self, sim_type: str) -> type:
        """
        Get appropriate Green function class for nonlocal calculations.

        Args:
            sim_type: 'stat' or 'ret'

        Returns:
            Green function class (GreenStatCover or GreenRetCover)
        """
        if not self._coverlayer_available:
            raise RuntimeError("Coverlayer module not available")

        if sim_type == 'stat':
            return self.GreenStatCover
        else:
            return self.GreenRetCover

    def create_coverlayer_object(self, eps_layer: Callable,
                                  thickness: Optional[float] = None,
                                  eps_core: Optional[Callable] = None
                                  ) -> Any:
        """
        Create a CoverLayer object for use with Green functions.

        Args:
            eps_layer: Dielectric function of the layer
            thickness: Layer thickness in nm
            eps_core: Optional core dielectric function

        Returns:
            CoverLayer object
        """
        if not self._coverlayer_available:
            raise RuntimeError("Coverlayer module not available")

        d = thickness if thickness is not None else self.cover_thickness
        return self.coverlayer_factory(eps_layer, d, eps_core)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of nonlocal settings.

        Returns:
            Dictionary with nonlocal configuration summary
        """
        return {
            'enabled': self.enabled,
            'model': self.model,
            'cover_thickness_nm': self.cover_thickness,
            'coverlayer_available': self._coverlayer_available,
            'drude_params': {k: v for k, v in self.DRUDE_PARAMS.items()},
        }


def compute_quantum_tunneling_epsilon(gap_nm: float, wavelength_nm: float,
                                       material: str = 'gold') -> complex:
    """
    Compute effective dielectric function for quantum tunneling regime.

    For gaps < 0.5 nm, electron tunneling becomes significant.
    This function provides a simple model for the tunneling conductivity.

    Reference: Esteban et al., Nature Communications 3, 825 (2012)

    Args:
        gap_nm: Gap size in nm
        wavelength_nm: Wavelength in nm
        material: Metal material

    Returns:
        Effective dielectric function in the gap
    """
    # Get Drude parameters
    handler = NonlocalHandler({'use_nonlocality': True})
    params = handler.get_drude_params(material)

    omega_p = params['omega_p']
    gamma = params['gamma']

    # Convert to angular frequency
    energy_eV = 1239.84 / wavelength_nm
    omega = energy_eV

    # Tunneling conductivity (simplified Simmons model)
    # sigma_t ~ exp(-2 * kappa * d) where kappa ~ 1/nm for typical barriers
    kappa = 1.0  # 1/nm, typical for vacuum barrier
    sigma_t = np.exp(-2 * kappa * gap_nm)

    # Effective Drude model with tunneling
    # gamma_eff = gamma + gamma_t where gamma_t accounts for tunneling
    gamma_t = sigma_t * omega_p / 10  # Empirical scaling
    gamma_eff = gamma + gamma_t

    # Effective dielectric
    eps_eff = 1.0 - omega_p**2 / (omega * (omega + 1j * gamma_eff))

    return eps_eff
