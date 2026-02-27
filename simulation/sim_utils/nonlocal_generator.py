import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.utils.constants import EV2NM


DRUDE_PARAMETERS = {
    'gold':     {'omega_p': 3.3, 'gamma': 0.165, 'beta': 0.0036, 'eps_inf': 1.0},
    'au':       {'omega_p': 3.3, 'gamma': 0.165, 'beta': 0.0036, 'eps_inf': 1.0},
    'silver':   {'omega_p': 3.8, 'gamma': 0.048, 'beta': 0.0036, 'eps_inf': 1.0},
    'ag':       {'omega_p': 3.8, 'gamma': 0.048, 'beta': 0.0036, 'eps_inf': 1.0},
    'aluminum': {'omega_p': 4.8, 'gamma': 0.14, 'beta': 0.0036, 'eps_inf': 1.0},
    'al':       {'omega_p': 4.8, 'gamma': 0.14, 'beta': 0.0036, 'eps_inf': 1.0},
}


class _ArtificialNonlocalEps(object):
    # Luo et al., PRL 111, 093901 (2013)
    # eps_nonlocal(enei) = eps_drude(enei) * eps_medium(enei)
    #     / (eps_drude(enei) - eps_medium(enei)) * ql(EV2NM / enei) * d

    def __init__(self,
            eps_drude_func: Any,
            eps_medium_func: Any,
            omega_p: float,
            gamma: float,
            beta: float,
            eps_inf: float,
            cover_thickness: float) -> None:

        self.eps_drude_func = eps_drude_func
        self.eps_medium_func = eps_medium_func
        self.omega_p = omega_p
        self.gamma = gamma
        self.beta = beta
        self.eps_inf = eps_inf
        self.d = cover_thickness

    def _ql(self, w: np.ndarray) -> np.ndarray:
        # longitudinal plasmon wavenumber
        return (2 * np.pi
                * np.sqrt(self.omega_p ** 2 / self.eps_inf
                           - w * (w + 1j * self.gamma))
                / self.beta)

    def __call__(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        enei = np.asarray(enei, dtype = float)
        w = EV2NM / enei  # photon energy in eV

        eps_drude, _ = self.eps_drude_func(enei)
        eps_medium, _ = self.eps_medium_func(enei)

        ql_val = self._ql(w)

        eps = (eps_drude * eps_medium
               / (eps_drude - eps_medium)
               * ql_val * self.d)

        k = 2 * np.pi / enei * np.sqrt(eps)

        return eps, k

    def __repr__(self) -> str:
        return '_ArtificialNonlocalEps(d = {})'.format(self.d)


class NonlocalGenerator(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose

        self.enabled = config.get('use_nonlocality', False)
        self.cover_thickness = config.get('nonlocal_cover_thickness', 0.05)  # nm

        self.omega_p = config.get('nonlocal_omega_p', 3.3)  # eV
        self.gamma = config.get('nonlocal_gamma', 0.165)  # eV
        self.beta = config.get('nonlocal_beta', 0.0036)  # eV*nm
        self.eps_inf = config.get('nonlocal_eps_inf', 1.0)

        if verbose and self.enabled:
            print('[info] Nonlocal corrections enabled:')
            print('  Cover layer thickness: {} nm'.format(self.cover_thickness))
            print('  Drude parameters: wp={} eV, gamma={} eV'.format(
                self.omega_p, self.gamma))

    def is_needed(self) -> bool:
        if not self.enabled:
            return False

        gap = self.config.get('gap', float('inf'))
        if gap >= 1.0:
            if self.verbose:
                print('[info] Warning: Gap = {} nm is large. '
                      'Nonlocal effects may be negligible.'.format(gap))

        return True

    def generate_artificial_epsilon(self,
            material_name: str = 'gold',
            eps_medium_func: Optional[Any] = None) -> _ArtificialNonlocalEps:

        if not self.enabled:
            return None

        mat_lower = material_name.lower()
        if mat_lower in DRUDE_PARAMETERS:
            params = DRUDE_PARAMETERS[mat_lower]
            omega_p = params['omega_p']
            gamma = params['gamma']
            beta = params['beta']
            eps_inf = params['eps_inf']
        else:
            omega_p = self.omega_p
            gamma = self.gamma
            beta = self.beta
            eps_inf = self.eps_inf

        # Drude dielectric function for the metal
        # eps_drude(w) = eps_inf - omega_p^2 / (w * (w + i*gamma))
        eps_drude_func = EpsDrude(eps0 = eps_inf, wp = omega_p, gammad = gamma)

        if eps_medium_func is None:
            eps_medium_func = EpsConst(1.0)

        nonlocal_eps = _ArtificialNonlocalEps(
            eps_drude_func = eps_drude_func,
            eps_medium_func = eps_medium_func,
            omega_p = omega_p,
            gamma = gamma,
            beta = beta,
            eps_inf = eps_inf,
            cover_thickness = self.cover_thickness)

        if self.verbose:
            print('[info] Generated artificial nonlocal epsilon for {}'.format(
                material_name))
            print('  d = {} nm, wp = {} eV, gamma = {} eV'.format(
                self.cover_thickness, omega_p, gamma))

        return nonlocal_eps

    def generate_drude_epsilon(self,
            material_name: str = 'gold') -> EpsDrude:

        mat_lower = material_name.lower()
        if mat_lower in DRUDE_PARAMETERS:
            params = DRUDE_PARAMETERS[mat_lower]
            omega_p = params['omega_p']
            gamma = params['gamma']
            eps_inf = params['eps_inf']
        else:
            omega_p = self.omega_p
            gamma = self.gamma
            eps_inf = self.eps_inf

        return EpsDrude(eps0 = eps_inf, wp = omega_p, gammad = gamma)

    def modify_materials_for_nonlocal(self,
            materials: List[str]) -> Tuple[List[str], Dict[int, Tuple[int, int]]]:

        if not self.enabled:
            return materials, {}

        metals = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al', 'copper', 'cu']

        # find the outermost metal (check from last to first)
        outermost_metal_idx = None
        for i in range(len(materials) - 1, -1, -1):
            mat_name = materials[i].lower() if isinstance(materials[i], str) else ''
            is_metal = any(metal in mat_name for metal in metals)
            if is_metal:
                outermost_metal_idx = i
                break

        if outermost_metal_idx is None:
            if self.verbose:
                print('[info] No outermost metal found - nonlocal will not be applied')
            return materials, {}

        if self.verbose:
            print('[info] Outermost metal: {} at index {}'.format(
                materials[outermost_metal_idx], outermost_metal_idx))

        modified_materials = []
        nonlocal_mapping = {}

        for i, mat in enumerate(materials):
            if i == outermost_metal_idx:
                # outermost metal gets Drude + nonlocal
                modified_materials.append(mat)
                modified_materials.append('{}_nonlocal'.format(mat))

                inner_idx = len(modified_materials) - 2
                outer_idx = len(modified_materials) - 1
                nonlocal_mapping[i] = (inner_idx, outer_idx)

                if self.verbose:
                    print('  {} : inner (Drude) index {}, outer (nonlocal) index {} [OUTERMOST]'.format(
                        mat, inner_idx, outer_idx))
            else:
                modified_materials.append(mat)
                if self.verbose:
                    print('  {} : standard index {}'.format(mat, len(modified_materials) - 1))

        return modified_materials, nonlocal_mapping

    def get_bem_options(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        npol = self.config.get('npol', 20)
        refine = self.config.get('refine', 3)

        return {'npol': npol, 'refine': refine}

    def check_applicability(self) -> Tuple[bool, List[str]]:
        warnings = []

        if not self.enabled:
            return True, []

        gap = self.config.get('gap', None)
        if gap is not None:
            if gap > 1.0:
                warnings.append(
                    'Gap = {} nm is large. Nonlocal effects may be negligible.'.format(gap))
            elif gap < 0.1:
                warnings.append(
                    'Gap = {} nm is extremely small. '
                    'Atomic-scale effects may dominate.'.format(gap))

        structure = self.config.get('structure', '')
        if 'sphere' in structure:
            diameter = self.config.get('diameter', self.config.get('core_diameter', 50))
            if diameter > 20:
                warnings.append(
                    'Particle diameter = {} nm. '
                    'Nonlocal effects are mainly surface phenomena.'.format(diameter))

        mesh_density = self.config.get('mesh_density', 12)
        if 'cube' in structure and mesh_density < 20:
            warnings.append(
                'Mesh density = {} may be insufficient for gap < 1 nm. '
                'Consider mesh_density >= 25.'.format(mesh_density))

        return len(warnings) == 0, warnings


def estimate_cover_thickness(gap: float,
        material: str = 'gold') -> float:

    d = max(0.02, min(0.1, gap / 10))
    return d


def get_drude_parameters(material: str) -> Dict[str, float]:

    mat_lower = material.lower()
    return DRUDE_PARAMETERS.get(mat_lower, DRUDE_PARAMETERS['gold']).copy()
