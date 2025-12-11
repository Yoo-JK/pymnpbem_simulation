"""
Material Builder for pyMNPBEM-based simulations.

Handles material definitions including:
- Built-in materials (gold, silver, aluminum, copper)
- Drude model metals
- Custom dielectric functions from files
- Medium (surrounding environment)
- Substrate materials
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import os
import sys


class MaterialBuilder:
    """
    Builds material (dielectric function) objects from configuration.

    Supports:
    - Constant dielectric (vacuum, glass, water)
    - Tabulated materials (gold, silver from data files)
    - Drude model metals
    - Custom refractive index files
    """

    # Built-in materials and their corresponding pyMNPBEM data files
    # Note: pyMNPBEM has gold.dat, silver.dat, goldpalik.dat, silverpalik.dat, copperpalik.dat
    BUILTIN_MATERIALS = {
        'gold': 'gold.dat',
        'au': 'gold.dat',
        'silver': 'silver.dat',
        'ag': 'silver.dat',
        'goldpalik': 'goldpalik.dat',
        'silverpalik': 'silverpalik.dat',
        'copperpalik': 'copperpalik.dat',
        'copper': 'copperpalik.dat',  # Use Palik data for copper
        'cu': 'copperpalik.dat',
    }

    # Common dielectric constants
    MEDIUM_CONSTANTS = {
        'vacuum': 1.0,
        'air': 1.0,
        'water': 1.77,  # eps = n^2, n~1.33
        'glass': 2.25,  # n~1.5
        'sio2': 2.13,   # n~1.46
        'ito': 3.8,     # approximate
        'tio2': 6.25,   # n~2.5
    }

    def __init__(self, structure_config: Dict[str, Any], pymnpbem_path: Optional[str] = None):
        """
        Initialize the material builder.

        Args:
            structure_config: Dictionary containing structure and material parameters
            pymnpbem_path: Path to pyMNPBEM installation (optional)
        """
        self.config = structure_config
        self.pymnpbem_path = pymnpbem_path

        # Import pyMNPBEM material modules
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        try:
            from mnpbem.material import EpsConst, EpsDrude, EpsTable, EpsFun
            self.EpsConst = EpsConst
            self.EpsDrude = EpsDrude
            self.EpsTable = EpsTable
            self.EpsFun = EpsFun
            self._pymnpbem_available = True
        except ImportError as e:
            print(f"Warning: Could not import pyMNPBEM material modules: {e}")
            self._pymnpbem_available = False

        # Store custom refractive index paths
        self.custom_paths = self.config.get('refractive_index_paths', {})

    def build(self) -> List[Any]:
        """
        Build the list of dielectric functions (epstab).

        Returns:
            List of dielectric function objects:
                epstab[0] = medium (always first)
                epstab[1:] = particle materials
        """
        epstab = []

        # 1. Build medium (always index 0 in epstab, index 1 in MNPBEM convention)
        medium_eps = self._build_medium()
        epstab.append(medium_eps)

        # 2. Build particle materials
        materials = self.config.get('materials', ['gold'])
        for mat in materials:
            mat_eps = self._build_material(mat)
            epstab.append(mat_eps)

        return epstab

    def _build_medium(self) -> Any:
        """Build the medium (surrounding environment) dielectric function."""
        medium = self.config.get('medium', 'air')

        if isinstance(medium, dict):
            # Custom medium specification
            if medium.get('type') == 'constant':
                eps_val = medium.get('epsilon', 1.0)
                return self.EpsConst(eps_val)
            elif medium.get('type') == 'drude':
                return self._build_drude_material(medium)
            elif medium.get('type') == 'file':
                return self._build_from_file(medium.get('path'))
            else:
                return self.EpsConst(1.0)
        elif isinstance(medium, str):
            medium_lower = medium.lower()
            if medium_lower in self.MEDIUM_CONSTANTS:
                return self.EpsConst(self.MEDIUM_CONSTANTS[medium_lower])
            else:
                # Try as a material name
                return self._build_material(medium)
        elif isinstance(medium, (int, float)):
            return self.EpsConst(float(medium))
        else:
            return self.EpsConst(1.0)

    def _build_material(self, material: Union[str, Dict]) -> Any:
        """Build a single material's dielectric function."""
        if isinstance(material, dict):
            return self._build_material_from_dict(material)
        elif isinstance(material, str):
            return self._build_material_from_name(material)
        else:
            raise ValueError(f"Invalid material specification: {material}")

    def _build_material_from_name(self, name: str) -> Any:
        """Build material from a name string."""
        name_lower = name.lower()

        # Check if custom path is provided
        if name_lower in self.custom_paths:
            return self._build_from_file(self.custom_paths[name_lower])

        # Check built-in materials
        if name_lower in self.BUILTIN_MATERIALS:
            data_file = self.BUILTIN_MATERIALS[name_lower]
            return self.EpsTable(data_file)

        # Check if it's a constant dielectric
        if name_lower in self.MEDIUM_CONSTANTS:
            return self.EpsConst(self.MEDIUM_CONSTANTS[name_lower])

        # Try Drude model for known metals
        if name_lower in ['au', 'gold', 'ag', 'silver', 'al', 'aluminum']:
            return self._build_drude_for_metal(name_lower)

        # Unknown material - assume it's a file path
        if os.path.exists(name):
            return self._build_from_file(name)

        raise ValueError(f"Unknown material: {name}")

    def _build_material_from_dict(self, mat_dict: Dict) -> Any:
        """Build material from a dictionary specification."""
        mat_type = mat_dict.get('type', 'constant')

        if mat_type == 'constant':
            eps_val = mat_dict.get('epsilon', 1.0)
            return self.EpsConst(complex(eps_val))

        elif mat_type == 'drude':
            return self._build_drude_material(mat_dict)

        elif mat_type == 'table':
            path = mat_dict.get('path')
            if path:
                return self._build_from_file(path)
            filename = mat_dict.get('filename')
            if filename:
                return self.EpsTable(filename)
            raise ValueError("Table material requires 'path' or 'filename'")

        elif mat_type == 'function':
            # Custom function - should be a callable
            func = mat_dict.get('function')
            if callable(func):
                return self.EpsFun(func)
            raise ValueError("Function material requires callable 'function'")

        else:
            raise ValueError(f"Unknown material type: {mat_type}")

    def _build_drude_material(self, mat_dict: Dict) -> Any:
        """Build a Drude model material from parameters."""
        # Check if using metal name
        if 'metal' in mat_dict:
            return self.EpsDrude(mat_dict['metal'])

        # Custom Drude parameters
        eps0 = mat_dict.get('eps0', mat_dict.get('eps_inf', 1.0))
        wp = mat_dict.get('wp', mat_dict.get('plasma_frequency', 9.0))
        gammad = mat_dict.get('gammad', mat_dict.get('damping', 0.1))

        return self.EpsDrude(eps0=eps0, wp=wp, gammad=gammad)

    def _build_drude_for_metal(self, metal: str) -> Any:
        """Build Drude model for known metals."""
        metal_params = {
            'au': {'name': 'Au'},
            'gold': {'name': 'Au'},
            'ag': {'name': 'Ag'},
            'silver': {'name': 'Ag'},
            'al': {'name': 'Al'},
            'aluminum': {'name': 'Al'},
        }

        if metal.lower() in metal_params:
            return self.EpsDrude(metal_params[metal.lower()]['name'])
        else:
            raise ValueError(f"Unknown metal for Drude model: {metal}")

    def _build_from_file(self, filepath: str) -> Any:
        """
        Build material from a refractive index file.

        File formats supported:
        - 3-column: wavelength(nm), n, k
        - 2-column: wavelength(nm), n (k assumed 0)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Refractive index file not found: {filepath}")

        # Read file and convert to epsilon
        wavelengths, eps_complex = self._load_refractive_index(filepath)

        # Create interpolating function
        from scipy.interpolate import interp1d

        eps_real_interp = interp1d(wavelengths, eps_complex.real, kind='cubic',
                                    bounds_error=False, fill_value='extrapolate')
        eps_imag_interp = interp1d(wavelengths, eps_complex.imag, kind='cubic',
                                    bounds_error=False, fill_value='extrapolate')

        def eps_func(wl):
            return eps_real_interp(wl) + 1j * eps_imag_interp(wl)

        return self.EpsFun(eps_func)

    def _load_refractive_index(self, filepath: str) -> tuple:
        """Load refractive index from file and convert to complex epsilon."""
        data = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        wl = float(parts[0])
                        n = float(parts[1])
                        k = float(parts[2]) if len(parts) >= 3 else 0.0
                        data.append([wl, n, k])
                except ValueError:
                    continue

        if not data:
            raise ValueError(f"No valid data found in file: {filepath}")

        data = np.array(data)
        wavelengths = data[:, 0]
        n = data[:, 1]
        k = data[:, 2]

        # Convert to complex epsilon: eps = (n + ik)^2 = n^2 - k^2 + 2ink
        eps_complex = (n + 1j * k) ** 2

        return wavelengths, eps_complex

    def get_material_info(self) -> Dict[str, Any]:
        """Get summary information about materials."""
        return {
            'medium': self.config.get('medium', 'air'),
            'materials': self.config.get('materials', []),
            'custom_paths': self.custom_paths,
        }

    def build_substrate(self) -> Optional[Dict[str, Any]]:
        """
        Build substrate configuration if specified.

        Returns:
            Dictionary with substrate info or None if no substrate
        """
        if not self.config.get('use_substrate', False):
            return None

        substrate_config = self.config.get('substrate', {})
        material = substrate_config.get('material', 'glass')
        position = substrate_config.get('position', 0)

        # Build substrate dielectric function
        substrate_eps = self._build_material(material)

        return {
            'eps': substrate_eps,
            'position': position,
            'material_name': material
        }
