import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
from mnpbem.materials import EpsConst, EpsTable, EpsDrude

from .refractive_index_loader import RefractiveIndexLoader
from .nonlocal_generator import NonlocalGenerator


# built-in material name -> Python MNPBEM epsilon object factory
_BUILTIN_CONSTANT_MATERIALS = {
    'air':      1.0,
    'vacuum':   1.0,
    'water':    1.33 ** 2,
    'glass':    2.25,
    'silicon':  11.7,
    'sapphire': 3.13,
    'sio2':     2.25,
    'agcl':     2.02,
}

_BUILTIN_TABLE_MATERIALS = {
    'gold':     'gold.dat',
    'au':       'gold.dat',
    'silver':   'silver.dat',
    'ag':       'silver.dat',
    'aluminum': 'aluminum.dat',
    'al':       'aluminum.dat',
    'copper':   'copperpalik.dat',
    'cu':       'copperpalik.dat',
}


class MaterialManager(object):

    METALS = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al', 'copper', 'cu']

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
        self.nonlocal_gen = NonlocalGenerator(config, verbose)
        self.complete_materials = self._build_complete_material_list()
        self.table_materials_data = {}

    def _is_metal(self, material: Union[str, Dict]) -> bool:
        if isinstance(material, str):
            mat_lower = material.lower()
            return any(metal in mat_lower for metal in self.METALS)
        return False

    def _is_conductive_junction(self) -> bool:
        gap = self.config.get('gap', None)
        if gap is None:
            return False
        return gap <= 0.0

    def _find_outermost_metal(self,
            materials: List[str]) -> Tuple[Optional[int], Optional[str]]:

        for i in range(len(materials) - 1, -1, -1):
            if self._is_metal(materials[i]):
                return i, materials[i]
        return None, None

    def _build_complete_material_list(self) -> List[Union[str, Dict]]:
        materials_list = []

        # 1. medium (always first)
        medium = self.config.get('medium', 'air')
        materials_list.append(medium)

        # 2. particle materials
        particle_materials = self.config.get('materials', [])
        if not isinstance(particle_materials, list):
            particle_materials = [particle_materials]
        materials_list.extend(particle_materials)

        # 3. substrate (if used, always last)
        if self.config.get('use_substrate', False):
            substrate = self.config.get('substrate', {})
            substrate_material = substrate.get('material', 'glass')
            materials_list.append(substrate_material)

        if self.verbose:
            print('[info] Complete material list: {}'.format(materials_list))

        return materials_list

    # =========================================================================
    # public API
    # =========================================================================

    def generate(self) -> Dict[str, Any]:
        # returns:
        # {
        #     'epstab': [eps_obj_1, eps_obj_2, ...],
        #     'inout':  np.ndarray of shape (n_boundaries, 2),
        #     'closed': list or None,
        # }

        use_substrate = self.config.get('use_substrate', False)
        use_nonlocal = self.nonlocal_gen.is_needed()

        if use_nonlocal:
            epstab = self._generate_nonlocal_epstab()
            inout = self._generate_inout_nonlocal()
        else:
            epstab = self._generate_epstab()
            inout = self._generate_inout()

        closed = self._generate_closed()

        result = {
            'epstab': epstab,
            'inout': inout,
            'closed': closed,
        }

        if use_substrate:
            substrate = self.config.get('substrate', {})
            result['substrate'] = {
                'z_interface': substrate.get('position', 0),
                'medium_idx': 0,           # index into epstab for medium
                'substrate_idx': len(epstab) - 1,  # index for substrate material
            }

        if self.verbose:
            print('[info] Generated materials:')
            print('  epstab length: {}'.format(len(epstab)))
            print('  inout shape: {}'.format(inout.shape))
            print('  closed: {}'.format(closed))

        return result

    # =========================================================================
    # epsilon table generation
    # =========================================================================

    def _generate_epstab(self) -> List[Any]:
        eps_list = []
        for i, material in enumerate(self.complete_materials):
            eps_obj = self._create_material(material, material_index = i)
            eps_list.append(eps_obj)
        return eps_list

    def _generate_nonlocal_epstab(self) -> List[Any]:
        if not self.nonlocal_gen.is_needed():
            return self._generate_epstab()

        materials_list = self.complete_materials
        particle_materials = self.config.get('materials', [])
        outermost_idx, outermost_metal = self._find_outermost_metal(particle_materials)

        if self.verbose:
            print('[info] === Generating Nonlocal Materials ===')
            if outermost_metal:
                print('  Outermost metal: {} (index {} in particle materials)'.format(
                    outermost_metal, outermost_idx))
            else:
                print('  No metal found in outermost layer - nonlocal will NOT be applied')

        if outermost_metal is None:
            if self.verbose:
                print('  Falling back to standard (non-nonlocal) material generation')
            return self._generate_epstab()

        # outermost_idx is relative to particle_materials
        # in complete_materials it's outermost_idx + 1 (medium is at index 0)
        outermost_complete_idx = outermost_idx + 1

        epstab = []

        # medium is always first
        medium_eps = self._create_material(materials_list[0], material_index = 0)
        epstab.append(medium_eps)

        for i, mat in enumerate(materials_list):
            if i == 0:
                # medium already added
                continue

            if i == outermost_complete_idx:
                # outermost metal: Drude + Nonlocal (2 entries)
                drude_eps = self.nonlocal_gen.generate_drude_epsilon(
                    material_name = mat if isinstance(mat, str) else 'gold')
                nonlocal_eps = self.nonlocal_gen.generate_artificial_epsilon(
                    material_name = mat if isinstance(mat, str) else 'gold',
                    eps_medium_func = medium_eps)
                epstab.append(drude_eps)
                epstab.append(nonlocal_eps)

                if self.verbose:
                    print('  Material {}: {} -> Drude + Nonlocal (outermost)'.format(i + 1, mat))
            else:
                # standard material
                eps_obj = self._create_material(mat, material_index = i)
                epstab.append(eps_obj)

                if self.verbose:
                    print('  Material {}: {} -> Standard'.format(i + 1, mat))

        return epstab

    # =========================================================================
    # material creation
    # =========================================================================

    def _create_material(self,
            material: Union[str, Dict],
            material_index: int = 0) -> Any:

        if isinstance(material, dict):
            return self._create_material_from_dict(material, material_index)
        elif isinstance(material, str):
            return self._create_material_from_name(material, material_index)
        else:
            raise ValueError(
                '[error] Invalid material specification: {}'.format(material))

    def _create_material_from_name(self,
            material: str,
            material_index: int = 0) -> Any:

        material_lower = material.lower()

        # check for custom refractive index paths
        refractive_index_paths = self.config.get('refractive_index_paths', {})

        if material_lower in refractive_index_paths:
            custom_value = refractive_index_paths[material_lower]

            if isinstance(custom_value, dict):
                if self.verbose:
                    print('[info] Using custom material definition for '
                          "'{}' from refractive_index_paths".format(material))
                return self._create_material(custom_value, material_index)

            elif isinstance(custom_value, str):
                custom_path = str(Path(custom_value).expanduser())
                if self.verbose:
                    print('[info] Using custom refractive index path for '
                          "'{}': {}".format(material, custom_path))
                return EpsTable(custom_path)

            else:
                raise ValueError(
                    "[error] Invalid value in refractive_index_paths for '{}': "
                    "expected string (file path) or dict (material definition), "
                    "got {}".format(material, type(custom_value)))

        if material_lower in _BUILTIN_CONSTANT_MATERIALS:
            eps_value = _BUILTIN_CONSTANT_MATERIALS[material_lower]
            return EpsConst(eps_value)

        if material_lower in _BUILTIN_TABLE_MATERIALS:
            filename = _BUILTIN_TABLE_MATERIALS[material_lower]
            return EpsTable(filename)

        raise ValueError('[error] Unknown material: <{}>'.format(material))

    def _create_material_from_dict(self,
            material: Dict[str, Any],
            material_index: int = 0) -> Any:

        mat_type = material.get('type', 'constant')

        match mat_type:

            case 'constant':
                epsilon = material['epsilon']
                return EpsConst(epsilon)

            case 'table':
                return self._handle_table_material(material, material_index)

            case 'drude':
                eps0 = material.get('eps0', 1.0)
                wp = material['wp']
                gammad = material['gammad']
                return EpsDrude(eps0 = eps0, wp = wp, gammad = gammad)

            case _:
                raise ValueError(
                    '[error] Unknown custom material type: <{}>'.format(mat_type))

    def _handle_table_material(self,
            material: Dict[str, Any],
            material_index: int) -> Any:

        filepath = material['file']
        filepath = Path(filepath).expanduser()

        if self.verbose:
            print('[info] Processing table material (index {})'.format(material_index))
            print('  File: {}'.format(filepath))

        try:
            loader = RefractiveIndexLoader(str(filepath), verbose = self.verbose)

            wavelength_range = self.config.get('wavelength_range', [400, 800, 80])
            target_wavelengths = np.linspace(
                wavelength_range[0],
                wavelength_range[1],
                wavelength_range[2])

            n_interp, k_interp = loader.interpolate(target_wavelengths)

            refractive_index = n_interp + 1j * k_interp
            epsilon_complex = refractive_index ** 2

            self.table_materials_data[material_index] = {
                'wavelengths': target_wavelengths,
                'n': n_interp,
                'k': k_interp,
                'epsilon': epsilon_complex,
            }

            # use EpsTable with the file path directly
            return EpsTable(str(filepath))

        except Exception as e:
            raise RuntimeError(
                "[error] Error processing table material '{}': {}".format(filepath, e))

    # =========================================================================
    # inout generation
    # =========================================================================

    def _generate_inout(self) -> np.ndarray:

        structure_inout_map = {
            'sphere':                       self._inout_single,
            'cube':                         self._inout_single,
            'rod':                          self._inout_single,
            'ellipsoid':                    self._inout_single,
            'triangle':                     self._inout_single,
            'dimer_sphere':                 self._inout_dimer,
            'dimer_cube':                   self._inout_dimer,
            'sphere_cluster_aggregate':     self._inout_sphere_cluster_aggregate,
            'core_shell_sphere':            self._inout_core_shell_single,
            'core_shell_cube':              self._inout_core_shell_single,
            'core_shell_rod':               self._inout_core_shell_single,
            'dimer_core_shell_cube':        self._inout_dimer_core_shell,
            'advanced_dimer_cube':          self._inout_advanced_dimer_cube,
            'connected_dimer_cube':         self._inout_connected_dimer_cube,
            'advanced_monomer_cube':        self._inout_advanced_monomer_cube,
            'from_shape':                   self._inout_from_shape,
        }

        if self.structure not in structure_inout_map:
            raise ValueError('[error] Unknown structure: <{}>'.format(self.structure))

        return structure_inout_map[self.structure]()

    def _inout_single(self) -> np.ndarray:
        # [inside=particle(2), outside=medium(1)]
        # 1-based indexing for MNPBEM compatibility
        return np.array([[2, 1]], dtype = int)

    def _inout_dimer(self) -> np.ndarray:
        return np.array([
            [2, 1],  # particle 1
            [2, 1],  # particle 2
        ], dtype = int)

    def _inout_core_shell_single(self) -> np.ndarray:
        return np.array([
            [2, 3],  # core: inside=core(2), outside=shell(3)
            [3, 1],  # shell: inside=shell(3), outside=medium(1)
        ], dtype = int)

    def _inout_dimer_core_shell(self) -> np.ndarray:
        return np.array([
            [2, 3],  # P1-core
            [3, 1],  # P1-shell
            [2, 3],  # P2-core
            [3, 1],  # P2-shell
        ], dtype = int)

    def _inout_sphere_cluster_aggregate(self) -> np.ndarray:
        n_spheres = self.config.get('n_spheres', 1)
        n_rows = n_spheres
        inout = np.empty((n_rows, 2), dtype = int)
        for i in range(n_spheres):
            inout[i, 0] = 2
            inout[i, 1] = 1
        return inout

    def _inout_advanced_dimer_cube(self) -> np.ndarray:
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        n_shells = len(shell_layers)
        n_layers = len(materials)  # 1 (core) + n_shells
        n_particles = 2

        if n_shells == 0:
            # no shells, just core
            inout = np.empty((n_particles, 2), dtype = int)
            for p in range(n_particles):
                inout[p, 0] = 2
                inout[p, 1] = 1
            return inout

        rows = []
        for p in range(n_particles):
            # core
            rows.append([2, 3])
            # shells
            for i in range(n_shells):
                mat_idx = 2 + (i + 1)  # shell material index (1-based in epstab)
                if i == n_shells - 1:
                    rows.append([mat_idx, 1])  # outermost shell -> medium
                else:
                    next_mat_idx = mat_idx + 1
                    rows.append([mat_idx, next_mat_idx])

        return np.array(rows, dtype = int)

    def _inout_connected_dimer_cube(self) -> np.ndarray:
        shell_layers = self.config.get('shell_layers', [])
        is_core_shell = len(shell_layers) > 0

        if not is_core_shell:
            return np.array([[2, 1]], dtype = int)

        gap = self.config.get('gap', 0)
        shell_thickness = shell_layers[0]
        core_gap = gap + 2 * shell_thickness
        fuse_cores = core_gap <= 0

        if fuse_cores:
            return np.array([
                [2, 3],  # fused core
                [3, 1],  # fused shell
            ], dtype = int)
        else:
            return np.array([
                [2, 3],  # core 1
                [2, 3],  # core 2
                [3, 1],  # fused shell
            ], dtype = int)

    def _inout_advanced_monomer_cube(self) -> np.ndarray:
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        n_shells = len(shell_layers)
        n_layers = len(materials)

        if n_shells == 0:
            return np.array([[2, 1]], dtype = int)

        rows = []
        # core
        rows.append([2, 3])
        # shells
        for i in range(n_shells):
            mat_idx = 2 + (i + 1)
            if i == n_shells - 1:
                rows.append([mat_idx, 1])
            else:
                next_mat_idx = mat_idx + 1
                rows.append([mat_idx, next_mat_idx])

        return np.array(rows, dtype = int)

    def _inout_from_shape(self) -> np.ndarray:
        n_materials = len(self.config.get('materials', []))

        n_rows = max(n_materials, 1)
        inout = np.empty((n_rows, 2), dtype = int)
        for i in range(n_rows):
            inout[i, 0] = i + 2  # material index (1-based, starting from 2)
            inout[i, 1] = 1      # outside = medium
        return inout

    # =========================================================================
    # inout generation (nonlocal mode)
    # =========================================================================

    def _generate_inout_nonlocal(self) -> np.ndarray:
        structure = self.config.get('structure', '')
        materials = self.config.get('materials', [])

        outermost_idx, outermost_metal = self._find_outermost_metal(materials)

        if self.verbose:
            print('[info] === Generating Inout (Nonlocal Mode) ===')
            if outermost_metal:
                print('  Outermost metal: {} at index {}'.format(
                    outermost_metal, outermost_idx))
            else:
                print('  No outermost metal - using standard inout')

        if outermost_metal is None:
            return self._generate_inout()

        if 'dimer' in structure:
            n_particles = 2
        else:
            n_particles = 1

        # build epstab index mapping
        # epstab = [medium, mat1, mat2, ..., outermost_drude, outermost_nonlocal, ...]
        # 1-based indexing: medium=1, then sequential
        mat_indices = {}
        epstab_idx = 2  # start from 2 (1 is medium)

        for i, mat in enumerate(materials):
            if i == outermost_idx:
                mat_indices[i] = {
                    'drude': epstab_idx,
                    'nonlocal': epstab_idx + 1,
                    'is_outermost_metal': True,
                }
                epstab_idx += 2
            else:
                mat_indices[i] = {
                    'standard': epstab_idx,
                    'is_outermost_metal': False,
                }
                epstab_idx += 1

        # generate inout rows
        n_layers = len(materials)
        rows = []

        for particle_idx in range(n_particles):
            for layer_idx in range(n_layers):
                is_outermost_layer = (layer_idx == n_layers - 1)
                is_outermost_metal = mat_indices[layer_idx].get('is_outermost_metal', False)

                if is_outermost_metal:
                    nonlocal_idx = mat_indices[layer_idx]['nonlocal']
                    drude_idx = mat_indices[layer_idx]['drude']
                    # outer boundary: nonlocal inside, medium(1) outside
                    rows.append([nonlocal_idx, 1])
                    # inner boundary: drude inside, nonlocal outside
                    rows.append([drude_idx, nonlocal_idx])
                else:
                    std_idx = mat_indices[layer_idx]['standard']

                    if is_outermost_layer:
                        rows.append([std_idx, 1])
                    elif layer_idx == 0:
                        next_layer_idx = layer_idx + 1
                        if mat_indices[next_layer_idx].get('is_outermost_metal'):
                            outside_idx = mat_indices[next_layer_idx]['drude']
                        else:
                            outside_idx = mat_indices[next_layer_idx]['standard']
                        rows.append([std_idx, outside_idx])
                    else:
                        next_layer_idx = layer_idx + 1
                        if mat_indices[next_layer_idx].get('is_outermost_metal'):
                            outside_idx = mat_indices[next_layer_idx]['drude']
                        else:
                            outside_idx = mat_indices[next_layer_idx]['standard']
                        rows.append([std_idx, outside_idx])

        return np.array(rows, dtype = int)

    # =========================================================================
    # closed surface generation
    # =========================================================================

    def _generate_closed(self) -> List[int]:
        use_nonlocal = self.nonlocal_gen.is_needed()
        materials = self.config.get('materials', [])
        use_conductive = self._is_conductive_junction()

        outermost_idx, outermost_metal = self._find_outermost_metal(materials)

        if use_nonlocal and outermost_metal is None:
            use_nonlocal = False

        # conductive junction for simple dimer structures with gap <= 0
        if use_conductive and not use_nonlocal:
            if self.structure in ['dimer_sphere', 'dimer_cube']:
                if self.verbose:
                    print('[info] Auto-detected gap <= 0: '
                          'Using conductive junction (CTP mode)')
                return [1, 2]
            elif self.structure == 'dimer_core_shell_cube':
                if self.verbose:
                    print('[info] Auto-detected gap <= 0: '
                          'Using conductive junction (CTP mode)')
                return [1, 2, 3, 4]

        structure_closed_map = {
            'sphere':                       [1],
            'cube':                         [1],
            'rod':                          [1],
            'ellipsoid':                    [1],
            'triangle':                     [1],
            'dimer_sphere':                 [1, 2],
            'dimer_cube':                   [1, 2],
            'core_shell_sphere':            [1, 2],
            'core_shell_cube':              [1, 2],
            'core_shell_rod':               [1, 2],
            'dimer_core_shell_cube':        [1, 2, 3, 4],
        }

        # structures with callable handlers
        callable_structures = {
            'sphere_cluster_aggregate':     self._closed_sphere_cluster_aggregate,
            'advanced_dimer_cube':          self._closed_advanced_dimer_cube,
            'connected_dimer_cube':         self._closed_connected_dimer_cube,
            'advanced_monomer_cube':        self._closed_advanced_monomer_cube,
            'from_shape':                   self._closed_from_shape,
        }

        if self.structure in structure_closed_map:
            result = structure_closed_map[self.structure]
            if use_nonlocal:
                return self._calculate_closed_for_nonlocal(outermost_idx)
            return result

        if self.structure in callable_structures:
            handler = callable_structures[self.structure]
            if self.structure in ['advanced_dimer_cube', 'advanced_monomer_cube']:
                return handler(use_nonlocal = use_nonlocal, outermost_idx = outermost_idx)
            else:
                return handler()

        raise ValueError('[error] Unknown structure: <{}>'.format(self.structure))

    def _calculate_closed_for_nonlocal(self,
            outermost_idx: Optional[int]) -> List[int]:

        materials = self.config.get('materials', [])
        n_layers = len(materials)
        structure = self.structure

        if 'dimer' in structure:
            n_particles = 2
        else:
            n_particles = 1

        # outermost metal gets +1 extra boundary
        boundaries_per_particle = n_layers + 1
        total_boundaries = n_particles * boundaries_per_particle
        return list(range(1, total_boundaries + 1))

    def _closed_sphere_cluster_aggregate(self) -> List[int]:
        n_spheres = self.config.get('n_spheres', 1)
        return list(range(1, n_spheres + 1))

    def _closed_advanced_dimer_cube(self,
            use_nonlocal: bool = False,
            outermost_idx: Optional[int] = None) -> List[int]:

        materials = self.config.get('materials', [])
        n_layers = len(materials)
        n_particles = 2

        if use_nonlocal and outermost_idx is not None:
            boundaries_per_particle = n_layers + 1
            n_boundaries_total = n_particles * boundaries_per_particle

            if self.verbose:
                print('[info] Advanced dimer with nonlocal on outermost metal only:')
                print('  {} layers per particle + 1 extra for nonlocal = {}'.format(
                    n_layers, boundaries_per_particle))
                print('  Total boundaries: {}'.format(n_boundaries_total))
        else:
            boundaries_per_particle = n_layers
            n_boundaries_total = n_particles * boundaries_per_particle

        return list(range(1, n_boundaries_total + 1))

    def _closed_connected_dimer_cube(self) -> List[int]:
        shell_layers = self.config.get('shell_layers', [])
        is_core_shell = len(shell_layers) > 0

        if not is_core_shell:
            return [1]

        gap = self.config.get('gap', 0)
        shell_thickness = shell_layers[0]
        core_gap = gap + 2 * shell_thickness
        fuse_cores = core_gap <= 0

        if fuse_cores:
            return [1, 2]
        else:
            return [1, 2, 3]

    def _closed_advanced_monomer_cube(self,
            use_nonlocal: bool = False,
            outermost_idx: Optional[int] = None) -> List[int]:

        materials = self.config.get('materials', [])
        n_layers = len(materials)
        n_particles = 1

        if use_nonlocal and outermost_idx is not None:
            boundaries_per_particle = n_layers + 1
            n_boundaries_total = n_particles * boundaries_per_particle

            if self.verbose:
                print('[info] Advanced monomer with nonlocal on outermost metal only:')
                print('  {} layers + 1 extra for nonlocal = {}'.format(
                    n_layers, boundaries_per_particle))
                print('  Total boundaries: {}'.format(n_boundaries_total))
        else:
            boundaries_per_particle = n_layers
            n_boundaries_total = n_particles * boundaries_per_particle

        return list(range(1, n_boundaries_total + 1))

    def _closed_from_shape(self) -> List[int]:
        n_materials = len(self.config.get('materials', []))
        assert n_materials > 0, '[error] No materials specified for DDA shape file'
        return list(range(1, n_materials + 1))
