"""
Material Manager

Manages material definitions and generates corresponding MATLAB code.
Supports:
  - Medium, materials, and substrate as separate configs
  - Enhanced 'table' type with automatic interpolation
  - Custom refractive_index_paths from config
  - Nonlocal quantum corrections for metals
"""

import numpy as np
from pathlib import Path
from .refractive_index_loader import RefractiveIndexLoader
from .nonlocal_generator import NonlocalGenerator


class MaterialManager:
    """Manages material definitions and dielectric functions."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
        self.nonlocal_gen = NonlocalGenerator(config, verbose)
        self.complete_materials = self._build_complete_material_list()
        self.table_materials_data = {}
    
    def _build_complete_material_list(self):
        """
        Build complete ordered material list from medium, materials, substrate.
        
        Returns:
            list: Complete material list in correct order for MNPBEM
        """
        materials_list = []
        
        # 1. Medium (always first)
        medium = self.config.get('medium', 'air')
        materials_list.append(medium)
        
        # 2. Particle materials
        particle_materials = self.config.get('materials', [])
        if not isinstance(particle_materials, list):
            particle_materials = [particle_materials]
        materials_list.extend(particle_materials)
        
        # 3. Substrate (if used, always last)
        if self.config.get('use_substrate', False):
            substrate = self.config.get('substrate', {})
            substrate_material = substrate.get('material', 'glass')
            materials_list.append(substrate_material)
        
        if self.verbose:
            print(f"Complete material list: {materials_list}")
        
        return materials_list
    
    def generate(self):
        """Generate material-related MATLAB code."""
        use_substrate = self.config.get('use_substrate', False)
        use_nonlocal = self.nonlocal_gen.is_needed()
        
        if use_substrate:
            return self._generate_substrate_materials()
        
        if use_nonlocal:
            # Use nonlocal-aware generation
            epstab_code = self._generate_nonlocal_epstab()
            inout_code = self._generate_inout_nonlocal()
            closed_code = self._generate_closed()
            
            code = f"""
%% Materials and Dielectric Functions (Nonlocal Mode)
{epstab_code}

%% Material Mapping
{inout_code}

%% Closed Surfaces
{closed_code}
"""
        else:
            # Standard generation
            epstab_code = self._generate_epstab()
            inout_code = self._generate_inout()
            closed_code = self._generate_closed()
            
            code = f"""
%% Materials and Dielectric Functions
{epstab_code}

%% Material Mapping
{inout_code}

%% Closed Surfaces
{closed_code}
"""
        
        return code
    
    def _generate_nonlocal_epstab(self):
        """Generate dielectric function table with nonlocal corrections."""
        if not self.nonlocal_gen.is_needed():
            return self._generate_epstab()
        
        if self.verbose:
            print("\n=== Generating Nonlocal Materials ===")
        
        materials_list = self.complete_materials
        metals = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al']
        
        epstab_entries = []
        material_descriptions = []
        
        for i, mat in enumerate(materials_list, 1):
            mat_lower = mat.lower() if isinstance(mat, str) else 'custom'
            is_metal = any(metal in mat_lower for metal in metals)
            
            if is_metal and i > 1:  # Skip medium (index 1)
                material_descriptions.append(f"% Material {i}: {mat} (Drude + Nonlocal)")
                epstab_entries.append(f"eps_{mat}_drude")
                epstab_entries.append(f"eps_{mat}_nonlocal")
                
                if self.verbose:
                    print(f"  Material {i}: {mat} → Drude + Nonlocal")
            else:
                material_descriptions.append(f"% Material {i}: {mat}")
                # 🔥 FIX: 변수명만 추출 (세미콜론 제거)
                mat_code = self._generate_single_material(mat, i)
                # epsconst(1) 같은 함수 호출을 변수로 만들기
                var_name = f"eps_mat{i}"
                epstab_entries.append(var_name)
                
                if self.verbose:
                    print(f"  Material {i}: {mat} → Standard")
        
        materials_code = "\n".join(material_descriptions)
        epstab_code = "epstab = { " + ", ".join(epstab_entries) + " };"
        
        full_code = f"""
%% Dielectric Functions with Nonlocal Corrections
{materials_code}

% Medium
eps_mat1 = {self._generate_single_material(materials_list[0], 1)};  % 🔥 세미콜론 추가!

% Generate artificial nonlocal dielectric functions
"""
        
        # Add artificial epsilon generation for each metal
        for i, mat in enumerate(materials_list[1:], start=2):  # 🔥 인덱스 수정
            mat_lower = mat.lower() if isinstance(mat, str) else ''
            is_metal = any(metal in mat_lower for metal in metals)
            
            if is_metal:
                full_code += self.nonlocal_gen.generate_artificial_epsilon(mat)
                full_code += "\n"
        
        # Add standard material definitions for non-metals
        for i, mat in enumerate(materials_list, 1):
            mat_lower = mat.lower() if isinstance(mat, str) else ''
            is_metal = any(metal in mat_lower for metal in metals)
            
            if not is_metal and i > 1:  # Skip medium (already defined)
                mat_def = self._generate_single_material(mat, i)
                full_code += f"eps_mat{i} = {mat_def};  % {mat}\n"
        
        full_code += f"\n{epstab_code}\n"
        
        return full_code
    
    def _generate_inout_nonlocal(self):
        """Generate inout matrix for nonlocal structure (FIXED VERSION)."""
        structure = self.config.get('structure', '')
        materials = self.config.get('materials', [])
        metals = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al']
        
        if 'dimer' in structure:
            n_particles = 2
        else:
            n_particles = 1
        
        # Build material index mapping
        # epstab structure: [medium, material1_drude, material1_nonlocal, material2, ...]
        mat_indices = {}
        epstab_idx = 2  # Start from 2 (1 is medium)
        
        for i, mat in enumerate(materials):
            mat_lower = mat.lower() if isinstance(mat, str) else ''
            is_metal = any(metal in mat_lower for metal in metals)
            
            if is_metal:
                mat_indices[i] = {
                    'drude': epstab_idx,
                    'nonlocal': epstab_idx + 1
                }
                epstab_idx += 2
            else:
                mat_indices[i] = {
                    'standard': epstab_idx
                }
                epstab_idx += 1
        
        # Generate inout rows
        inout_rows = []
        
        for particle_idx in range(n_particles):
            for layer_idx, mat in enumerate(materials):
                mat_lower = mat.lower() if isinstance(mat, str) else ''
                is_metal = any(metal in mat_lower for metal in metals)
                
                if is_metal:
                    # Outer boundary: nonlocal inside, medium outside
                    nonlocal_idx = mat_indices[layer_idx]['nonlocal']
                    inout_rows.append(f"{nonlocal_idx}, 1")
                    
                    # Inner boundary: drude inside, nonlocal outside
                    drude_idx = mat_indices[layer_idx]['drude']
                    inout_rows.append(f"{drude_idx}, {nonlocal_idx}")
                else:
                    # Standard material
                    std_idx = mat_indices[layer_idx]['standard']
                    if layer_idx == 0:
                        inout_rows.append(f"{std_idx}, 1")
                    else:
                        prev_idx = mat_indices[layer_idx-1].get('nonlocal', mat_indices[layer_idx-1].get('standard'))
                        inout_rows.append(f"{std_idx}, {prev_idx}")
        
        inout_str = "; ...\n         ".join(inout_rows)
        
        code = f"""
%% Material Mapping (with Nonlocal)
% inout(i, :) = [material_inside, material_outside] for boundary i
inout = [ {inout_str} ];

fprintf('  Total boundaries: %d\\n', size(inout, 1));
"""
        return code
    
    def _generate_substrate_materials(self):
        """Generate materials code for substrate configuration."""
        substrate = self.config.get('substrate', {})
        z_interface = substrate.get('position', 0)
        
        epstab_code = self._generate_epstab()
        substrate_idx = len(self.complete_materials)
        
        code = f"""
%% Materials and Dielectric Functions (with Substrate)
{epstab_code}

%% Layer Structure Setup
fprintf('Setting up layer structure...\\n');
z_interface = {z_interface};

if exist('layerstructure', 'class')
    op_layer = layerstructure.options;
    layer = layerstructure(epstab, [1, {substrate_idx}], z_interface, op_layer);
    op.layer = layer;
    fprintf('  Layer structure created at z=%.2f nm\\n', z_interface);
else
    warning('layerstructure class not found. Running without substrate.');
end

%% Material Mapping
{self._generate_inout()}

%% Closed Surfaces
{self._generate_closed()}
"""
        return code
    
    def _generate_epstab(self):
        """Generate epstab (dielectric function table)."""
        eps_list = []
        
        for i, material in enumerate(self.complete_materials):
            eps_code = self._material_to_eps(material, material_index=i)
            eps_list.append(eps_code)
        
        eps_str = ', '.join(eps_list)
        code = f"epstab = {{ {eps_str} }};"
        return code
    
    def _generate_single_material(self, material, material_index):
        """Generate single material definition."""
        return self._material_to_eps(material, material_index)
    
    def _material_to_eps(self, material, material_index=0):
        """Convert material specification to MATLAB epsilon code."""
        material_map = {
            'air': "epsconst(1)",
            'vacuum': "epsconst(1)",
            'water': "epsconst(1.33^2)",
            'glass': "epsconst(2.25)",
            'silicon': "epsconst(11.7)",
            'sapphire': "epsconst(3.13)",
            'sio2': "epsconst(2.25)",
            'agcl': "epsconst(2.02)",
            'gold': "epstable('gold.dat')",
            'silver': "epstable('silver.dat')",
            'aluminum': "epstable('aluminum.dat')"
        }
        
        if isinstance(material, dict):
            mat_type = material.get('type', 'constant')
            
            if mat_type == 'constant':
                epsilon = material['epsilon']
                return f"epsconst({epsilon})"
            
            elif mat_type == 'table':
                return self._handle_table_material(material, material_index)
            
            elif mat_type == 'function':
                formula = material['formula']
                unit = material.get('unit', 'nm')
                
                if unit == 'eV':
                    return f"epsfun(@(w) {formula}, 'eV')"
                else:
                    return f"epsfun(@(enei) {formula})"
            
            else:
                raise ValueError(f"Unknown custom material type: {mat_type}")
        
        elif isinstance(material, str):
            material_lower = material.lower()
            
            # Check for custom refractive index paths
            refractive_index_paths = self.config.get('refractive_index_paths', {})
            
            if material_lower in refractive_index_paths:
                custom_value = refractive_index_paths[material_lower]
                
                if isinstance(custom_value, dict):
                    if self.verbose:
                        print(f"Using custom material definition for '{material}' from refractive_index_paths")
                    return self._material_to_eps(custom_value, material_index)
                
                elif isinstance(custom_value, str):
                    custom_path = str(Path(custom_value).expanduser())
                    if self.verbose:
                        print(f"Using custom refractive index path for '{material}': {custom_path}")
                    return f"epstable('{custom_path}')"
                
                else:
                    raise ValueError(
                        f"Invalid value in refractive_index_paths for '{material}': "
                        f"expected string (file path) or dict (material definition), "
                        f"got {type(custom_value)}"
                    )
            
            if material_lower in material_map:
                return material_map[material_lower]
            else:
                raise ValueError(f"Unknown material: {material}")
        
        else:
            raise ValueError(f"Invalid material specification: {material}")
    
    def _handle_table_material(self, material, material_index):
        """Handle 'table' type material with automatic interpolation."""
        filepath = material['file']
        filepath = Path(filepath).expanduser()
        
        if self.verbose:
            print(f"\n--- Processing table material (index {material_index}) ---")
            print(f"File: {filepath}")
        
        try:
            loader = RefractiveIndexLoader(filepath, verbose=self.verbose)
            
            wavelength_range = self.config.get('wavelength_range', [400, 800, 80])
            target_wavelengths = np.linspace(
                wavelength_range[0],
                wavelength_range[1],
                wavelength_range[2]
            )
            
            n_interp, k_interp = loader.interpolate(target_wavelengths)
            
            refractive_index = n_interp + 1j * k_interp
            epsilon_complex = refractive_index ** 2
            
            self.table_materials_data[material_index] = {
                'wavelengths': target_wavelengths,
                'n': n_interp,
                'k': k_interp,
                'epsilon': epsilon_complex
            }
            
            epsilon_str = self._format_complex_array(epsilon_complex)
            matlab_code = f"epsconst({epsilon_str})"
            
            if self.verbose:
                print(f"Generated MATLAB code with {len(epsilon_complex)} wavelength points")
            
            return matlab_code
        
        except Exception as e:
            raise RuntimeError(f"Error processing table material '{filepath}': {e}")
    
    def _format_complex_array(self, complex_array):
        """Format complex numpy array for MATLAB."""
        values = []
        for val in complex_array:
            real_part = val.real
            imag_part = val.imag
            
            if imag_part >= 0:
                values.append(f"{real_part:.6f}+{imag_part:.6f}i")
            else:
                values.append(f"{real_part:.6f}{imag_part:.6f}i")
        
        return "[" + ", ".join(values) + "]"
    
    def _generate_inout(self):
        """Generate inout matrix based on structure."""
        structure_inout_map = {
            'sphere': self._inout_single,
            'cube': self._inout_single,
            'rod': self._inout_single,
            'ellipsoid': self._inout_single,
            'triangle': self._inout_single,
            'dimer_sphere': self._inout_dimer,
            'sphere_cluster_aggregate': self._inout_sphere_cluster_aggregate,
            'dimer_cube': self._inout_dimer,
            'core_shell_sphere': self._inout_core_shell_single,
            'core_shell_cube': self._inout_core_shell_single,
            'core_shell_rod': self._inout_core_shell_single,
            'dimer_core_shell_cube': self._inout_dimer_core_shell,
            'advanced_dimer_cube': self._inout_advanced_dimer_cube,
            'from_shape': self._inout_from_shape
        }
        
        if self.structure not in structure_inout_map:
            raise ValueError(f"Unknown structure: {self.structure}")
        
        return structure_inout_map[self.structure]()
    
    def _inout_single(self):
        """Inout for single particle."""
        code = "inout = [2, 1];"
        return code
    
    def _inout_dimer(self):
        """Inout for dimer (two identical particles)."""
        code = """inout = [
    2, 1;  % Particle 1
    2, 1   % Particle 2
];"""
        return code
    
    def _inout_core_shell_single(self):
        """Inout for single core-shell particle."""
        code = """inout = [
    2, 3;  % Core (particles{1}): inside=core(2), outside=shell(3)
    3, 1   % Shell (particles{2}): inside=shell(3), outside=medium(1)
];"""
        return code
    
    def _inout_dimer_core_shell(self):
        """Inout for dimer of core-shell particles."""
        code = """inout = [
    2, 3;  % P1-Core: inside=core(2), outside=shell(3)
    3, 1;  % P1-Shell: inside=shell(3), outside=medium(1)
    2, 3;  % P2-Core: inside=core(2), outside=shell(3)
    3, 1   % P2-Shell: inside=shell(3), outside=medium(1)
];"""
        return code

    def _inout_sphere_cluster_aggregate(self):
        """Inout for sphere cluster aggregate."""
        n_spheres = self.config.get('n_spheres', 1)
        
        inout_lines = []
        for i in range(n_spheres):
            if i < n_spheres - 1:
                inout_lines.append(f"    2, 1;  % Sphere {i+1}")
            else:
                inout_lines.append(f"    2, 1   % Sphere {i+1}")
        
        code = "inout = [\n" + "\n".join(inout_lines) + "\n];"
        return code
    
    def _inout_advanced_dimer_cube(self):
        """Inout for advanced dimer cube with multi-shell structure."""
        shell_layers = self.config.get('shell_layers', [])
        n_shells = len(shell_layers)
        use_nonlocal = self.nonlocal_gen.is_needed()

        if use_nonlocal and n_shells == 0:
            # epstab: [medium(1), metal_drude(2), metal_nonlocal(3)]
            # Particles: [P1-outer, P1-inner, P2-outer, P2-inner]
            code = """inout = [
    3, 1;  % P1-Outer: nonlocal(3) inside, medium(1) outside
    2, 3;  % P1-Inner: drude(2) inside, nonlocal(3) outside
    3, 1;  % P2-Outer: nonlocal(3) inside, medium(1) outside
    2, 3   % P2-Inner: drude(2) inside, nonlocal(3) outside
];"""
            return code
        
        elif use_nonlocal and n_shells > 0:

            raise NotImplementedError(
                "Nonlocal with shell_layers is not yet implemented. "
                "Use shell_layers=[] for nonlocal simulations."
            )        
        
        inout_lines = []
        
        # Particle 1
        if n_shells == 0:
            inout_lines.append(f"    2, 1;  % P1-Core")
        else:
            inout_lines.append(f"    2, 3;  % P1-Core: outside=shell1")
        
        for i in range(n_shells):
            shell_num = i + 1
            mat_idx = 2 + shell_num
            
            if i == n_shells - 1:
                inout_lines.append(f"    {mat_idx}, 1;  % P1-Shell{shell_num}: outside=medium")
            else:
                next_shell_mat = mat_idx + 1
                inout_lines.append(f"    {mat_idx}, {next_shell_mat};  % P1-Shell{shell_num}")
        
        # Particle 2
        if n_shells == 0:
            inout_lines.append(f"    2, 1;  % P2-Core")
        else:
            inout_lines.append(f"    2, 3;  % P2-Core: outside=shell1")
        
        for i in range(n_shells):
            shell_num = i + 1
            mat_idx = 2 + shell_num
            
            if i == n_shells - 1:
                inout_lines.append(f"    {mat_idx}, 1;  % P2-Shell{shell_num}: outside=medium")
            else:
                next_shell_mat = mat_idx + 1
                inout_lines.append(f"    {mat_idx}, {next_shell_mat};  % P2-Shell{shell_num}")
        
        if inout_lines:
            inout_lines[-1] = inout_lines[-1].rstrip(';')
        
        code = "inout = [\n" + "\n".join(inout_lines) + "\n];"
        return code
    
    def _inout_from_shape(self):
        """Inout for DDA shape file with multiple materials."""
        n_materials = len(self.config.get('materials', []))
        
        if n_materials == 1:
            code = "inout = [2, 1];"
        elif n_materials == 2:
            code = """inout = [
    2, 1;  % Material 1
    3, 1   % Material 2
];"""
        else:
            inout_lines = []
            for i in range(n_materials):
                mat_idx = i + 2
                inout_lines.append(f"    {mat_idx}, 1;  % Material {i+1}")
            
            if inout_lines:
                inout_lines[-1] = inout_lines[-1].rstrip(';')
            
            code = "inout = [\n" + "\n".join(inout_lines) + "\n];"
        
        return code
    
    def _generate_closed(self):
        """Generate closed surfaces specification."""
        use_nonlocal = self.nonlocal_gen.is_needed()

        structure_closed_map = {
            'sphere': "closed = 1;",
            'cube': "closed = 1;",
            'rod': "closed = 1;",
            'ellipsoid': "closed = 1;",
            'triangle': "closed = 1;",
            'dimer_sphere': "closed = [1, 2];",
            'dimer_cube': "closed = [1, 2];",
            'core_shell_sphere': "closed = [1, 2];",
            'core_shell_cube': "closed = [1, 2];",
            'core_shell_rod': "closed = [1, 2];",
            'dimer_core_shell_cube': "closed = [1, 2, 3, 4];",
            'sphere_cluster_aggregate': self._closed_sphere_cluster_aggregate,
            'advanced_dimer_cube': self._closed_advanced_dimer_cube,
            'from_shape': self._closed_from_shape
        }
        
        if self.structure not in structure_closed_map:
            raise ValueError(f"Unknown structure: {self.structure}")
        
        result = structure_closed_map[self.structure]

        if use_nonlocal and isinstance(result, str):
            import re
            match = re.findall(r'\d+', result)
            if match:
                indices = [int(x) for x in match]
                # 각 입자가 outer + inner 경계를 가지므로 2배
                doubled = []
                for idx in indices:
                    doubled.append(idx * 2 - 1)  # outer
                    doubled.append(idx * 2)      # inner
                return f"closed = [{', '.join(map(str, doubled))}];"


        if callable(result):

            if self.structure == 'advanced_dimer_cube':

                return result(use_nonlocal=use_nonlocal)

            else:

                return result()
        else:
            return result

    
    def _closed_advanced_dimer_cube(self, use_nonlocal=False):
        """Closed surfaces for advanced dimer cube."""
        n_shells = len(self.config.get('shell_layers', []))
        n_particles_base = 2 * (1 + n_shells)

        if use_nonlocal:
            n_particles_total = n_particles_base * 2
            if self.verbose:
                print(f"  ✓ Advanced dimer with nonlocal: {n_particles_base} base → {n_particles_total} with covers")
        else:
            n_particles_total = n_particles_base

        closed_indices = list(range(1, n_particles_total + 1))
        return f"closed = [{', '.join(map(str, closed_indices))}];"
    
    def _closed_from_shape(self):
        """Closed surfaces for DDA shape file."""
        n_materials = len(self.config.get('materials', []))
        
        if n_materials == 0:
            raise ValueError("No materials specified for DDA shape file")
        elif n_materials == 1:
            return "closed = 1;"
        else:
            closed_indices = list(range(1, n_materials + 1))
            return f"closed = [{', '.join(map(str, closed_indices))}];"

    def _closed_sphere_cluster_aggregate(self):
        """Closed surfaces for sphere cluster aggregate."""
        n_spheres = self.config.get('n_spheres', 1)
        closed_indices = list(range(1, n_spheres + 1))
        return f"closed = [{', '.join(map(str, closed_indices))}];"
