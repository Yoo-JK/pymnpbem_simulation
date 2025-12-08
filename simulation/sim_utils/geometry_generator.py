"""
Geometry Generator

Generates MATLAB code for creating various nanoparticle geometries.
Supports:
  - Built-in structures (MNPBEM predefined shapes)
  - DDA .shape files (with material indices)
  - Large mesh support via .mat file export
  - Nonlocal quantum corrections (cover layers)
"""

import numpy as np
from pathlib import Path
from .nonlocal_generator import NonlocalGenerator


# ============================================================================
# DDA Shape File Loader
# ============================================================================

class ShapeFileLoader:
    """Load and process DDA .shape files with material indices."""
    
    def __init__(self, shape_path, voxel_size=1.0, method='surface', verbose=False):
        """
        Initialize shape file loader.
        
        Args:
            shape_path: Path to DDA .shape file
            voxel_size: Physical size of each voxel (nm)
            method: 'surface' (fast) or 'cube' (accurate)
            verbose: Print debug information
        """
        self.shape_path = Path(shape_path)
        self.voxel_size = voxel_size
        self.method = method
        self.verbose = verbose
        
        if not self.shape_path.exists():
            raise FileNotFoundError(f"Shape file not found: {self.shape_path}")
        
        if method not in ['surface', 'cube']:
            raise ValueError(f"method must be 'surface' or 'cube', got '{method}'")
        
        self.voxel_data = None  # Will store [i, j, k, mat_idx]
        self.unique_materials = None
        self.material_particles = {}  # {mat_idx: {'vertices': ..., 'faces': ...}}
    
    def load(self):
        """Load shape file and extract voxel data with materials."""
        if self.verbose:
            print(f"  Loading DDA shape file: {self.shape_path}")
        
        # Read file and skip non-numeric lines
        data_lines = []
        with open(self.shape_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not (line[0].isdigit() or line[0] == '-'):
                    if self.verbose:
                        print(f"    Skipping header/comment line: {line}")
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        i, j, k, mat_type = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        data_lines.append([i, j, k, mat_type])
                except (ValueError, IndexError):
                    if self.verbose:
                        print(f"    Skipping invalid line: {line}")
                    continue
        
        if not data_lines:
            raise ValueError(f"No valid voxel data found in {self.shape_path}")
        
        data = np.array(data_lines, dtype=int)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 4:
            raise ValueError(
                f"Shape file must have at least 4 columns [i,j,k,mat_type], "
                f"got {data.shape[1]} columns"
            )
        
        self.voxel_data = data[:, :4]
        self.unique_materials = np.unique(self.voxel_data[:, 3])
        
        if self.verbose:
            print(f"    Total voxels: {len(self.voxel_data)}")
            print(f"    Unique materials: {self.unique_materials.tolist()}")
            for mat_idx in self.unique_materials:
                count = np.sum(self.voxel_data[:, 3] == mat_idx)
                print(f"      Material {mat_idx}: {count} voxels")
        
        # Convert each material to mesh
        for mat_idx in self.unique_materials:
            mat_voxels = self.voxel_data[self.voxel_data[:, 3] == mat_idx][:, :3]
            
            if self.verbose:
                print(f"    Converting material {mat_idx}...")
            
            if self.method == 'surface':
                vertices, faces = self._voxels_to_surface_mesh(mat_voxels)
            else:
                vertices, faces = self._voxels_to_cube_mesh(mat_voxels)
            
            self.material_particles[mat_idx] = {
                'vertices': vertices,
                'faces': faces
            }
            
            if self.verbose:
                print(f"      → {len(vertices)} vertices, {len(faces)} faces")
        
        return self.material_particles
    
    def _voxels_to_surface_mesh(self, voxel_coords):
        """Convert voxels to surface mesh (only external faces)."""
        voxel_set = set(map(tuple, voxel_coords))
        
        vertices_list = []
        faces_list = []
        vertex_map = {}
        
        cube_face_offsets = [
            [[0,0,0], [1,0,0], [1,1,0], [0,1,0]],  # bottom
            [[0,0,1], [0,1,1], [1,1,1], [1,0,1]],  # top
            [[0,0,0], [0,1,0], [0,1,1], [0,0,1]],  # left
            [[1,0,0], [1,0,1], [1,1,1], [1,1,0]],  # right
            [[0,0,0], [0,0,1], [1,0,1], [1,0,0]],  # front
            [[0,1,0], [1,1,0], [1,1,1], [0,1,1]]   # back
        ]
        
        neighbors = [
            [0, 0, -1], [0, 0, 1], [-1, 0, 0],
            [1, 0, 0], [0, -1, 0], [0, 1, 0]
        ]
        
        for voxel in voxel_coords:
            i, j, k = voxel
            
            for face_idx, neighbor_offset in enumerate(neighbors):
                neighbor = (i + neighbor_offset[0],
                           j + neighbor_offset[1],
                           k + neighbor_offset[2])
                
                if neighbor not in voxel_set:
                    face_verts_offsets = cube_face_offsets[face_idx]
                    vert_indices = []
                    
                    for vert_offset in face_verts_offsets:
                        vx = (i + vert_offset[0]) * self.voxel_size
                        vy = (j + vert_offset[1]) * self.voxel_size
                        vz = (k + vert_offset[2]) * self.voxel_size
                        vert_key = (vx, vy, vz)
                        
                        if vert_key not in vertex_map:
                            vertex_map[vert_key] = len(vertices_list)
                            vertices_list.append([vx, vy, vz])
                        
                        vert_indices.append(vertex_map[vert_key] + 1)
                    
                    faces_list.append([vert_indices[0], vert_indices[1], vert_indices[2], np.nan])
                    faces_list.append([vert_indices[0], vert_indices[2], vert_indices[3], np.nan])
        
        return np.array(vertices_list), np.array(faces_list)
    
    def _voxels_to_cube_mesh(self, voxel_coords):
        """Convert each voxel to a cube mesh."""
        cube_vert_template = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)
        
        cube_face_template = np.array([
            [1, 2, 3, np.nan], [1, 3, 4, np.nan],
            [5, 8, 7, np.nan], [5, 7, 6, np.nan],
            [1, 5, 6, np.nan], [1, 6, 2, np.nan],
            [4, 3, 7, np.nan], [4, 7, 8, np.nan],
            [1, 4, 8, np.nan], [1, 8, 5, np.nan],
            [2, 6, 7, np.nan], [2, 7, 3, np.nan]
        ])
        
        all_verts = []
        all_faces = []
        
        for voxel in voxel_coords:
            i, j, k = voxel
            
            cube_verts = cube_vert_template * self.voxel_size
            cube_verts[:, 0] += i * self.voxel_size
            cube_verts[:, 1] += j * self.voxel_size
            cube_verts[:, 2] += k * self.voxel_size
            
            vert_offset = len(all_verts)
            all_verts.extend(cube_verts)
            all_faces.extend(cube_face_template + vert_offset)
        
        return np.array(all_verts), np.array(all_faces)
    
    def generate_matlab_code(self, material_names, output_dir=None):
        """Generate MATLAB code for all material particles."""
        if self.material_particles is None:
            raise RuntimeError("Shape file not loaded. Call load() first.")
        
        if isinstance(material_names, list):
            mat_name_dict = {i+1: name for i, name in enumerate(material_names)}
        else:
            mat_name_dict = material_names
        
        code = """
%% Geometry: From DDA Shape File
fprintf('Creating particles from DDA shape file...\\n');
fprintf('  Voxel size: %.2f nm\\n', {voxel_size});
fprintf('  Method: {method}\\n');
fprintf('  Number of materials: %d\\n', {n_materials});

""".format(
            voxel_size=self.voxel_size,
            method=self.method,
            n_materials=len(self.unique_materials)
        )
        
        use_mat_files = False
        total_vertices = sum(len(data['vertices']) for data in self.material_particles.values())
        
        if total_vertices > 100000:
            use_mat_files = True
            if output_dir is None:
                raise ValueError("output_dir must be provided for large meshes")
            
            if self.verbose:
                print(f"  Large mesh detected ({total_vertices} vertices)")
                print(f"  Saving geometry data to .mat files in {output_dir}")
        
        particles_list = []
        
        for mat_idx in sorted(self.unique_materials):
            data = self.material_particles[mat_idx]
            vertices = data['vertices']
            faces = data['faces']
            
            mat_name = mat_name_dict.get(mat_idx, f'material_{mat_idx}')
            
            if use_mat_files:
                try:
                    import scipy.io as sio
                except ImportError:
                    raise ImportError("scipy is required for saving large meshes")
                
                mat_filename = f'geometry_mat{mat_idx}.mat'
                mat_filepath = Path(output_dir) / mat_filename
                
                faces_matlab = faces.copy()
                if faces_matlab.min() == 0:
                    faces_matlab = faces_matlab + 1
                
                sio.savemat(
                    str(mat_filepath),
                    {
                        f'verts_{mat_idx}': vertices,
                        f'faces_{mat_idx}': faces_matlab
                    },
                    do_compression=True
                )
                
                if self.verbose:
                    print(f"    Saved material {mat_idx} to {mat_filename}")
                
                code += f"""
% Material index {mat_idx}: {mat_name}
fprintf('  Loading material {mat_idx} ({mat_name}) from file...\\n');
geom_data_{mat_idx} = load('{mat_filename}');
verts_{mat_idx} = geom_data_{mat_idx}.verts_{mat_idx};
faces_{mat_idx} = geom_data_{mat_idx}.faces_{mat_idx};

% DDA meshes use flat triangular faces only
p{mat_idx} = particle(verts_{mat_idx}, faces_{mat_idx}, op, 'interp', 'flat');
fprintf('  Material {mat_idx} ({mat_name}): %d vertices, %d faces\\n', ...
        size(verts_{mat_idx}, 1), size(faces_{mat_idx}, 1));
"""
            else:
                verts_str, faces_str = self._arrays_to_matlab(vertices, faces)
                
                code += f"""
% Material index {mat_idx}: {mat_name}
verts_{mat_idx} = {verts_str};
faces_{mat_idx} = {faces_str};

% DDA meshes use flat triangular faces only
p{mat_idx} = particle(verts_{mat_idx}, faces_{mat_idx}, op, 'interp', 'flat');
fprintf('  Material {mat_idx} ({mat_name}): %d vertices, %d faces\\n', ...
        size(verts_{mat_idx}, 1), size(faces_{mat_idx}, 1));
"""
            
            particles_list.append(f'p{mat_idx}')
        
        particles_str = ', '.join(particles_list)
        code += f"\nparticles = {{{particles_str}}};\n"
        
        return code
    
    def _arrays_to_matlab(self, vertices, faces):
        """Convert numpy arrays to MATLAB format."""
        verts_str = "[\n"
        for v in vertices:
            verts_str += f"    {v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f};\n"
        verts_str += "]"
        
        faces_str = "[\n"
        for f in faces:
            if len(f) >= 4:
                if np.isnan(f[3]):
                    faces_str += f"    {int(f[0])}, {int(f[1])}, {int(f[2])}, NaN;\n"
                else:
                    faces_str += f"    {int(f[0])}, {int(f[1])}, {int(f[2])}, {int(f[3])};\n"
            else:
                faces_str += f"    {int(f[0])}, {int(f[1])}, {int(f[2])}, NaN;\n"
        faces_str += "]"
        
        return verts_str, faces_str


# ============================================================================
# Geometry Generator
# ============================================================================

class GeometryGenerator:
    """Generates geometry-related MATLAB code."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.structure = config['structure']
        self.nonlocal_gen = NonlocalGenerator(config, verbose)
    
    def generate(self):
        """Generate geometry code based on structure type."""
        structure_map = {
            'sphere': self._sphere,
            'cube': self._cube,
            'rod': self._rod,
            'ellipsoid': self._ellipsoid,
            'triangle': self._triangle,
            'dimer_sphere': self._dimer_sphere,
            'dimer_cube': self._dimer_cube,
            'core_shell_sphere': self._core_shell_sphere,
            'core_shell_cube': self._core_shell_cube,
            'core_shell_rod': self._core_shell_rod,
            'dimer_core_shell_cube': self._dimer_core_shell_cube,
            'advanced_dimer_cube': self._advanced_dimer_cube,
            'sphere_cluster_aggregate': self._sphere_cluster_aggregate,
            'from_shape': self._from_shape,
        }
        
        if self.structure not in structure_map:
            raise ValueError(f"Unknown structure type: {self.structure}")
        
        # Generate base geometry
        base_code = structure_map[self.structure]()
        
        # Apply nonlocal cover layers if enabled
        if self.nonlocal_gen.is_needed():
            base_code = self._apply_nonlocal_coverlayer(base_code)
        
        return base_code

    def _mesh_density_to_n_rod(self, mesh_density):
        """
        Convert single mesh_density value to [nphi, ntheta, nz] for trirod.
        
        MNPBEM's trirod default is [15, 20, 20] which gives ~900 vertices.
        This method maintains similar aspect ratios for different densities.
        
        Args:
            mesh_density (int): Target mesh density (similar to sphere)
            
        Returns:
            list: [nphi, ntheta, nz] for trirod
        """
        
        # Predefined mapping for common values
        MESH_MAPPING = {
            32:   [8,   10,  10],   # Very coarse (~240 vertices)
            60:   [10,  12,  12],   # Coarse (~360 vertices)
            144:  [15,  20,  20],   # Standard (~900 vertices) - MNPBEM default
            256:  [20,  25,  25],   # Medium (~1400 vertices)
            400:  [25,  30,  30],   # Fine (~2100 vertices)
            576:  [30,  35,  35],   # Very fine (~3150 vertices)
            900:  [35,  40,  40],   # Ultra fine (~4200 vertices)
        }
        
        # Return exact match if available
        if mesh_density in MESH_MAPPING:
            if self.verbose:
                print(f"    mesh_density {mesh_density} → {MESH_MAPPING[mesh_density]}")
            return MESH_MAPPING[mesh_density]
        
        # Find closest match (within 15%)
        closest = min(MESH_MAPPING.keys(), key=lambda x: abs(x - mesh_density))
        if abs(closest - mesh_density) < mesh_density * 0.15:
            if self.verbose:
                print(f"    mesh_density {mesh_density} rounded to {closest} → {MESH_MAPPING[closest]}")
            return MESH_MAPPING[closest]
        
        # Calculate for custom values
        # Total vertices ≈ nphi * (2*ntheta + nz)
        # Maintain ratio nphi:ntheta:nz = 15:20:20 = 3:4:4
        k = np.sqrt(mesh_density / 4.0)
        nphi = max(8, int(np.round(k)))
        ntheta = max(10, int(np.round(k * 4.0 / 3.0)))
        nz = max(10, int(np.round(k * 4.0 / 3.0)))
        
        result = [nphi, ntheta, nz]
        if self.verbose:
            approx_verts = nphi * (2 * ntheta + nz)
            print(f"    mesh_density {mesh_density} (custom) → {result} (~{approx_verts} vertices)")
        
        return result
    
    def _apply_nonlocal_coverlayer(self, base_geometry_code):
        """Apply nonlocal cover layer to existing geometry."""
        if not self.nonlocal_gen.is_needed():
            return base_geometry_code
        
        is_applicable, warnings = self.nonlocal_gen.check_applicability()
        if warnings:
            warning_str = "\n".join([f"%   - {w}" for w in warnings])
            warning_code = f"""
%% [!] Nonlocal Warnings:
{warning_str}
"""
        else:
            warning_code = ""
        
        structure = self.config.get('structure', '')
        
        if 'sphere' in structure:
            cover_code = self._apply_coverlayer_sphere()
        elif 'cube' in structure:
            cover_code = self._apply_coverlayer_cube()
        else:
            if self.verbose:
                print(f"  ⚠ Warning: Nonlocal not fully implemented for structure '{structure}'")
            cover_code = self._apply_coverlayer_manual()
        
        combined_code = f"""
{base_geometry_code}

{warning_code}

%% Apply Nonlocal Cover Layers
fprintf('\\n=== Applying Nonlocal Cover Layers ===\\n');
{cover_code}
"""
        return combined_code
    
    def _apply_coverlayer_sphere(self):
        """Apply cover layer to sphere structures."""
        d = self.nonlocal_gen.cover_thickness
        
        code = f"""
% Apply nonlocal cover layers to spheres
d_cover = {d};
particles_with_cover = {{}};

for i = 1:length(particles)
    p_inner = particles{{i}};
    p_outer = coverlayer.shift( p_inner, d_cover );
    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner;
    fprintf('  [OK] Particle %d: added %.3f nm cover layer\\n', i, d_cover);
end

particles = particles_with_cover;
fprintf('  Total particles after cover layers: %d\\n', length(particles));
"""
        return code
    
    def _apply_coverlayer_cube(self):
        """Apply cover layer to cube structures."""
        d = self.nonlocal_gen.cover_thickness
        structure = self.config.get('structure', '')

        if structure == 'advanced_dimer_cube':
            shell_layers = self.config.get('shell_layers', [])
            if len(shell_layers) > 0:
                core_size = self.config.get('core_size', 30)
                materials = self.config.get('materials', [])
                roundings = self.config.get('roundings', None)
                if roundings is None:
                    rounding = self.config.get('rounding', 0.25)
                    roundings = [rounding] * len(materials)
                mesh = self.config.get('mesh_density', 12)

                code = f"""
% Apply nonlocal cover layers to core-shell structure
d_cover = {d};
particles_with_cover = {{}};

for i = 1:length(particles)
    p_outer = particles{{i}};
    verts = p_outer.verts;
    current_size = max(verts(:,1)) - min(verts(:,1));
    
    layer_idx = mod(i-1, {len(materials)}) + 1;
    roundings_array = {roundings};
    rounding_val = roundings_array(layer_idx);
    
    size_inner = current_size - 2*d_cover;
    p_inner = tricube({mesh}, size_inner, 'e', rounding_val);
    
    center_outer = mean(p_outer.verts, 1);
    center_inner = mean(p_inner.verts, 1);
    p_inner = shift(p_inner, center_outer - center_inner);
    
    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner;
end

particles = particles_with_cover;
fprintf('  Total boundaries: %d\\n', length(particles));
"""
                return code
            
            # advanced_dimer_cube 파라미터 가져오기
            core_size = self.config.get('core_size', 30)
            roundings = self.config.get('roundings', None)
            if roundings is None:
                rounding = self.config.get('rounding', 0.25)
            else:
                rounding = roundings[0]
            mesh = self.config.get('mesh_density', 12)
            
            code = f"""
% Apply nonlocal cover layers to advanced_dimer_cube
d_cover = {d};
particles_with_cover = {{}};

fprintf('  Applying cover layers to advanced_dimer_cube...\\n');

for i = 1:length(particles)
    p_outer = particles{{i}};  % Original cube (outer boundary)
    
    % Create inner boundary (smaller cube)
    size_inner = {core_size} - 2*d_cover;
    p_inner = tricube({mesh}, size_inner, 'e', {rounding});
    
    % Align centers
    center_outer = mean(p_outer.verts, 1);
    center_inner = mean(p_inner.verts, 1);
    p_inner = shift(p_inner, center_outer - center_inner);
    
    % Add: outer first, then inner
    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner;
    
    fprintf('    [OK] Particle %d: cover layer %.3f nm\\n', i, d_cover);
end

particles = particles_with_cover;
fprintf('  Total boundaries: %d\\n', length(particles));
"""
            return code        
        
        else:

            code = f"""
% Apply nonlocal cover layers to cubes
d_cover = {d};
particles_with_cover = {{}};

for i = 1:length(particles)
    p_inner = particles{{i}};
    verts = p_inner.verts;
    size_current = max(verts(:,1)) - min(verts(:,1));
    
    rounding_current = {self.config.get('rounding', 0.25)};
    mesh_current = {self.config.get('mesh_density', 12)};
    
    p_inner_smaller = tricube(mesh_current, size_current - 2*d_cover, 'e', rounding_current);
    center_orig = mean(p_inner.verts, 1);
    center_new = mean(p_inner_smaller.verts, 1);
    p_inner_smaller = shift(p_inner_smaller, center_orig - center_new);
    
    p_outer = p_inner;
    particles_with_cover{{end+1}} = p_outer;
    particles_with_cover{{end+1}} = p_inner_smaller;
    
    fprintf('  [OK] Particle %d: added %.3f nm cover layer (cube)\\n', i, d_cover);
end

particles = particles_with_cover;
fprintf('  Total particles after cover layers: %d\\n', length(particles));
"""
            return code
    
    def _apply_coverlayer_manual(self):
        """Generic cover layer application."""
        d = self.nonlocal_gen.cover_thickness
        
        code = f"""
% Manual cover layer application
d_cover = {d};

fprintf('  [!] Manual cover layer mode\\n');
fprintf('  [!] Verify geometry visually before running full simulation!\\n');

if length(particles) > 0
    p1 = particles{{1}};
    p1_outer = coverlayer.shift( p1, d_cover );
    particles = {{p1_outer, p1}};
    fprintf('  [OK] Applied cover layer to first particle\\n');
end
"""
        return code
    
    # ========================================================================
    # Built-in Structures
    # ========================================================================
    
    def _sphere(self):
        """Generate code for single sphere."""
        diameter = self.config.get('diameter', 10)
        mesh = self.config.get('mesh_density', 144)
        
        code = f"""
%% Geometry: Single Sphere
diameter = {diameter};
p = trisphere({mesh}, diameter);
particles = {{p}};
"""
        return code
    
    def _cube(self):
        """Generate code for single cube."""
        size = self.config.get('size', 20)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        
        code = f"""
%% Geometry: Single Cube
cube_size = {size};
rounding_param = {rounding};
p = tricube({mesh}, cube_size, 'e', rounding_param);
particles = {{p}};
"""
        return code
    
    def _rod(self):
        """Generate code for rod/cylinder (horizontal).
        
        Mesh can be specified in two ways:
        1. mesh_density (auto-calculated)
        2. rod_mesh = [nphi, ntheta, nz] (manual)
        """
        diameter = self.config.get('diameter', 10)
        height = self.config.get('height', 50)
        
        # Check if user provided explicit mesh parameters
        if 'rod_mesh' in self.config:
            # User specifies [nphi, ntheta, nz] directly
            n = self.config['rod_mesh']
            nphi, ntheta, nz = n
            
            code = f"""
%% Geometry: Rod (horizontal along x-axis)
diameter = {diameter};
height = {height};

% User-specified mesh: [{nphi}, {ntheta}, {nz}]
p = trirod(diameter, height, [{nphi}, {ntheta}, {nz}]);
p = rot(p, 90, [0, 1, 0]);

particles = {{p}};
"""
        else:
            # Use mesh_density (existing behavior)
            mesh = self.config.get('mesh_density', 144)
            n = self._mesh_density_to_n_rod(mesh)
            
            code = f"""
%% Geometry: Rod (horizontal along x-axis)
diameter = {diameter};
height = {height};

% Auto-calculated mesh from mesh_density={mesh}: {n}
p = trirod(diameter, height, {n});
p = rot(p, 90, [0, 1, 0]);

particles = {{p}};
"""
        
        return code
    
    def _ellipsoid(self):
        """Generate code for ellipsoid."""
        axes = self.config.get('axes', [10, 15, 20])
        mesh = self.config.get('mesh_density', 144)
        
        code = f"""
%% Geometry: Ellipsoid
p = trisphere({mesh}, 1);
p.verts(:, 1) = p.verts(:, 1) * {axes[0]};
p.verts(:, 2) = p.verts(:, 2) * {axes[1]};
p.verts(:, 3) = p.verts(:, 3) * {axes[2]};
particles = {{p}};
"""
        return code
    
    def _triangle(self):
        """Generate code for triangular nanoparticle."""
        side_length = self.config.get('side_length', 30)
        thickness = self.config.get('thickness', 5)
        
        code = f"""
%% Geometry: Triangle
side_length = {side_length};
thickness = {thickness};
poly = round(polygon(3, 'size', [side_length, side_length * 2/sqrt(3)]));
edge = edgeprofile(thickness, 11);
p = tripolygon(poly, edge);
particles = {{p}};
"""
        return code
    
    def _dimer_sphere(self):
        """Generate code for two coupled spheres."""
        diameter = self.config.get('diameter', 10)
        gap = self.config.get('gap', 5)
        mesh = self.config.get('mesh_density', 144)
        
        code = f"""
%% Geometry: Dimer - Two Spheres
diameter = {diameter};
gap = {gap};
shift_distance = (diameter + gap) / 2;

p1 = trisphere({mesh}, diameter);
p1 = shift(p1, [-shift_distance, 0, 0]);

p2 = trisphere({mesh}, diameter);
p2 = shift(p2, [shift_distance, 0, 0]);

particles = {{p1, p2}};
"""
        return code

    def _sphere_cluster_aggregate(self):
        """Generate compact sphere cluster (close-packed aggregate structure).

        Structures:
            N=1: Single sphere
            N=2: Dimer (horizontal)
            N=3: Triangle (2 bottom, 1 top)
            N=4: Center + 3 surrounding (hexagonal positions)
            N=5: Center + 4 surrounding
            N=6: Center + 5 surrounding
            N=7: Center + 6 surrounding (complete hexagon, close-packed)

        For N=4~7, spheres are arranged with one center sphere and surrounding
        spheres placed at 60° intervals (hexagonal pattern). At N=7, all 6
        surrounding positions are filled, creating a perfect close-packed structure.

        Gap parameter controls spacing between all contacting sphere pairs.
        """
        n_spheres = self.config.get('n_spheres', 1)
        diameter = self.config.get('diameter', 50)
        gap = self.config.get('gap', -0.1)
        mesh = self.config.get('mesh_density', 144)
        
        # Center-to-center spacing for contact
        spacing = diameter + gap

        # 60-degree triangle height
        dy_60deg = spacing * 0.866025404  # sin(60°) = sqrt(3)/2

        # Hexagonal surrounding positions (60° intervals, starting from +x direction)
        # Used for N=4~7: center + surrounding spheres
        hex_positions = []
        for i in range(6):
            angle = i * 60 * np.pi / 180  # 0°, 60°, 120°, 180°, 240°, 300°
            x = spacing * np.cos(angle)
            y = spacing * np.sin(angle)
            hex_positions.append((x, y))

        # Define xy positions for each cluster (z=0 for all, substrate contact handled separately)
        # Format: [(x, y), ...]
        # N=1,2,3: Original configurations
        # N=4~7: Center sphere + surrounding spheres in hexagonal positions
        cluster_positions = {
            1: [(0, 0)],

            2: [(-spacing/2, 0),
                (spacing/2, 0)],

            3: [(-spacing/2, 0),         # bottom-left
                (spacing/2, 0),          # bottom-right
                (0, dy_60deg)],          # top

            # N=4~7: Center (0,0) + hexagonal surrounding positions
            4: [(0, 0)] + hex_positions[0:3],  # center + 3 surrounding

            5: [(0, 0)] + hex_positions[0:4],  # center + 4 surrounding

            6: [(0, 0)] + hex_positions[0:5],  # center + 5 surrounding

            7: [(0, 0)] + hex_positions[0:6],  # center + 6 surrounding (complete hexagon)
        }
        
        if n_spheres not in cluster_positions:
            raise ValueError(f"n_spheres must be 1-7, got {n_spheres}")
        
        positions = cluster_positions[n_spheres]
        
        # Generate MATLAB code
        code = f"""
%% Geometry: Compact Sphere Cluster (Close-Packed Aggregate)
n_spheres = {n_spheres};
diameter = {diameter};
gap = {gap};  % negative = 0.1nm overlap (contact)
spacing = diameter + gap;  % {spacing:.3f} nm

fprintf('\\n=== Creating Compact Sphere Cluster ===\\n');
fprintf('  Number of spheres: %d\\n', n_spheres);
fprintf('  Diameter: %.2f nm\\n', diameter);
fprintf('  Gap: %.3f nm (%.1f nm overlap)\\n', gap, abs(gap));
fprintf('  Center-to-center spacing: %.3f nm\\n', spacing);
fprintf('  Structure type: ');

% Define positions for each sphere
positions = [
"""
        
        # Add position coordinates
        for i, (x, y) in enumerate(positions):
            if i < len(positions) - 1:
                code += f"    {x:.6f}, {y:.6f}, 0;  % Sphere {i+1}\n"
            else:
                code += f"    {x:.6f}, {y:.6f}, 0   % Sphere {i+1}\n"
        
        code += """];

% Determine structure name
switch n_spheres
    case 1
        fprintf('Single sphere\\n');
    case 2
        fprintf('Dimer\\n');
    case 3
        fprintf('Triangle\\n');
    case 4
        fprintf('Center + 3 surrounding\\n');
    case 5
        fprintf('Center + 4 surrounding\\n');
    case 6
        fprintf('Center + 5 surrounding\\n');
    case 7
        fprintf('Center + 6 surrounding (complete hexagon)\\n');
end

% Create particles
particles = {};
for i = 1:n_spheres
    % Create sphere
    p_sphere = trisphere(""" + f"{mesh}" + """, diameter);
    
    % Shift to position
    p_sphere = shift(p_sphere, positions(i, :));
    
    % Add to particle list
    particles{end+1} = p_sphere;
    
    fprintf('  Sphere %d: (%.2f, %.2f, 0) nm\\n', i, positions(i, 1), positions(i, 2));
end

% Calculate cluster bounds
x_coords = positions(:, 1);
y_coords = positions(:, 2);
x_min = min(x_coords) - diameter/2;
x_max = max(x_coords) + diameter/2;
y_min = min(y_coords) - diameter/2;
y_max = max(y_coords) + diameter/2;

fprintf('  Cluster bounds: x=[%.2f, %.2f], y=[%.2f, %.2f] nm\\n', ...
        x_min, x_max, y_min, y_max);
fprintf('  All spheres in XY plane (z=0)\\n');
fprintf('=================================\\n');
"""
        
        return code
    
    def _dimer_cube(self):
        """Generate code for two coupled cubes."""
        size = self.config.get('size', 20)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        
        code = f"""
%% Geometry: Dimer - Two Cubes
cube_size = {size};
gap = {gap};
rounding_param = {rounding};
shift_distance = (cube_size + gap) / 2;

p1 = tricube({mesh}, cube_size, 'e', rounding_param);
p1 = shift(p1, [-shift_distance, 0, 0]);

p2 = tricube({mesh}, cube_size, 'e', rounding_param);
p2 = shift(p2, [shift_distance, 0, 0]);

particles = {{p1, p2}};
"""
        return code
    
    def _core_shell_sphere(self):
        """Generate code for core-shell sphere."""
        core_diameter = self.config.get('core_diameter', 10)
        shell_thickness = self.config.get('shell_thickness', 5)
        mesh = self.config.get('mesh_density', 144)
        shell_diameter = core_diameter + 2 * shell_thickness
        
        code = f"""
%% Geometry: Core-Shell Sphere
core_diameter = {core_diameter};
shell_thickness = {shell_thickness};
shell_diameter = core_diameter + 2 * shell_thickness;

p_core = trisphere({mesh}, core_diameter);
p_shell = trisphere({mesh}, shell_diameter);

particles = {{p_core, p_shell}};
"""
        return code
    
    def _core_shell_cube(self):
        """Generate code for core-shell cube."""
        core_size = self.config.get('core_size')
        shell_thickness = self.config.get('shell_thickness')
        rounding = self.config.get('rounding')
        mesh = self.config.get('mesh_density', 12)
        shell_size = core_size + 2 * shell_thickness
        
        code = f"""
%% Geometry: Core-Shell Cube
core_size = {core_size};
shell_thickness = {shell_thickness};
shell_size = core_size + 2 * shell_thickness;
rounding_param = {rounding};

p_core = tricube({mesh}, core_size, 'e', rounding_param);
p_shell = tricube({mesh}, shell_size, 'e', rounding_param);

particles = {{p_core, p_shell}};
"""
        return code

    def _core_shell_rod(self):
        """Generate code for core-shell rod with complete shell coverage.
        
        Mesh specification:
            Option 1: mesh_density (auto-calculated)
            Option 2: rod_mesh = [nphi, ntheta, nz] (manual override)
        """
        core_diameter = self.config.get('core_diameter', 15)
        shell_thickness = self.config.get('shell_thickness', 5)
        height = self.config.get('height', 80)
        
        # Check for manual mesh specification
        if 'rod_mesh' in self.config:
            n = self.config['rod_mesh']
            if len(n) != 3:
                raise ValueError(f"rod_mesh must have 3 values [nphi, ntheta, nz], got {n}")
        else:
            # Auto-calculate from mesh_density
            mesh = self.config.get('mesh_density', 144)
            n = self._mesh_density_to_n_rod(mesh)
        
        shell_diameter = core_diameter + 2 * shell_thickness
        shell_height = height
        core_height = height - 2 * shell_thickness
        
        code = f"""
%% Geometry: Core-Shell Rod
core_diameter = {core_diameter};
shell_thickness = {shell_thickness};
shell_diameter = core_diameter + 2 * shell_thickness;
shell_height = {height};
core_height = shell_height - 2 * shell_thickness;

% Mesh: {n}
% Create rod particles (initially standing along z-axis)
p_core = trirod(core_diameter, core_height, {n}, 'triangles');
p_shell = trirod(shell_diameter, shell_height, {n}, 'triangles');

% Rotate 90 degrees to lie down along x-axis
p_core = rot(p_core, 90, [0, 1, 0]);
p_shell = rot(p_shell, 90, [0, 1, 0]);

particles = {{p_core, p_shell}};
"""
        return code
    
    def _dimer_core_shell_cube(self):
        """Generate code for two core-shell cubes."""
        core_size = self.config.get('core_size', 20)
        shell_thickness = self.config.get('shell_thickness', 5)
        gap = self.config.get('gap', 10)
        rounding = self.config.get('rounding', 0.25)
        mesh = self.config.get('mesh_density', 12)
        shell_size = core_size + 2 * shell_thickness
        
        code = f"""
%% Geometry: Dimer Core-Shell Cubes
core_size = {core_size};
shell_thickness = {shell_thickness};
shell_size = core_size + 2 * shell_thickness;
gap = {gap};
rounding_param = {rounding};
shift_distance = (shell_size + gap) / 2;

% Particle 1 (Left)
core1 = tricube({mesh}, core_size, 'e', rounding_param);
core1 = shift(core1, [-shift_distance, 0, 0]);

shell1 = tricube({mesh}, shell_size, 'e', rounding_param);
shell1 = shift(shell1, [-shift_distance, 0, 0]);

% Particle 2 (Right)
core2 = tricube({mesh}, core_size, 'e', rounding_param);
core2 = shift(core2, [shift_distance, 0, 0]);

shell2 = tricube({mesh}, shell_size, 'e', rounding_param);
shell2 = shift(shell2, [shift_distance, 0, 0]);

particles = {{core1, shell1, core2, shell2}};
"""
        return code
    
    def _advanced_dimer_cube(self):
        """Generate advanced dimer cube with full control."""
        core_size = self.config.get('core_size', 30)
        shell_layers = self.config.get('shell_layers', [])
        materials = self.config.get('materials', [])
        mesh = self.config.get('mesh_density', 12)
        
        if len(materials) != 1 + len(shell_layers):
            raise ValueError(
                f"materials length ({len(materials)}) must equal "
                f"1 (core) + {len(shell_layers)} (shells) = {1 + len(shell_layers)}"
            )
        
        if 'roundings' in self.config:
            roundings = self.config.get('roundings')
            if len(roundings) != len(materials):
                raise ValueError(
                    f"roundings length ({len(roundings)}) must equal "
                    f"materials length ({len(materials)})"
                )
        elif 'rounding' in self.config:
            single_rounding = self.config.get('rounding', 0.25)
            roundings = [single_rounding] * len(materials)
        else:
            roundings = [0.25] * len(materials)
        
        gap = self.config.get('gap', 10)
        offset = self.config.get('offset', [0, 0, 0])
        tilt_angle = self.config.get('tilt_angle', 0)
        tilt_axis = self.config.get('tilt_axis', [0, 1, 0])
        rotation_angle = self.config.get('rotation_angle', 0)
        
        sizes = [core_size]
        for thickness in shell_layers:
            sizes.append(sizes[-1] + 2 * thickness)
        
        total_size = sizes[-1]
        shift_distance = (total_size + gap) / 2
        
        code = f"""
%% Geometry: Advanced Dimer Cube
mesh_density = {mesh};
gap = {gap};
shift_distance = {shift_distance};

"""
        
        # Particle 1
        code += "\n%% === Particle 1 (Left) ===\n"
        particles_list = []
        
        for i, (size, material, rounding) in enumerate(zip(sizes, materials, roundings)):
            if i == 0:
                code += f"% Core: {material}\n"
                code += f"p1_core = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p1_core = shift(p1_core, [-shift_distance, 0, 0]);\n"
                particles_list.append("p1_core")
            else:
                shell_num = i
                code += f"\n% Shell {shell_num}: {material}\n"
                code += f"p1_shell{shell_num} = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p1_shell{shell_num} = shift(p1_shell{shell_num}, [-shift_distance, 0, 0]);\n"
                particles_list.append(f"p1_shell{shell_num}")
        
        # Particle 2
        code += "\n%% === Particle 2 (Right with transformations) ===\n"
        
        for i, (size, material, rounding) in enumerate(zip(sizes, materials, roundings)):
            if i == 0:
                code += f"% Core: {material}\n"
                code += f"p2_core = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p2_core = rot(p2_core, {rotation_angle}, [0, 0, 1]);\n"
                code += f"p2_core = rot(p2_core, {tilt_angle}, [{tilt_axis[0]}, {tilt_axis[1]}, {tilt_axis[2]}]);\n"
                code += f"p2_core = shift(p2_core, [shift_distance, 0, 0]);\n"
                code += f"p2_core = shift(p2_core, [{offset[0]}, {offset[1]}, {offset[2]}]);\n"
                particles_list.append("p2_core")
            else:
                shell_num = i
                code += f"\n% Shell {shell_num}: {material}\n"
                code += f"p2_shell{shell_num} = tricube(mesh_density, {size}, 'e', {rounding});\n"
                code += f"p2_shell{shell_num} = rot(p2_shell{shell_num}, {rotation_angle}, [0, 0, 1]);\n"
                code += f"p2_shell{shell_num} = rot(p2_shell{shell_num}, {tilt_angle}, [{tilt_axis[0]}, {tilt_axis[1]}, {tilt_axis[2]}]);\n"
                code += f"p2_shell{shell_num} = shift(p2_shell{shell_num}, [shift_distance, 0, 0]);\n"
                code += f"p2_shell{shell_num} = shift(p2_shell{shell_num}, [{offset[0]}, {offset[1]}, {offset[2]}]);\n"
                particles_list.append(f"p2_shell{shell_num}")
        
        particles_str = ", ".join(particles_list)
        code += f"\n%% Combine all particles\nparticles = {{{particles_str}}};\n"
        
        return code
    
    def _from_shape(self):
        """Generate code for DDA shape file import."""
        shape_file = self.config.get('shape_file')
        if not shape_file:
            raise ValueError("'shape_file' must be specified for 'from_shape' structure")
        
        voxel_size = self.config.get('voxel_size', 1.0)
        method = self.config.get('voxel_method', 'surface')
        materials = self.config.get('materials', [])
        
        if not materials:
            raise ValueError("'materials' list must be specified for DDA shape files")
        
        output_dir = self.config.get('output_dir', './results')
        
        if self.verbose:
            print(f"Loading DDA shape file...")
        
        loader = ShapeFileLoader(shape_file, voxel_size=voxel_size, method=method, verbose=self.verbose)
        loader.load()
        
        code = loader.generate_matlab_code(materials, output_dir=output_dir)
        
        return code
