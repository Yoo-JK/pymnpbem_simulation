"""
Geometry Cross-Section Calculator

Calculates 2D cross-sections of 3D nanoparticle structures at a given z-plane.
"""

import numpy as np


class GeometryCrossSection:
    """
    Calculates material boundary cross-sections for visualization.
    
    Given a z-plane coordinate, computes the 2D cross-section of the
    nanoparticle structure to overlay on field plots.
    """
    
    def __init__(self, config, verbose=False):
        """
        Initialize with simulation configuration.
        
        Parameters
        ----------
        config : dict
            Simulation configuration containing structure parameters
        verbose : bool
            Print debug information
        """
        self.config = config
        self.verbose = verbose
        self.structure_type = config.get('structure', 'sphere')
        
        if self.verbose:
            print(f"[GeometryCrossSection] Structure type: {self.structure_type}")
    
    def get_cross_section(self, z_plane):
        """
        Calculate cross-section at given z-plane.
        
        Parameters
        ----------
        z_plane : float
            Z-coordinate of the plane (in nm)
        
        Returns
        -------
        list of dict
            List of cross-section elements (circles or rectangles)
            Each dict contains:
                - 'type': 'circle' or 'rectangle'
                - 'center': [x, y] for circles
                - 'radius': radius for circles
                - 'bounds': [x_min, x_max, y_min, y_max] for rectangles
                - 'layer': layer index (0=core, 1=shell1, etc.)
                - 'label': description string
        """
        if self.verbose:
            print(f"\n[GeometryCrossSection] Calculating cross-section at z = {z_plane:.2f} nm")
        
        # Route to appropriate structure handler
        structure_handlers = {
            'sphere': self._sphere_cross_section,
            'core_shell': self._core_shell_cross_section,
            'core_shell_sphere': self._core_shell_cross_section,  # Alias
            'core_shell_cube': self._cube_cross_section,  # Simplified
            'core_shell_rod': self._rod_cross_section,  # Simplified
            'cube': self._cube_cross_section,
            'dimer': self._dimer_cross_section,
            'dimer_sphere': self._dimer_cross_section,  # Alias
            'dimer_cube': self._dimer_cube_cross_section,
            'dimer_core_shell_cube': self._dimer_cube_cross_section,  # Simplified
            'advanced_dimer_cube': self._dimer_cube_cross_section,  # Simplified
            'rod': self._rod_cross_section,
            'ellipsoid': self._ellipsoid_cross_section,
            'sphere_cluster': self._sphere_cluster_cross_section,
            'sphere_cluster_aggregate': self._sphere_cluster_cross_section,  # ✅ ADD THIS
            'triangle': self._triangle_cross_section,
            'from_shape': self._from_shape_cross_section,
        }
        
        handler = structure_handlers.get(self.structure_type)
        
        if handler is None:
            if self.verbose:
                print(f"  Warning: Unknown structure type '{self.structure_type}'")
                print(f"  Supported structures: {list(structure_handlers.keys())}")
            return []
        
        return handler(z_plane)
    
    # ========================================================================
    # Structure-specific cross-section calculators
    # ========================================================================
    
    def _sphere_cross_section(self, z_plane):
        """Calculate cross-section for single sphere."""
        radius = self.config.get('radius', 10.0)
        center = self.config.get('center', [0, 0, 0])
        
        z_center = center[2]
        z_diff = z_plane - z_center
        
        # Check if plane intersects sphere
        if abs(z_diff) > radius:
            if self.verbose:
                print(f"  No intersection: |z_diff| = {abs(z_diff):.2f} > radius = {radius:.2f}")
            return []
        
        # Calculate cross-section radius: r = sqrt(R² - z²)
        r_cross = np.sqrt(radius**2 - z_diff**2)
        
        if self.verbose:
            print(f"  Sphere intersection: r_cross = {r_cross:.2f} nm")
        
        return [{
            'type': 'circle',
            'center': [center[0], center[1]],
            'radius': r_cross,
            'layer': 0,
            'label': 'Sphere'
        }]
    
    def _core_shell_cross_section(self, z_plane):
        """Calculate cross-section for core-shell structure."""
        radii = self.config.get('radii', [10.0])
        center = self.config.get('center', [0, 0, 0])
        
        if not isinstance(radii, list):
            radii = [radii]
        
        z_center = center[2]
        z_diff = z_plane - z_center
        
        sections = []
        
        # Process each layer (from largest to smallest for proper overlay)
        for i, radius in enumerate(sorted(radii, reverse=True)):
            # Check if plane intersects this layer
            if abs(z_diff) > radius:
                continue
            
            # Calculate cross-section radius
            r_cross = np.sqrt(radius**2 - z_diff**2)
            
            # Determine layer label
            if len(radii) == 1:
                label = 'Sphere'
            elif i == len(radii) - 1:
                label = 'Core'
            else:
                shell_num = len(radii) - i - 1
                label = f'Shell {shell_num}'
            
            sections.append({
                'type': 'circle',
                'center': [center[0], center[1]],
                'radius': r_cross,
                'layer': len(radii) - i - 1,
                'label': label
            })
            
            if self.verbose:
                print(f"  Layer {i} ({label}): R = {radius:.2f} nm, r_cross = {r_cross:.2f} nm")
        
        # Reverse to draw from core to shell
        return sections[::-1]
    
    def _cube_cross_section(self, z_plane):
        """Calculate cross-section for cube."""
        side_length = self.config.get('side_length', 20.0)
        center = self.config.get('center', [0, 0, 0])
        
        # Calculate cube bounds
        half_size = side_length / 2
        x_min = center[0] - half_size
        x_max = center[0] + half_size
        y_min = center[1] - half_size
        y_max = center[1] + half_size
        z_min = center[2] - half_size
        z_max = center[2] + half_size
        
        # Check if plane intersects cube
        if z_plane < z_min or z_plane > z_max:
            if self.verbose:
                print(f"  No intersection: z_plane = {z_plane:.2f} outside [{z_min:.2f}, {z_max:.2f}]")
            return []
        
        if self.verbose:
            print(f"  Cube intersection: square from ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f})")
        
        return [{
            'type': 'rectangle',
            'bounds': [x_min, x_max, y_min, y_max],
            'layer': 0,
            'label': 'Cube'
        }]
    
    def _dimer_cross_section(self, z_plane):
        """Calculate cross-section for sphere dimer."""
        radius = self.config.get('radius', 10.0)
        gap = self.config.get('gap', 2.0)
        center = self.config.get('center', [0, 0, 0])
        dimer_axis = self.config.get('dimer_axis', 'x')
        
        # Calculate offset along dimer axis
        offset = radius + gap / 2
        
        # Calculate positions of two spheres
        if dimer_axis == 'x':
            pos1 = [center[0] - offset, center[1], center[2]]
            pos2 = [center[0] + offset, center[1], center[2]]
        elif dimer_axis == 'y':
            pos1 = [center[0], center[1] - offset, center[2]]
            pos2 = [center[0], center[1] + offset, center[2]]
        elif dimer_axis == 'z':
            pos1 = [center[0], center[1], center[2] - offset]
            pos2 = [center[0], center[1], center[2] + offset]
        else:
            if self.verbose:
                print(f"  Warning: Unknown dimer_axis '{dimer_axis}'")
            return []
        
        sections = []
        
        # Sphere 1
        z_diff1 = z_plane - pos1[2]
        if abs(z_diff1) <= radius:
            r_cross1 = np.sqrt(radius**2 - z_diff1**2)
            sections.append({
                'type': 'circle',
                'center': [pos1[0], pos1[1]],
                'radius': r_cross1,
                'layer': 0,
                'label': 'Sphere 1'
            })
            if self.verbose:
                print(f"  Sphere 1: center = ({pos1[0]:.1f}, {pos1[1]:.1f}), r_cross = {r_cross1:.2f} nm")
        
        # Sphere 2
        z_diff2 = z_plane - pos2[2]
        if abs(z_diff2) <= radius:
            r_cross2 = np.sqrt(radius**2 - z_diff2**2)
            sections.append({
                'type': 'circle',
                'center': [pos2[0], pos2[1]],
                'radius': r_cross2,
                'layer': 1,
                'label': 'Sphere 2'
            })
            if self.verbose:
                print(f"  Sphere 2: center = ({pos2[0]:.1f}, {pos2[1]:.1f}), r_cross = {r_cross2:.2f} nm")
        
        return sections
    
    def _dimer_cube_cross_section(self, z_plane):
        """Calculate cross-section for cube dimer."""
        side_length = self.config.get('side_length', 20.0)
        gap = self.config.get('gap', 2.0)
        center = self.config.get('center', [0, 0, 0])
        dimer_axis = self.config.get('dimer_axis', 'x')
        
        # Calculate offset along dimer axis
        offset = side_length / 2 + gap / 2
        
        # Calculate positions of two cubes
        if dimer_axis == 'x':
            pos1 = [center[0] - offset, center[1], center[2]]
            pos2 = [center[0] + offset, center[1], center[2]]
        elif dimer_axis == 'y':
            pos1 = [center[0], center[1] - offset, center[2]]
            pos2 = [center[0], center[1] + offset, center[2]]
        elif dimer_axis == 'z':
            pos1 = [center[0], center[1], center[2] - offset]
            pos2 = [center[0], center[1], center[2] + offset]
        else:
            if self.verbose:
                print(f"  Warning: Unknown dimer_axis '{dimer_axis}'")
            return []
        
        half_size = side_length / 2
        sections = []
        
        # Cube 1
        z_min1 = pos1[2] - half_size
        z_max1 = pos1[2] + half_size
        
        if z_min1 <= z_plane <= z_max1:
            x_min1 = pos1[0] - half_size
            x_max1 = pos1[0] + half_size
            y_min1 = pos1[1] - half_size
            y_max1 = pos1[1] + half_size
            
            sections.append({
                'type': 'rectangle',
                'bounds': [x_min1, x_max1, y_min1, y_max1],
                'layer': 0,
                'label': 'Cube 1'
            })
            if self.verbose:
                print(f"  Cube 1: bounds = ({x_min1:.1f}, {y_min1:.1f}) to ({x_max1:.1f}, {y_max1:.1f})")
        
        # Cube 2
        z_min2 = pos2[2] - half_size
        z_max2 = pos2[2] + half_size
        
        if z_min2 <= z_plane <= z_max2:
            x_min2 = pos2[0] - half_size
            x_max2 = pos2[0] + half_size
            y_min2 = pos2[1] - half_size
            y_max2 = pos2[1] + half_size
            
            sections.append({
                'type': 'rectangle',
                'bounds': [x_min2, x_max2, y_min2, y_max2],
                'layer': 1,
                'label': 'Cube 2'
            })
            if self.verbose:
                print(f"  Cube 2: bounds = ({x_min2:.1f}, {y_min2:.1f}) to ({x_max2:.1f}, {y_max2:.1f})")
        
        return sections
    
    def _rod_cross_section(self, z_plane):
        """Calculate cross-section for cylindrical rod."""
        radius = self.config.get('radius', 5.0)
        length = self.config.get('length', 40.0)
        center = self.config.get('center', [0, 0, 0])
        axis = self.config.get('axis', 'z')
        
        if axis == 'z':
            # Rod along z-axis: cross-section is circle
            z_min = center[2] - length / 2
            z_max = center[2] + length / 2
            
            if z_plane < z_min or z_plane > z_max:
                if self.verbose:
                    print(f"  No intersection: z_plane outside rod bounds")
                return []
            
            if self.verbose:
                print(f"  Rod (z-axis): circular cross-section with r = {radius:.2f} nm")
            
            return [{
                'type': 'circle',
                'center': [center[0], center[1]],
                'radius': radius,
                'layer': 0,
                'label': 'Rod'
            }]
        
        elif axis == 'x':
            # Rod along x-axis: cross-section is rectangle (if z_plane intersects)
            z_center = center[2]
            if abs(z_plane - z_center) > radius:
                if self.verbose:
                    print(f"  No intersection: |z_diff| > radius")
                return []
            
            x_min = center[0] - length / 2
            x_max = center[0] + length / 2
            
            # Calculate y-extent at this z
            z_diff = z_plane - z_center
            y_half = np.sqrt(radius**2 - z_diff**2)
            y_min = center[1] - y_half
            y_max = center[1] + y_half
            
            if self.verbose:
                print(f"  Rod (x-axis): rectangular cross-section")
            
            return [{
                'type': 'rectangle',
                'bounds': [x_min, x_max, y_min, y_max],
                'layer': 0,
                'label': 'Rod'
            }]
        
        elif axis == 'y':
            # Rod along y-axis: cross-section is rectangle
            z_center = center[2]
            if abs(z_plane - z_center) > radius:
                if self.verbose:
                    print(f"  No intersection: |z_diff| > radius")
                return []
            
            y_min = center[1] - length / 2
            y_max = center[1] + length / 2
            
            # Calculate x-extent at this z
            z_diff = z_plane - z_center
            x_half = np.sqrt(radius**2 - z_diff**2)
            x_min = center[0] - x_half
            x_max = center[0] + x_half
            
            if self.verbose:
                print(f"  Rod (y-axis): rectangular cross-section")
            
            return [{
                'type': 'rectangle',
                'bounds': [x_min, x_max, y_min, y_max],
                'layer': 0,
                'label': 'Rod'
            }]
        
        return []
    
    def _ellipsoid_cross_section(self, z_plane):
        """
        Calculate cross-section for ellipsoid.
        
        For simplicity, approximates as circle at z_plane.
        For accurate ellipse rendering, would need to track semi-major/minor axes.
        """
        semi_axes = self.config.get('semi_axes', [10.0, 10.0, 15.0])
        center = self.config.get('center', [0, 0, 0])
        
        a, b, c = semi_axes  # x, y, z semi-axes
        z_center = center[2]
        z_diff = z_plane - z_center
        
        # Check if plane intersects ellipsoid
        if abs(z_diff) > c:
            if self.verbose:
                print(f"  No intersection: |z_diff| = {abs(z_diff):.2f} > c = {c:.2f}")
            return []
        
        # Ellipse equation at z: (x/a')² + (y/b')² = 1
        # where a' = a * sqrt(1 - (z/c)²), b' = b * sqrt(1 - (z/c)²)
        scale = np.sqrt(1 - (z_diff / c)**2)
        a_cross = a * scale
        b_cross = b * scale
        
        # For visualization simplicity, use average radius (circle approximation)
        r_cross = (a_cross + b_cross) / 2
        
        if self.verbose:
            print(f"  Ellipsoid intersection: a' = {a_cross:.2f}, b' = {b_cross:.2f}, r_avg = {r_cross:.2f} nm")
            if abs(a_cross - b_cross) > 0.1:
                print(f"    Note: Approximating ellipse as circle")
        
        return [{
            'type': 'circle',
            'center': [center[0], center[1]],
            'radius': r_cross,
            'layer': 0,
            'label': 'Ellipsoid'
        }]
    
    def _sphere_cluster_cross_section(self, z_plane):
        """
        Calculate cross-section for sphere cluster (multiple spheres).
        
        FIXED: Now reads parameters directly from config (not cluster_config)
        and matches MATLAB's sphere position calculation.
        """
        # FIX: Read parameters directly from config (not nested in cluster_config)
        num_spheres = self.config.get('n_spheres', 1)
        diameter = self.config.get('diameter', 50.0)
        gap = self.config.get('gap', -0.1)
        
        # Calculate radius and center-to-center spacing
        radius = diameter / 2
        spacing = diameter + gap  # center-to-center spacing
        
        # Calculate sphere positions using MATLAB's formula
        positions = self._calculate_cluster_positions(num_spheres, spacing)
        
        sections = []
        
        for i, pos in enumerate(positions):
            z_diff = z_plane - pos[2]
            
            # Check if plane intersects this sphere
            if abs(z_diff) > radius:
                continue
            
            # Calculate cross-section radius
            r_cross = np.sqrt(radius**2 - z_diff**2)
            
            sections.append({
                'type': 'circle',
                'center': [pos[0], pos[1]],
                'radius': r_cross,
                'layer': i,
                'label': f'Sphere {i+1}'
            })
            
            if self.verbose:
                print(f"  Sphere {i+1}: center = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), r_cross = {r_cross:.2f} nm")
        
        return sections
    
    def _calculate_cluster_positions(self, num_spheres, spacing):
        """
        Calculate positions of spheres in a cluster.

        FIXED: Now matches MATLAB's position calculation exactly.

        Parameters
        ----------
        num_spheres : int
            Number of spheres (1-7)
        spacing : float
            Center-to-center spacing (diameter + gap)

        Returns
        -------
        list of [x, y, z]
            Positions of sphere centers (matching MATLAB geometry_generator.py)

        Structure layouts:
            N=1: Single sphere at origin
            N=2: Dimer (horizontal)
            N=3: Triangle (2 bottom, 1 top)
            N=4~7: Center + hexagonal surrounding (60° intervals)
        """
        # 60-degree triangle height (for triangular arrangements)
        dy_60deg = spacing * 0.866025404  # sin(60°) = sqrt(3)/2

        # Hexagonal surrounding positions (60° intervals, starting from +x direction)
        # Used for N=4~7: center + surrounding spheres
        # Matches MATLAB's geometry_generator.py exactly
        import math
        hex_positions = []
        for i in range(6):
            angle = i * 60 * math.pi / 180  # 0°, 60°, 120°, 180°, 240°, 300°
            x = spacing * math.cos(angle)
            y = spacing * math.sin(angle)
            hex_positions.append((x, y, 0))

        # ✅ FIX: These positions now EXACTLY match MATLAB's geometry_generator.py
        cluster_positions = {
            1: [(0, 0, 0)],

            2: [(-spacing/2, 0, 0),
                (spacing/2, 0, 0)],

            3: [(-spacing/2, 0, 0),         # bottom-left
                (spacing/2, 0, 0),          # bottom-right
                (0, dy_60deg, 0)],          # top

            # N=4~7: Center (0,0,0) + hexagonal surrounding positions
            4: [(0, 0, 0)] + hex_positions[0:3],  # center + 3 surrounding

            5: [(0, 0, 0)] + hex_positions[0:4],  # center + 4 surrounding

            6: [(0, 0, 0)] + hex_positions[0:5],  # center + 5 surrounding

            7: [(0, 0, 0)] + hex_positions[0:6],  # center + 6 surrounding (complete hexagon)
        }

        if num_spheres not in cluster_positions:
            raise ValueError(f"n_spheres must be 1-7, got {num_spheres}")

        positions = cluster_positions[num_spheres]

        # Convert to list of [x, y, z]
        return [[x, y, z] for x, y, z in positions]

    def _triangle_cross_section(self, z_plane):
        """Calculate cross-section for triangular prism."""
        side_length = self.config.get('side_length', 50.0)
        thickness = self.config.get('thickness', 10.0)
        center = self.config.get('center', [0, 0, 0])
        
        half_thick = thickness / 2
        z_min = center[2] - half_thick
        z_max = center[2] + half_thick
        
        # Check if plane intersects triangle
        if z_plane < z_min or z_plane > z_max:
            if self.verbose:
                print(f"  No intersection: z_plane outside triangle thickness")
            return []
        
        # For simplicity, approximate as circle with equivalent area
        # Triangle area = (sqrt(3)/4) * side^2
        # Circle area = pi * r^2
        # r = sqrt(area / pi) = sqrt((sqrt(3)/4) * side^2 / pi)
        area = (np.sqrt(3) / 4) * side_length**2
        r_equiv = np.sqrt(area / np.pi)
        
        if self.verbose:
            print(f"  Triangle: approximated as circle with r = {r_equiv:.2f} nm")
        
        return [{
            'type': 'circle',
            'center': [center[0], center[1]],
            'radius': r_equiv,
            'layer': 0,
            'label': 'Triangle'
        }]
    
    def _from_shape_cross_section(self, z_plane):
        """Calculate cross-section for DDA shape file import."""
        # For DDA shapes, we don't know the geometry in advance
        # Return empty list (no visualization overlay)
        if self.verbose:
            print(f"  DDA shape file: no cross-section visualization available")
        return []        
