"""
Nonlocal Generator

Generates quantum correction layers for sub-nanometer gaps.
Implements the Luo et al. (PRL 111, 093901, 2013) model.
"""

import numpy as np


class NonlocalGenerator:
    """Generates nonlocal quantum correction layers for plasmonic simulations."""
    
    def __init__(self, config, verbose=False):
        """
        Initialize nonlocal generator.
        
        Args:
            config (dict): Configuration dictionary
            verbose (bool): Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        
        # Nonlocal parameters
        self.enabled = config.get('use_nonlocality', False)
        self.cover_thickness = config.get('nonlocal_cover_thickness', 0.05)  # nm
        
        # Drude model parameters for gold (default)
        # Can be overridden via config
        self.omega_p = config.get('nonlocal_omega_p', 3.3)  # eV
        self.gamma = config.get('nonlocal_gamma', 0.165)  # eV  
        self.beta = config.get('nonlocal_beta', 0.0036)  # eV·nm
        self.eps_inf = config.get('nonlocal_eps_inf', 1.0)
        
        if verbose and self.enabled:
            print(f"\n✓ Nonlocal corrections enabled:")
            print(f"  - Cover layer thickness: {self.cover_thickness} nm")
            print(f"  - Drude parameters: ωp={self.omega_p} eV, γ={self.gamma} eV")
    
    def is_needed(self):
        """Check if nonlocal corrections are needed."""
        if not self.enabled:
            return False
        
        # Check gap size
        gap = self.config.get('gap', float('inf'))
        if gap >= 1.0:
            if self.verbose:
                print(f"\n⚠ Warning: Gap = {gap} nm is large. Nonlocal effects may be negligible.")
        
        return True
    
    def generate_artificial_epsilon(self, material_name='gold'):
        """
        Generate artificial dielectric function for nonlocal corrections.
        
        Based on Luo et al., PRL 111, 093901 (2013).
        
        Args:
            material_name (str): Name of the metal ('gold' or 'silver')
        
        Returns:
            str: MATLAB code for artificial epsilon
        """
        if not self.enabled:
            return ""
        
        # Get material-specific parameters
        if material_name.lower() in ['gold', 'au']:
            omega_p = 3.3  # eV
            gamma = 0.165  # eV
            beta = 0.0036  # eV·nm
            eps_inf = 1.0
        elif material_name.lower() in ['silver', 'ag']:
            omega_p = 3.8  # eV
            gamma = 0.048  # eV
            beta = 0.0036  # eV·nm
            eps_inf = 1.0
        else:
            # Use provided or default values
            omega_p = self.omega_p
            gamma = self.gamma
            beta = self.beta
            eps_inf = self.eps_inf
        
        d = self.cover_thickness
        
        code = f"""
%% Nonlocal Artificial Dielectric Function ({material_name})
% Luo et al., PRL 111, 093901 (2013) model
d_nonlocal = {d};  % Cover layer thickness (nm)

% Drude model parameters
omega_p = {omega_p};  % Plasma frequency (eV)
gamma_drude = {gamma};  % Damping rate (eV)
beta_eff = {beta};  % Effective velocity (eV·nm)
eps_infinity = {eps_inf};  % Background permittivity

% Longitudinal plasmon wavenumber
units;  % Load MNPBEM unit conversion
ql = @( w ) 2 * pi * sqrt( omega_p^2 / eps_infinity - w .* ( w + 1i * gamma_drude ) ) / beta_eff;

% Local Drude dielectric function for {material_name}
eps_{material_name}_drude = epsfun( @( w ) eps_infinity - omega_p^2 ./ ( w .* ( w + 1i * gamma_drude ) ), 'eV' );

% Artificial nonlocal permittivity
eps_{material_name}_nonlocal = epsfun( @( enei ) eps_{material_name}_drude( enei ) .* eps_mat1( enei ) ./  ...
            ( eps_{material_name}_drude( enei ) - eps_mat1( enei ) ) .* ql( eV2nm ./ enei ) * d_nonlocal );

fprintf('  [OK] Nonlocal corrections: {material_name} with d=%.3f nm\\n', d_nonlocal);
"""
        return code
    
    def generate_coverlayer_code(self, particle_name, layer_index=1):
        """
        Generate MATLAB code to create cover layer on a particle.
        
        Args:
            particle_name (str): Name of the particle variable
            layer_index (int): Layer number for naming
        
        Returns:
            str: MATLAB code for cover layer generation
        """
        if not self.enabled:
            return ""
        
        d = self.cover_thickness
        
        code = f"""
% Generate nonlocal cover layer for {particle_name}
{particle_name}_inner = {particle_name};  % Inner boundary (original)
{particle_name}_outer = coverlayer.shift( {particle_name}_inner, {d} );  % Outer boundary (shifted)

% Replace original particle with cover layer structure
{particle_name} = {particle_name}_outer;
{particle_name}_coverlayer = {particle_name}_inner;
"""
        return code
    
    def modify_materials_for_nonlocal(self, materials):
        """
        Modify material list to include nonlocal corrections.
        
        For each metal, we need:
          - Original metal → inner boundary
          - Nonlocal artificial epsilon → outer boundary (cover layer)
        
        Args:
            materials (list): Original material list
        
        Returns:
            tuple: (modified_materials, nonlocal_mapping)
                modified_materials: New material list with nonlocal materials
                nonlocal_mapping: Dict mapping original index to (inner, outer) indices
        """
        if not self.enabled:
            return materials, {}
        
        metals = ['gold', 'silver', 'au', 'ag', 'aluminum', 'al', 'copper', 'cu']
        
        modified_materials = []
        nonlocal_mapping = {}
        
        for i, mat in enumerate(materials):
            mat_name = mat.lower() if isinstance(mat, str) else 'unknown'
            
            # Check if this is a metal that needs nonlocal correction
            is_metal = any(metal in mat_name for metal in metals)
            
            if is_metal:
                # Add both inner (Drude) and outer (nonlocal) materials
                modified_materials.append(mat)  # Inner: Drude
                modified_materials.append(f"{mat}_nonlocal")  # Outer: nonlocal
                
                inner_idx = len(modified_materials) - 2
                outer_idx = len(modified_materials) - 1
                nonlocal_mapping[i] = (inner_idx, outer_idx)
                
                if self.verbose:
                    print(f"  - {mat}: inner (Drude) index {inner_idx}, outer (nonlocal) index {outer_idx}")
            else:
                # Non-metal: keep as is
                modified_materials.append(mat)
        
        return modified_materials, nonlocal_mapping
    
    def generate_bem_options(self):
        """
        Generate BEM options for nonlocal simulations.
        
        Requires high-precision integration for close elements.
        
        Returns:
            str: MATLAB code for BEM options
        """
        if not self.enabled:
            return ""
        
        npol = self.config.get('npol', 20)
        refine = self.config.get('refine', 3)
        
        code = f"""
%% Enhanced BEM Options for Nonlocal Simulations
% High-precision integration for close boundary elements
op = bemoptions( op, 'npol', {npol}, 'refine', {refine} );
fprintf('  [OK] Nonlocal BEM options: npol=%d, refine=%d\\n', {npol}, {refine});
"""
        return code
    
    def generate_refine_function(self, particle_var='p', cover_pairs=None):
        """
        Generate refined integration function for cover layer boundaries.
        
        Args:
            particle_var (str): Name of the comparticle variable
            cover_pairs (list): List of [inner, outer] boundary pairs
                               If None, auto-detect from config
        
        Returns:
            str: MATLAB code for refined integration
        """
        if not self.enabled:
            return ""
        
        if cover_pairs is None:
            # Auto-detect from structure
            # For dimer: [[1,2], [3,4]] typical
            # For single: [[1,2]]
            structure = self.config.get('structure', '')
            if 'dimer' in structure:
                cover_pairs = [[1, 2], [3, 4]]
            else:
                cover_pairs = [[1, 2]]
        
        # Convert to MATLAB array format
        pairs_str = "; ".join([f"{p[0]}, {p[1]}" for p in cover_pairs])
        
        code = f"""
% Refined integration for cover layer boundaries
refun = coverlayer.refine( {particle_var}, [ {pairs_str} ] );
fprintf('  [OK] Cover layer refinement: %d boundary pairs\\n', {len(cover_pairs)});
"""
        return code
    
    def check_applicability(self):
        """
        Check if nonlocal corrections are applicable and warn if not.
        
        Returns:
            tuple: (is_applicable, warnings)
        """
        warnings = []
        
        if not self.enabled:
            return True, []
        
        # Check gap size
        gap = self.config.get('gap', None)
        if gap is not None:
            if gap > 1.0:
                warnings.append(f"Gap = {gap} nm is large. Nonlocal effects may be negligible.")
            elif gap < 0.1:
                warnings.append(f"Gap = {gap} nm is extremely small. Atomic-scale effects may dominate.")
        
        # Check particle size
        structure = self.config.get('structure', '')
        if 'sphere' in structure:
            diameter = self.config.get('diameter', self.config.get('core_diameter', 50))
            if diameter > 20:
                warnings.append(f"Particle diameter = {diameter} nm. Nonlocal effects are mainly surface phenomena.")
        
        # Check mesh density
        mesh_density = self.config.get('mesh_density', 12)
        if 'cube' in structure and mesh_density < 20:
            warnings.append(f"Mesh density = {mesh_density} may be insufficient for gap < 1 nm. Consider mesh_density ≥ 25.")
        
        return len(warnings) == 0, warnings


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_cover_thickness(gap, material='gold'):
    """
    Estimate appropriate cover layer thickness based on gap size.
    
    Rule of thumb: d ~ gap / 10, but clamped to [0.02, 0.1] nm
    
    Args:
        gap (float): Gap size in nm
        material (str): Material name
    
    Returns:
        float: Recommended cover thickness in nm
    """
    d = max(0.02, min(0.1, gap / 10))
    return d


def get_drude_parameters(material):
    """
    Get Drude model parameters for common metals.
    
    Args:
        material (str): Material name
    
    Returns:
        dict: Drude parameters (omega_p, gamma, beta, eps_inf)
    """
    params = {
        'gold': {'omega_p': 3.3, 'gamma': 0.165, 'beta': 0.0036, 'eps_inf': 1.0},
        'au': {'omega_p': 3.3, 'gamma': 0.165, 'beta': 0.0036, 'eps_inf': 1.0},
        'silver': {'omega_p': 3.8, 'gamma': 0.048, 'beta': 0.0036, 'eps_inf': 1.0},
        'ag': {'omega_p': 3.8, 'gamma': 0.048, 'beta': 0.0036, 'eps_inf': 1.0},
        'aluminum': {'omega_p': 4.8, 'gamma': 0.14, 'beta': 0.0036, 'eps_inf': 1.0},
        'al': {'omega_p': 4.8, 'gamma': 0.14, 'beta': 0.0036, 'eps_inf': 1.0},
    }
    
    mat_lower = material.lower()
    return params.get(mat_lower, params['gold'])  # Default to gold