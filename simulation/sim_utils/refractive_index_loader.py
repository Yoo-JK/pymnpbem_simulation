"""
Refractive Index Loader

Loads refractive index data from files and provides interpolation.
Supports various file formats with wavelength-dependent n and k values.
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline  # ← 추가


class RefractiveIndexLoader:
    """Load and interpolate refractive index data from files."""
    
    def __init__(self, filepath, verbose=False):
        """
        Initialize refractive index loader.
        
        Args:
            filepath (str or Path): Path to refractive index data file
            verbose (bool): Enable verbose output
        """
        self.filepath = Path(filepath)
        self.verbose = verbose
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Refractive index file not found: {filepath}")
        
        # Load data
        self.wavelengths, self.n_values, self.k_values = self._load_file()
        
        if self.verbose:
            print(f"Loaded refractive index data from: {self.filepath}")
            print(f"  Wavelength range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm")
            print(f"  Number of points: {len(self.wavelengths)}")
    
    def _load_file(self):
        """
        Load refractive index data from file.
        
        Supported formats:
        1. Two columns: wavelength(nm), n
        2. Three columns: wavelength(nm), n, k
        
        Returns:
            tuple: (wavelengths, n_values, k_values)
        """
        try:
            # Try loading with numpy (handles comments automatically)
            data = np.loadtxt(self.filepath, comments='#')
            
            if data.ndim == 1:
                # Single row - reshape
                data = data.reshape(1, -1)
            
            n_cols = data.shape[1]
            
            if n_cols == 2:
                # Format: wavelength, n
                wavelengths = data[:, 0]
                n_values = data[:, 1]
                k_values = np.zeros_like(n_values)  # No absorption
                
                if self.verbose:
                    print("  Format: wavelength(nm), n (no absorption)")
            
            elif n_cols == 3:
                # Format: wavelength, n, k
                wavelengths = data[:, 0]
                n_values = data[:, 1]
                k_values = data[:, 2]
                
                if self.verbose:
                    print("  Format: wavelength(nm), n, k")
            
            else:
                raise ValueError(f"Unsupported file format: {n_cols} columns found. Expected 2 or 3.")
            
            # Sort by wavelength (required for interpolation)
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            n_values = n_values[sort_idx]
            k_values = k_values[sort_idx]
            
            return wavelengths, n_values, k_values
        
        except Exception as e:
            raise RuntimeError(f"Error loading refractive index file: {e}")
    
    def interpolate(self, target_wavelengths):
        """
        Interpolate refractive index at target wavelengths using cubic spline interpolation.
        
        Args:
            target_wavelengths (array-like): Target wavelengths in nm
        
        Returns:
            tuple: (n_interpolated, k_interpolated)
        """
        target_wavelengths = np.asarray(target_wavelengths)
        
        # ═══════════════════════════════════════════════════════════
        # ✅ 수정: Linear → Cubic Spline 보간
        # ═══════════════════════════════════════════════════════════
        
        # Cubic spline interpolation for n (MATLAB과 동일)
        cs_n = CubicSpline(self.wavelengths, self.n_values)
        n_interp = cs_n(target_wavelengths)
        
        # Cubic spline interpolation for k
        cs_k = CubicSpline(self.wavelengths, self.k_values)
        k_interp = cs_k(target_wavelengths)
        
        if self.verbose:
            print(f"Interpolated refractive index at {len(target_wavelengths)} wavelengths")
            print("  Interpolation method: Cubic Spline")
            # Check if extrapolation occurred
            if target_wavelengths.min() < self.wavelengths.min() or \
               target_wavelengths.max() > self.wavelengths.max():
                print("  WARNING: Some wavelengths outside data range (extrapolation used)")
        
        return n_interp, k_interp
    
    def get_epsilon(self, target_wavelengths):
        """
        Get complex dielectric function epsilon = (n + ik)^2 at target wavelengths.
        
        Args:
            target_wavelengths (array-like): Target wavelengths in nm
        
        Returns:
            array: Complex epsilon values
        """
        n_interp, k_interp = self.interpolate(target_wavelengths)
        
        # epsilon = (n + ik)^2
        refractive_index = n_interp + 1j * k_interp
        epsilon = refractive_index ** 2
        
        return epsilon


def load_and_interpolate(filepath, target_wavelengths, verbose=False):
    """
    Convenience function to load and interpolate refractive index data.
    
    Args:
        filepath (str or Path): Path to refractive index data file
        target_wavelengths (array-like): Target wavelengths in nm
        verbose (bool): Enable verbose output
    
    Returns:
        tuple: (n_interpolated, k_interpolated, epsilon_complex)
    """
    loader = RefractiveIndexLoader(filepath, verbose=verbose)
    n_interp, k_interp = loader.interpolate(target_wavelengths)
    epsilon = loader.get_epsilon(target_wavelengths)
    
    return n_interp, k_interp, epsilon