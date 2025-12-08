"""
Field Analysis Utilities

Analyzes electromagnetic field distributions.
"""

import numpy as np
from scipy.ndimage import maximum_filter


class FieldAnalyzer:
    """Analyzes electromagnetic field data."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def analyze_field(self, field_data):
        """
        Comprehensive field analysis.
        
        Parameters
        ----------
        field_data : dict or list of dict
            Single field data dict or list of field data for multiple polarizations
        
        Returns
        -------
        dict or list of dict
            Analysis results (single dict if input is dict, list if input is list)
        """
        # ✅ FIX: Handle both single and multiple polarizations properly
        if isinstance(field_data, list):
            # Multiple polarizations: analyze each separately
            if self.verbose:
                print(f"  Analyzing {len(field_data)} polarizations...")
            
            return [self._analyze_single_field(pol_data) for pol_data in field_data]
        else:
            # Single polarization
            return self._analyze_single_field(field_data)
    
    def _analyze_single_field(self, pol_data):
        """
        Analyze a single polarization field data.
        
        Parameters
        ----------
        pol_data : dict
            Field data for a single polarization
        
        Returns
        -------
        dict
            Analysis results
        """
        # Extract data
        enhancement = pol_data.get('enhancement')
        intensity = pol_data.get('intensity')
        x_grid = pol_data.get('x_grid')
        y_grid = pol_data.get('y_grid')
        z_grid = pol_data.get('z_grid')
        
        if enhancement is None:
            return None

        # Handle complex data
        if np.iscomplexobj(enhancement):
            if self.verbose:
                print("  Converting complex enhancement to magnitude...")
            enhancement = np.abs(enhancement)

        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        # Extract polarization once to avoid multiple .get() calls
        polarization = pol_data.get('polarization')
        if hasattr(polarization, 'tolist'):
            polarization = polarization.tolist()

        analysis = {
            'wavelength': pol_data.get('wavelength'),
            'wavelength_idx': pol_data.get('wavelength_idx'),
            'polarization': polarization,
            'polarization_idx': pol_data.get('polarization_idx'),
        }

        # Enhancement statistics
        analysis['enhancement_stats'] = self._calculate_statistics(enhancement)

        # Intensity statistics (only if intensity data exists)
        if intensity is not None:
            analysis['intensity_stats'] = self._calculate_statistics(intensity)
        else:
            analysis['intensity_stats'] = None
        
        # Find hotspots
        hotspots = self._find_hotspots(enhancement, x_grid, y_grid, z_grid)
        analysis['hotspots'] = hotspots
        
        # Volume above threshold
        analysis['high_field_regions'] = self._analyze_high_field_regions(
            enhancement, x_grid, y_grid, z_grid
        )
        
        if self.verbose:
            self._print_analysis(analysis)
        
        return analysis
    
    def _calculate_statistics(self, data):
        """Calculate statistical measures of field data."""
        if not isinstance(data, np.ndarray):
            data = np.array([data])

        if data.ndim == 0:
            data = np.array([data.item()])
        
        data_flat = data.flatten()
        data_flat = data_flat[np.isfinite(data_flat)]  # Remove inf/nan
        
        if len(data_flat) == 0:
            return {
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'percentile_90': 0.0,
                'percentile_95': 0.0,
                'percentile_99': 0.0
            }
        
        stats = {
            'max': float(np.max(data_flat)),
            'min': float(np.min(data_flat)),
            'mean': float(np.mean(data_flat)),
            'median': float(np.median(data_flat)),
            'std': float(np.std(data_flat)),
            'percentile_90': float(np.percentile(data_flat, 90)),
            'percentile_95': float(np.percentile(data_flat, 95)),
            'percentile_99': float(np.percentile(data_flat, 99))
        }
        
        return stats
    
    def _find_hotspots(self, enhancement, x_grid, y_grid, z_grid, 
                       num_hotspots=10, min_distance=3):
        """
        Find local maximum enhancement hotspots.
        """
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([enhancement])
        
        if enhancement.ndim == 0:
            enhancement = np.array([enhancement.item()])
        
        if enhancement.size == 1:
            hotspot = {
                'rank': 1,
                'position': [float(x_grid), float(y_grid), float(z_grid)],
                'enhancement': float(enhancement.item()),
                'intensity_enhancement': float(enhancement.item()**2)
            }
            return [hotspot]

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)
            if self.verbose:
                print("  Converting complex enhancement to magnitude")

        neighborhood_size = min_distance * 2 + 1
        local_max = maximum_filter(enhancement, size=neighborhood_size)
        
        # Points that are equal to local max are local maxima
        is_local_max = (enhancement == local_max)
        
        # Also require enhancement > 1 (at least some enhancement)
        is_local_max = is_local_max & (enhancement > 1.0)
        
        # Get indices of local maxima
        max_indices = np.where(is_local_max)
        max_values = enhancement[max_indices]
        
        # Sort by enhancement value (descending)
        sorted_idx = np.argsort(max_values)[::-1]
        
        # Take top N hotspots
        hotspots = []
        for i in range(min(num_hotspots, len(sorted_idx))):
            idx = sorted_idx[i]
            
            # Get position indices
            if enhancement.ndim == 2:
                row_idx, col_idx = max_indices[0][idx], max_indices[1][idx]
                x_pos = float(x_grid[row_idx, col_idx])
                y_pos = float(y_grid[row_idx, col_idx])
                z_pos = float(z_grid[row_idx, col_idx])
            else:  # 3D
                i_idx, j_idx, k_idx = max_indices[0][idx], max_indices[1][idx], max_indices[2][idx]
                x_pos = float(x_grid[i_idx, j_idx, k_idx])
                y_pos = float(y_grid[i_idx, j_idx, k_idx])
                z_pos = float(z_grid[i_idx, j_idx, k_idx])
            
            hotspot = {
                'rank': i + 1,
                'position': [x_pos, y_pos, z_pos],
                'enhancement': float(max_values[idx]),
                'intensity_enhancement': float(max_values[idx]**2)
            }
            hotspots.append(hotspot)
        
        return hotspots
    
    def _analyze_high_field_regions(self, enhancement, x_grid, y_grid, z_grid):
        """
        Analyze regions with high field enhancement.
        
        Returns volume/area above various enhancement thresholds.
        """
        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([enhancement])
        
        if enhancement.ndim == 0:
            enhancement = np.array([enhancement.item()])
        
        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([x_grid])
            y_grid = np.array([y_grid])
            z_grid = np.array([z_grid])
        
        if enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, 1)
            x_grid = x_grid.reshape(1, 1) if x_grid.ndim == 1 else x_grid
            y_grid = y_grid.reshape(1, 1) if y_grid.ndim == 1 else y_grid
            z_grid = z_grid.reshape(1, 1) if z_grid.ndim == 1 else z_grid

        # Calculate grid spacing (assuming uniform)
        if enhancement.ndim == 2:
            dx = np.abs(x_grid[0, 1] - x_grid[0, 0]) if x_grid.shape[1] > 1 else 1.0
            dy = np.abs(y_grid[1, 0] - y_grid[0, 0]) if y_grid.shape[0] > 1 else 1.0
            dz = 0
            element_area = dx * dy if dx > 0 and dy > 0 else 1.0
            is_3d = False
        else:  # 3D
            dx = np.abs(x_grid[0, 0, 1] - x_grid[0, 0, 0]) if x_grid.shape[2] > 1 else 1.0
            dy = np.abs(y_grid[0, 1, 0] - y_grid[0, 0, 0]) if y_grid.shape[1] > 1 else 1.0
            dz = np.abs(z_grid[1, 0, 0] - z_grid[0, 0, 0]) if z_grid.shape[0] > 1 else 1.0
            element_volume = dx * dy * dz if dx > 0 and dy > 0 and dz > 0 else 1.0
            is_3d = True
        
        thresholds = [2, 5, 10, 20, 50, 100]
        regions = {}
        
        for threshold in thresholds:
            mask = enhancement > threshold
            count = np.sum(mask)
            
            if is_3d:
                volume = count * element_volume
                regions[f'enhancement_above_{threshold}'] = {
                    'num_points': int(count),
                    'volume_nm3': float(volume)
                }
            else:
                area = count * element_area
                regions[f'enhancement_above_{threshold}'] = {
                    'num_points': int(count),
                    'area_nm2': float(area)
                }
        
        return regions
    
    def _print_analysis(self, analysis):
        """Print field analysis summary."""
        pol_idx = analysis.get('polarization_idx', '?')
        wl_idx = analysis.get('wavelength_idx', '?')
        print(f"\n  Field Analysis (λ = {analysis['wavelength']:.1f} nm, wl_idx={wl_idx}, pol_idx={pol_idx}):")
        print("  " + "-"*50)
        
        # Enhancement statistics
        stats = analysis['enhancement_stats']
        print(f"  Enhancement Statistics:")
        print(f"    Max:       {stats['max']:.2f}")
        print(f"    Mean:      {stats['mean']:.2f}")
        print(f"    Median:    {stats['median']:.2f}")
        print(f"    95th %ile: {stats['percentile_95']:.2f}")
        
        # Top hotspots
        if analysis['hotspots']:
            print(f"\n  Top {len(analysis['hotspots'])} Hotspots:")
            for hotspot in analysis['hotspots'][:5]:  # Print top 5
                pos = hotspot['position']
                print(f"    #{hotspot['rank']}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) nm "
                      f"| E/E₀ = {hotspot['enhancement']:.2f}")
