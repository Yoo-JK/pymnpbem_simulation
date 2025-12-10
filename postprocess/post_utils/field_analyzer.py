"""
Field Analyzer for pyMNPBEM simulation results.

Provides analysis of electric field distributions:
- Hotspot detection
- Enhancement statistics
- Volume integration
"""

import numpy as np
from scipy.ndimage import maximum_filter, label
from typing import Dict, List, Any, Optional


class FieldAnalyzer:
    """
    Analyzes electric field distributions from plasmonic simulations.

    Features:
    - Hotspot detection and characterization
    - Enhancement statistics
    - Volume integration for field enhancement
    """

    def __init__(self, field_data: Dict[str, np.ndarray]):
        """
        Initialize the field analyzer.

        Args:
            field_data: Dictionary with field enhancement data
        """
        self.enhancement = field_data.get('enhancement')
        self.x = field_data.get('x')
        self.y = field_data.get('y')
        self.z = field_data.get('z')
        self.X = field_data.get('X')
        self.Y = field_data.get('Y')
        self.Z = field_data.get('Z')

    def find_hotspots(self, n_hotspots: int = 10,
                      min_distance: int = 3,
                      threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find field hotspot locations.

        Args:
            n_hotspots: Maximum number of hotspots to find
            min_distance: Minimum grid-point distance between hotspots
            threshold: Minimum fraction of max value for hotspot

        Returns:
            List of hotspot dictionaries
        """
        if self.enhancement is None:
            return []

        enhancement = self.enhancement
        max_val = np.max(enhancement)
        threshold_val = max_val * threshold

        # Find local maxima
        local_max = maximum_filter(enhancement, size=min_distance)
        is_max = (enhancement == local_max) & (enhancement > threshold_val)

        # Get positions of maxima
        max_positions = np.where(is_max)
        max_values = enhancement[is_max]

        # Sort by value
        sort_idx = np.argsort(max_values)[::-1]

        hotspots = []
        for i in range(min(n_hotspots, len(sort_idx))):
            idx = sort_idx[i]

            # Get grid indices and coordinates
            if enhancement.ndim == 2:
                gi, gj = max_positions[0][idx], max_positions[1][idx]
                x = float(self.X[gi, gj]) if self.X is not None else float(gi)
                y = float(self.Y[gi, gj]) if self.Y is not None else float(gj)
                z = float(self.Z[gi, gj]) if self.Z is not None else 0.0
                grid_idx = [int(gi), int(gj)]
            else:
                gi, gj, gk = max_positions[0][idx], max_positions[1][idx], max_positions[2][idx]
                x = float(self.X[gi, gj, gk]) if self.X is not None else float(gi)
                y = float(self.Y[gi, gj, gk]) if self.Y is not None else float(gj)
                z = float(self.Z[gi, gj, gk]) if self.Z is not None else float(gk)
                grid_idx = [int(gi), int(gj), int(gk)]

            hotspots.append({
                'position': [x, y, z],
                'enhancement': float(max_values[idx]),
                'grid_index': grid_idx,
                'rank': i + 1
            })

        return hotspots

    def get_statistics(self, exclude_inside: bool = True) -> Dict[str, float]:
        """
        Compute statistics of field enhancement.

        Args:
            exclude_inside: Whether to exclude points inside particle (zero values)

        Returns:
            Dictionary with statistics
        """
        if self.enhancement is None:
            return {}

        enhancement = self.enhancement

        if exclude_inside:
            mask = enhancement > 0
            if not np.any(mask):
                return {
                    'max': 0.0,
                    'min': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'median': 0.0,
                    'p90': 0.0,
                    'p99': 0.0,
                }
            valid = enhancement[mask]
        else:
            valid = enhancement.flatten()

        return {
            'max': float(np.max(valid)),
            'min': float(np.min(valid)),
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid)),
            'median': float(np.median(valid)),
            'p90': float(np.percentile(valid, 90)),
            'p99': float(np.percentile(valid, 99)),
        }

    def compute_enhanced_volume(self, thresholds: List[float] = [10, 100, 1000]) -> Dict[float, float]:
        """
        Compute volume where enhancement exceeds given thresholds.

        Args:
            thresholds: List of enhancement thresholds

        Returns:
            Dictionary mapping threshold to volume (in nm^3)
        """
        if self.enhancement is None or self.x is None:
            return {t: 0.0 for t in thresholds}

        # Calculate voxel volume
        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        dy = self.y[1] - self.y[0] if len(self.y) > 1 else 1.0
        dz = self.z[1] - self.z[0] if len(self.z) > 1 else 1.0
        voxel_volume = abs(dx * dy * dz)

        volumes = {}
        for threshold in thresholds:
            n_voxels = np.sum(self.enhancement > threshold)
            volumes[threshold] = float(n_voxels * voxel_volume)

        return volumes

    def compute_integral_enhancement(self) -> float:
        """
        Compute integrated field enhancement over the computation volume.

        Returns:
            Integrated enhancement (enhancement * volume)
        """
        if self.enhancement is None or self.x is None:
            return 0.0

        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        dy = self.y[1] - self.y[0] if len(self.y) > 1 else 1.0
        dz = self.z[1] - self.z[0] if len(self.z) > 1 else 1.0
        voxel_volume = abs(dx * dy * dz)

        # Exclude inside particle
        mask = self.enhancement > 0
        integral = np.sum(self.enhancement[mask]) * voxel_volume

        return float(integral)

    def find_gap_hotspot(self, particle_bounds: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Find the hotspot in the gap region (for dimers).

        Args:
            particle_bounds: Optional bounds of individual particles

        Returns:
            Hotspot information or None
        """
        if self.enhancement is None:
            return None

        # Simple heuristic: find maximum near x=0 for face-to-face dimers
        if self.X is not None:
            # Create mask for gap region (near x=0)
            gap_mask = np.abs(self.X) < np.max(np.abs(self.x)) * 0.3
            gap_enhancement = np.where(gap_mask, self.enhancement, 0)

            max_idx = np.unravel_index(np.argmax(gap_enhancement), gap_enhancement.shape)

            if self.enhancement.ndim == 2:
                x = float(self.X[max_idx])
                y = float(self.Y[max_idx])
                z = float(self.Z[max_idx])
            else:
                x = float(self.X[max_idx])
                y = float(self.Y[max_idx])
                z = float(self.Z[max_idx])

            return {
                'position': [x, y, z],
                'enhancement': float(self.enhancement[max_idx]),
                'grid_index': list(max_idx),
            }

        return None

    def get_enhancement_profile(self, axis: str = 'x',
                                 position: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Extract enhancement profile along an axis.

        Args:
            axis: 'x', 'y', or 'z'
            position: Position along the other axes

        Returns:
            Dictionary with coordinates and enhancement values
        """
        if self.enhancement is None:
            return {}

        if axis == 'x' and len(self.y) == 1:
            # XZ plane, profile along x at given z
            z_idx = np.argmin(np.abs(self.z - position))
            return {
                'coordinate': self.x,
                'enhancement': self.enhancement[:, z_idx] if self.enhancement.ndim == 2 else self.enhancement[:, 0, z_idx],
                'axis': 'x',
                'position': float(self.z[z_idx])
            }
        elif axis == 'z' and len(self.y) == 1:
            # XZ plane, profile along z at given x
            x_idx = np.argmin(np.abs(self.x - position))
            return {
                'coordinate': self.z,
                'enhancement': self.enhancement[x_idx, :] if self.enhancement.ndim == 2 else self.enhancement[x_idx, 0, :],
                'axis': 'z',
                'position': float(self.x[x_idx])
            }

        return {}
