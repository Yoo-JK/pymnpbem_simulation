import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import maximum_filter
from .edge_filter import find_edge_artifacts


class FieldAnalyzer(object):

    def __init__(self,
            verbose: bool = False) -> None:

        self.verbose = verbose
        self.near_field_distances = [2.0, 15.0]

    def analyze_field(self,
            field_data: Union[Dict, List[Dict]]) -> Union[Optional[Dict], List[Optional[Dict]]]:

        if isinstance(field_data, list):
            if self.verbose:
                print('  Analyzing {} polarizations...'.format(len(field_data)))

            return [self._analyze_single_field(pol_data) for pol_data in field_data]
        else:
            return self._analyze_single_field(field_data)

    def _analyze_single_field(self,
            pol_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        enhancement = pol_data.get('enhancement')
        intensity = pol_data.get('intensity')
        x_grid = pol_data.get('x_grid')
        y_grid = pol_data.get('y_grid')
        z_grid = pol_data.get('z_grid')

        if enhancement is None:
            return None

        if np.iscomplexobj(enhancement):
            if self.verbose:
                print('  Converting complex enhancement to magnitude...')
            enhancement = np.abs(enhancement)

        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)

        polarization = pol_data.get('polarization')
        if hasattr(polarization, 'tolist'):
            polarization = polarization.tolist()

        analysis = {
            'wavelength': pol_data.get('wavelength'),
            'wavelength_idx': pol_data.get('wavelength_idx'),
            'polarization': polarization,
            'polarization_idx': pol_data.get('polarization_idx'),
        }

        analysis['enhancement_stats'] = self._calculate_statistics(enhancement)

        if intensity is not None:
            analysis['intensity_stats'] = self._calculate_statistics(intensity)
        else:
            analysis['intensity_stats'] = None

        hotspots = self._find_hotspots(enhancement, x_grid, y_grid, z_grid)
        analysis['hotspots'] = hotspots

        analysis['high_field_regions'] = self._analyze_high_field_regions(
            enhancement, x_grid, y_grid, z_grid
        )

        if self.verbose:
            self._print_analysis(analysis)

        return analysis

    def _calculate_statistics(self,
            data: Any) -> Dict[str, float]:

        if not isinstance(data, np.ndarray):
            data = np.array([data])

        if data.ndim == 0:
            data = np.array([data.item()])

        data_flat = data.flatten()
        data_flat = data_flat[np.isfinite(data_flat)]

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

    def _find_hotspots(self,
            enhancement: Any,
            x_grid: Any,
            y_grid: Any,
            z_grid: Any,
            num_hotspots: int = 10,
            min_distance: int = 3) -> List[Dict[str, Any]]:

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
                print('  Converting complex enhancement to magnitude')

        neighborhood_size = min_distance * 2 + 1
        local_max = maximum_filter(enhancement, size = neighborhood_size)

        is_local_max = (enhancement == local_max)
        is_local_max = is_local_max & (enhancement > 1.0)

        max_indices = np.where(is_local_max)
        max_values = enhancement[max_indices]

        sorted_idx = np.argsort(max_values)[::-1]

        hotspots = []
        for i in range(min(num_hotspots, len(sorted_idx))):
            idx = sorted_idx[i]

            ndim = enhancement.ndim
            n_indices = len(max_indices)

            if ndim == 1 or n_indices == 1:
                flat_idx = max_indices[0][idx]
                x_pos = float(x_grid.flat[flat_idx]) if hasattr(x_grid, 'flat') else float(x_grid)
                y_pos = float(y_grid.flat[flat_idx]) if hasattr(y_grid, 'flat') else float(y_grid)
                z_pos = float(z_grid.flat[flat_idx]) if hasattr(z_grid, 'flat') else float(z_grid)
            elif ndim == 2 or n_indices == 2:
                idx_0, idx_1 = max_indices[0][idx], max_indices[1][idx]
                x_pos = float(x_grid[idx_0, idx_1]) if x_grid.ndim >= 2 else float(x_grid.flat[idx_0])
                y_pos = float(y_grid[idx_0, idx_1]) if y_grid.ndim >= 2 else float(y_grid.flat[idx_0])
                z_pos = float(z_grid[idx_0, idx_1]) if z_grid.ndim >= 2 else float(z_grid.flat[idx_0])
            else:
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

    def _analyze_high_field_regions(self,
            enhancement: Any,
            x_grid: Any,
            y_grid: Any,
            z_grid: Any) -> Dict[str, Dict[str, Any]]:

        if not isinstance(enhancement, np.ndarray):
            enhancement = np.array([enhancement])

        if enhancement.ndim == 0:
            enhancement = np.array([enhancement.item()])

        if not isinstance(x_grid, np.ndarray):
            x_grid = np.array([x_grid])
        if not isinstance(y_grid, np.ndarray):
            y_grid = np.array([y_grid])
        if not isinstance(z_grid, np.ndarray):
            z_grid = np.array([z_grid])

        if x_grid.ndim == 0:
            x_grid = np.array([x_grid.item()])
        if y_grid.ndim == 0:
            y_grid = np.array([y_grid.item()])
        if z_grid.ndim == 0:
            z_grid = np.array([z_grid.item()])

        if enhancement.ndim == 1:
            enhancement = enhancement.reshape(1, -1)
            x_grid = x_grid.reshape(1, -1) if x_grid.ndim == 1 else x_grid
            y_grid = y_grid.reshape(1, -1) if y_grid.ndim == 1 else y_grid
            z_grid = z_grid.reshape(1, -1) if z_grid.ndim == 1 else z_grid

        if enhancement.ndim == 2:
            dx = np.abs(x_grid[0, 1] - x_grid[0, 0]) if x_grid.shape[1] > 1 else 1.0
            dy = np.abs(y_grid[1, 0] - y_grid[0, 0]) if y_grid.shape[0] > 1 else 1.0
            element_area = dx * dy if dx > 0 and dy > 0 else 1.0
            is_3d = False
        else:
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
                regions['enhancement_above_{}'.format(threshold)] = {
                    'num_points': int(count),
                    'volume_nm3': float(volume)
                }
            else:
                area = count * element_area
                regions['enhancement_above_{}'.format(threshold)] = {
                    'num_points': int(count),
                    'area_nm2': float(area)
                }

        return regions

    def _print_analysis(self,
            analysis: Dict[str, Any]) -> None:

        pol_idx = analysis.get('polarization_idx', '?')
        wl_idx = analysis.get('wavelength_idx', '?')
        print('\n  Field Analysis (lambda = {:.1f} nm, wl_idx={}, pol_idx={}):'.format(
            analysis['wavelength'], wl_idx, pol_idx))
        print('  ' + '-' * 50)

        stats = analysis['enhancement_stats']
        print('  Enhancement Statistics:')
        print('    Max:       {:.2f}'.format(stats['max']))
        print('    Mean:      {:.2f}'.format(stats['mean']))
        print('    Median:    {:.2f}'.format(stats['median']))
        print('    95th %ile: {:.2f}'.format(stats['percentile_95']))

        if analysis['hotspots']:
            print('\n  Top {} Hotspots:'.format(len(analysis['hotspots'])))
            for hotspot in analysis['hotspots'][:5]:
                pos = hotspot['position']
                print('    #{}: ({:.1f}, {:.1f}, {:.1f}) nm | E/E0 = {:.2f}'.format(
                    hotspot['rank'], pos[0], pos[1], pos[2], hotspot['enhancement']))

    def calculate_near_field_integration(self,
            field_data_list: List[Dict[str, Any]],
            config: Dict[str, Any],
            geometry: Any,
            center_only: bool = False) -> Optional[Dict[str, Any]]:

        structure_type = config.get('structure', 'unknown')

        if not self._is_structure_supported_for_integration(structure_type):
            if self.verbose:
                print('  [!] Structure <{}> not supported for near-field integration'.format(structure_type))
            return None

        results = {}

        for field_data in field_data_list:
            wl = field_data['wavelength']
            pol_idx = field_data.get('polarization_idx')

            if self.verbose:
                pol_str = 'pol{}'.format(pol_idx + 1) if pol_idx is not None else 'unpolarized'
                mode_str = ' (center only)' if center_only else ''
                print('\n  Processing lambda={:.1f} nm, {}{}'.format(wl, pol_str, mode_str))

            integration_result = self._integrate_single_field(
                field_data, config, geometry, center_only = center_only
            )

            if wl not in results:
                results[wl] = {}

            if pol_idx is not None:
                key = 'polarization_{}'.format(pol_idx + 1)
            else:
                key = 'unpolarized'

            results[wl][key] = integration_result

        return results

    def _integrate_single_field(self,
            field_data: Dict[str, Any],
            config: Dict[str, Any],
            geometry: Any,
            center_only: bool = False) -> Dict[str, Any]:

        enhancement = field_data['enhancement']
        intensity = field_data['intensity']
        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']

        e_sq = field_data.get('e_sq')
        e0_sq = field_data.get('e0_sq')
        e_sq_int = field_data.get('e_sq_int')

        if np.iscomplexobj(enhancement):
            enhancement = np.abs(enhancement)
        if intensity is not None and np.iscomplexobj(intensity):
            intensity = np.abs(intensity)
        if e_sq is not None and np.iscomplexobj(e_sq):
            e_sq = np.abs(e_sq)
        if e0_sq is not None and np.iscomplexobj(e0_sq):
            e0_sq = np.abs(e0_sq)
        if e_sq_int is not None and np.iscomplexobj(e_sq_int):
            e_sq_int = np.abs(e_sq_int)

        if self.verbose:
            enh_finite = enhancement[np.isfinite(enhancement)]
            print('    [info] Enhancement array:')
            print('      Shape: {}'.format(enhancement.shape))
            print('      Total points: {}'.format(enhancement.size))
            print('      Finite points: {}'.format(len(enh_finite)))
            print('      NaN points: {}'.format(np.sum(np.isnan(enhancement))))
            print('      Inf points: {}'.format(np.sum(np.isinf(enhancement))))
            if len(enh_finite) > 0:
                print('      Range: {:.3f} ~ {:.3f}'.format(np.min(enh_finite), np.max(enh_finite)))
                print('      Mean: {:.3f}'.format(np.mean(enh_finite)))

        spheres = self._get_sphere_boundaries(config, geometry, center_only = center_only)

        if spheres is None or len(spheres) == 0:
            if self.verbose:
                print('    [!] Could not determine sphere boundaries')
            return self._empty_integration_result()

        n_spheres = len(spheres)

        if self.verbose:
            print('    [info] Spheres ({} total):'.format(n_spheres))
            for i, (cx, cy, cz, r) in enumerate(spheres):
                print('      Sphere {}: center=({:.1f}, {:.1f}, {:.1f}), radius={:.1f} nm'.format(
                    i + 1, cx, cy, cz, r))

        results_by_depth = {}

        for depth in self.near_field_distances:

            if self.verbose:
                print('    [info] Processing depth = {:.1f} nm (interior)'.format(depth))

            distance_mask = self._create_distance_mask(x_grid, y_grid, z_grid, spheres, depth)

            artifact_mask, n_artifacts = find_edge_artifacts(
                enhancement, x_grid, y_grid, z_grid, spheres,
                mask = distance_mask, edge_threshold = 1.0, isolation_ratio = 1.3,
                verbose = self.verbose
            )

            if self.verbose and n_artifacts > 0:
                print('    Edge artifacts found: {} pixels'.format(n_artifacts))

            result_strict = self._calculate_with_filtering(
                enhancement, intensity, distance_mask, n_spheres,
                e_sq = e_sq, e0_sq = e0_sq, e_sq_int = e_sq_int, method = 'strict',
                artifact_mask = artifact_mask
            )

            result_conservative = self._calculate_with_filtering(
                enhancement, intensity, distance_mask, n_spheres,
                e_sq = e_sq, e0_sq = e0_sq, e_sq_int = e_sq_int, method = 'conservative',
                artifact_mask = artifact_mask
            )

            results_by_depth[depth] = {
                'strict': result_strict,
                'conservative': result_conservative,
                'n_spheres': n_spheres,
                'n_artifacts_removed': n_artifacts
            }

        grid_info = {
            'total_points': int(np.prod(enhancement.shape)),
            'valid_points': int(np.sum(~np.isnan(enhancement)))
        }

        return {
            'depths': results_by_depth,
            'grid_info': grid_info
        }

    def _calculate_with_filtering(self,
            enhancement: np.ndarray,
            intensity: Optional[np.ndarray],
            distance_mask: np.ndarray,
            n_spheres: int,
            e_sq: Optional[np.ndarray] = None,
            e0_sq: Optional[np.ndarray] = None,
            e_sq_int: Optional[np.ndarray] = None,
            method: str = 'strict',
            artifact_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:

        final_mask = distance_mask.copy()

        if self.verbose:
            print('      [info] {} filtering:'.format(method.upper()))
            print('        Distance mask: {} points'.format(np.sum(distance_mask)))

        if method == 'strict':
            valid_enh = np.isfinite(enhancement)

            if self.verbose:
                print('        Finite enhancement: {} points'.format(np.sum(valid_enh)))

            final_mask = final_mask & valid_enh
            excluded_outliers = 0

        elif method == 'conservative':
            valid_enh = np.isfinite(enhancement)

            if self.verbose:
                print('        Finite enhancement: {} points'.format(np.sum(valid_enh)))

            if np.sum(valid_enh) > 0:
                threshold = np.percentile(enhancement[valid_enh], 99.9)
                outlier_mask = enhancement <= threshold * 10

                if self.verbose:
                    print('        99.9%ile threshold: {:.3f}'.format(threshold))
                    print('        Outlier cutoff: {:.3f}'.format(threshold * 10))

                excluded_outliers = int(np.sum(distance_mask & valid_enh & ~outlier_mask))

                final_mask = final_mask & valid_enh & outlier_mask
            else:
                final_mask = final_mask & valid_enh
                excluded_outliers = 0

        excluded_artifacts = 0
        if artifact_mask is not None:
            excluded_artifacts = int(np.sum(final_mask & artifact_mask))
            final_mask = final_mask & ~artifact_mask
            if self.verbose and excluded_artifacts > 0:
                print('        Edge artifact filter: removed {} pixels'.format(excluded_artifacts))

        if self.verbose:
            print('        Final mask: {} points'.format(np.sum(final_mask)))

        enh_in_region = enhancement[final_mask]

        if len(enh_in_region) == 0:

            if self.verbose:
                print('        [!] No valid points found in region!')

            return self._empty_integration_result(method)

        enh_sum = float(np.sum(enh_in_region))
        enh_mean = float(np.mean(enh_in_region))

        enh_per_sphere = enh_sum / n_spheres if n_spheres > 0 else 0.0

        if self.verbose:
            print('        Enhancement sum: {:.3f}'.format(enh_sum))
            print('        Enhancement mean: {:.3f}'.format(enh_mean))
            print('        Per-sphere: {:.3f}'.format(enh_per_sphere))

        result = {
            'enhancement_sum': enh_sum,
            'enhancement_mean': enh_mean,
            'enhancement_per_sphere': enh_per_sphere,
            'valid_points': int(np.sum(final_mask)),
        }

        if intensity is not None:
            int_in_region = intensity[final_mask]
            int_sum = float(np.sum(int_in_region))
            int_mean = float(np.mean(int_in_region))
            int_per_sphere = int_sum / n_spheres if n_spheres > 0 else 0.0

            result['intensity_sum'] = int_sum
            result['intensity_mean'] = int_mean
            result['intensity_per_sphere'] = int_per_sphere
        else:
            result['intensity_sum'] = None
            result['intensity_mean'] = None
            result['intensity_per_sphere'] = None

        e_sq_to_use = e_sq_int if e_sq_int is not None else e_sq

        if e_sq_to_use is not None and e0_sq is not None:
            e_sq_in_region = e_sq_to_use[final_mask]
            e0_sq_in_region = e0_sq[final_mask]

            valid_e_sq = np.isfinite(e_sq_in_region)
            valid_e0_sq = np.isfinite(e0_sq_in_region)
            valid_both = valid_e_sq & valid_e0_sq

            if np.sum(valid_both) > 0:
                e_sq_sum_val = float(np.sum(e_sq_in_region[valid_both]))
                e0_sq_sum_val = float(np.sum(e0_sq_in_region[valid_both]))

                if e0_sq_sum_val > 1e-20:
                    energy_ratio = e_sq_sum_val / e0_sq_sum_val
                    energy_ratio_per_sphere = energy_ratio / n_spheres if n_spheres > 0 else 0.0
                else:
                    energy_ratio = None
                    energy_ratio_per_sphere = None
                    e_sq_sum_val = None
                    e0_sq_sum_val = None

                result['e_sq_sum'] = e_sq_sum_val
                result['e0_sq_sum'] = e0_sq_sum_val
                result['energy_ratio'] = energy_ratio
                result['energy_ratio_per_sphere'] = energy_ratio_per_sphere

                if self.verbose:
                    if energy_ratio is not None:
                        print('        Energy ratio: sum(|E|^2)/sum(|E0|^2) = {:.6f}'.format(energy_ratio))
                        print('        Per-sphere energy ratio: {:.6f}'.format(energy_ratio_per_sphere))
            else:
                result['e_sq_sum'] = None
                result['e0_sq_sum'] = None
                result['energy_ratio'] = None
                result['energy_ratio_per_sphere'] = None
        else:
            result['e_sq_sum'] = None
            result['e0_sq_sum'] = None
            result['energy_ratio'] = None
            result['energy_ratio_per_sphere'] = None

        if method == 'conservative':
            result['excluded_outliers'] = excluded_outliers

        result['excluded_artifacts'] = excluded_artifacts

        return result

    def _create_distance_mask(self,
            x_grid: np.ndarray,
            y_grid: np.ndarray,
            z_grid: np.ndarray,
            spheres: List[Tuple[float, float, float, float]],
            depth: float) -> np.ndarray:

        shape = x_grid.shape

        integration_mask = np.zeros(shape, dtype = bool)

        if self.verbose:
            print('      Grid shape: {}, total points: {}'.format(shape, np.prod(shape)))

        for sphere_idx, (cx, cy, cz, radius) in enumerate(spheres):
            dist_from_center = np.sqrt(
                (x_grid - cx)**2 +
                (y_grid - cy)**2 +
                (z_grid - cz)**2
            )

            dist_from_surface = dist_from_center - radius

            inside_near_surface = (
                (dist_from_surface <= 0) &
                (dist_from_surface >= -depth)
            )

            if self.verbose:
                n_inside_total = np.sum(dist_from_surface <= 0)
                n_inside_near = np.sum(inside_near_surface)
                print('      Sphere {}: {}/{} points in near-surface region'.format(
                    sphere_idx + 1, n_inside_near, n_inside_total))

            integration_mask = integration_mask | inside_near_surface

        if self.verbose:
            n_total = np.prod(shape)
            n_selected = np.sum(integration_mask)
            print('    Integration region ({:.1f}nm interior): {}/{} points ({:.1f}%)'.format(
                depth, n_selected, n_total, 100 * n_selected / n_total))

        return integration_mask

    def _get_sphere_boundaries(self,
            config: Dict[str, Any],
            geometry: Any,
            center_only: bool = False) -> Optional[List[Tuple[float, float, float, float]]]:

        structure = config.get('structure', 'unknown')

        if structure in ['sphere_cluster_aggregate', 'sphere_cluster']:
            return self._get_cluster_spheres(config, geometry, center_only = center_only)
        elif structure == 'sphere':
            return self._get_single_sphere(config)
        elif structure in ['dimer_sphere', 'dimer']:
            return self._get_dimer_spheres(config)
        else:
            if self.verbose:
                print('    [!] Sphere boundary extraction not implemented for <{}>'.format(structure))
            return None

    def _get_cluster_spheres(self,
            config: Dict[str, Any],
            geometry: Any,
            center_only: bool = False) -> List[Tuple[float, float, float, float]]:

        n_spheres = config.get('n_spheres', 1)
        diameter = config.get('diameter', 50.0)
        gap = config.get('gap', -0.1)

        radius = diameter / 2
        spacing = diameter + gap

        positions = geometry._calculate_cluster_positions(n_spheres, spacing)

        spheres = [(pos[0], pos[1], pos[2], radius) for pos in positions]

        if center_only:
            spheres = [spheres[0]]
            if self.verbose:
                print('    Using center sphere only (r={:.1f} nm)'.format(radius))
        else:
            if self.verbose:
                print('    Using all {} spheres (r={:.1f} nm)'.format(len(spheres), radius))

        return spheres

    def _get_single_sphere(self,
            config: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:

        diameter = config.get('diameter', 50.0)
        radius = diameter / 2
        center = config.get('center', [0, 0, 0])

        return [(center[0], center[1], center[2], radius)]

    def _get_dimer_spheres(self,
            config: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:

        diameter = config.get('diameter', 50.0)
        gap = config.get('gap', 5.0)
        radius = diameter / 2

        spacing = diameter + gap
        offset = spacing / 2

        return [
            (-offset, 0, 0, radius),
            (offset, 0, 0, radius)
        ]

    def _is_structure_supported_for_integration(self,
            structure_type: str) -> bool:

        supported = [
            'sphere',
            'sphere_cluster',
            'sphere_cluster_aggregate',
            'dimer_sphere',
            'dimer'
        ]
        return structure_type in supported

    def _empty_integration_result(self,
            method: str = 'strict') -> Dict[str, Any]:

        result = {
            'enhancement_sum': 0.0,
            'enhancement_mean': 0.0,
            'enhancement_per_sphere': 0.0,
            'intensity_sum': 0.0,
            'intensity_mean': 0.0,
            'intensity_per_sphere': 0.0,
            'e_sq_sum': None,
            'e0_sq_sum': None,
            'energy_ratio': None,
            'energy_ratio_per_sphere': None,
            'valid_points': 0,
        }
        if method == 'conservative':
            result['excluded_outliers'] = 0
        return result

    def save_near_field_results(self,
            results: Dict[str, Any],
            config: Dict[str, Any],
            output_path: str,
            center_only: bool = False) -> None:

        with open(output_path, 'w') as f:
            self._write_integration_header(f, config, center_only = center_only)
            self._write_integration_results(f, results)
            self._write_integration_summary(f, results)

        if self.verbose:
            mode_str = ' (center sphere only)' if center_only else ''
            print('\n[OK] Near-field integration results{} saved: {}'.format(mode_str, output_path))

    def _write_integration_header(self,
            f: Any,
            config: Dict[str, Any],
            center_only: bool = False) -> None:

        f.write('=' * 80 + '\n')
        title_parts = ['Near-Field Integration Analysis (INTERIOR)']
        if center_only:
            title_parts.append('CENTER SPHERE ONLY')
        title_parts.append('HYBRID EDGE FILTER')
        f.write(' - '.join(title_parts) + '\n')
        f.write('=' * 80 + '\n\n')

        f.write('Configuration:\n')
        f.write('  Integration depths: {} nm from particle surface (interior)\n'.format(
            ', '.join(['{:.1f}'.format(d) for d in self.near_field_distances])))
        f.write('  Artifact filter: Hybrid edge + spatial isolation (edge<=1nm, isolation_ratio>5x)\n')

        structure_type = config.get('structure', 'unknown')
        f.write('  Structure: {}\n'.format(structure_type))

        if structure_type in ['sphere_cluster_aggregate', 'sphere_cluster']:
            n_spheres = config.get('n_spheres', 1)
            diameter = config.get('diameter', 50.0)
            gap = config.get('gap', -0.1)
            f.write('  Total spheres in cluster: {}\n'.format(n_spheres))
            if center_only:
                f.write('  Integration target: CENTER SPHERE ONLY (sphere #0)\n')
            else:
                f.write('  Integration target: ALL SPHERES\n')
            f.write('  Sphere diameter: {:.1f} nm\n'.format(diameter))
            f.write('  Gap: {:.3f} nm\n'.format(gap))

        f.write('\n' + '=' * 80 + '\n\n')

    def _write_integration_results(self,
            f: Any,
            results: Dict[str, Any]) -> None:

        for wl in sorted(results.keys()):
            f.write('Results at wavelength = {:.1f} nm:\n'.format(wl))
            f.write('\n' + '-' * 80 + '\n')

            wl_results = results[wl]

            for pol_key in sorted(wl_results.keys()):
                pol_data = wl_results[pol_key]

                if pol_key == 'unpolarized':
                    pol_label = 'Unpolarized (average)'
                else:
                    pol_num = pol_key.split('_')[1]
                    pol_label = 'Polarization {}'.format(pol_num)

                f.write('{}\n'.format(pol_label))
                f.write('-' * 80 + '\n\n')

                if 'grid_info' in pol_data:
                    grid_info = pol_data['grid_info']
                    f.write('Grid information:\n')
                    f.write('  Total grid points:       {}\n'.format(grid_info['total_points']))
                    f.write('  Valid points (not NaN):  {}\n\n'.format(grid_info['valid_points']))

                if 'depths' in pol_data:
                    for depth in sorted(pol_data['depths'].keys()):
                        depth_data = pol_data['depths'][depth]
                        n_spheres = depth_data.get('n_spheres', 1)

                        f.write('Integration depth: {:.1f} nm (interior)\n'.format(depth))
                        f.write('  Number of spheres: {}\n\n'.format(n_spheres))

                        strict = depth_data['strict']
                        f.write('  Strict filtering (Inf only):\n')
                        f.write('    Enhancement sum:         {:15.3f}  [sum(|E/E0|)]\n'.format(strict['enhancement_sum']))
                        f.write('    Enhancement per sphere:  {:15.3f}\n'.format(strict['enhancement_per_sphere']))
                        if strict['intensity_sum'] is not None:
                            f.write('    Intensity sum:           {:15.3f}  [sum(|E/E0|^2)]\n'.format(strict['intensity_sum']))
                            f.write('    Intensity per sphere:    {:15.3f}\n'.format(strict['intensity_per_sphere']))
                        if strict.get('energy_ratio') is not None:
                            f.write('    Energy ratio:            {:15.6f}  [sum(|E|^2)/sum(|E0|^2)]\n'.format(strict['energy_ratio']))
                            if strict.get('energy_ratio_per_sphere') is not None:
                                f.write('    Energy ratio per sphere: {:15.6f}\n'.format(strict['energy_ratio_per_sphere']))
                        f.write('    Valid points in region:  {:15d}\n'.format(strict['valid_points']))
                        if strict.get('excluded_artifacts', 0) > 0:
                            f.write('    Excluded edge artifacts: {:15d}\n'.format(strict['excluded_artifacts']))
                        f.write('    Mean enhancement:        {:15.3f}\n'.format(strict['enhancement_mean']))
                        if strict['intensity_mean'] is not None:
                            f.write('    Mean intensity:          {:15.3f}\n'.format(strict['intensity_mean']))
                        f.write('\n')

                        cons = depth_data['conservative']
                        f.write('  Conservative filtering (Inf + outliers):\n')
                        f.write('    Enhancement sum:         {:15.3f}  [sum(|E/E0|)]\n'.format(cons['enhancement_sum']))
                        f.write('    Enhancement per sphere:  {:15.3f}\n'.format(cons['enhancement_per_sphere']))
                        if cons['intensity_sum'] is not None:
                            f.write('    Intensity sum:           {:15.3f}  [sum(|E/E0|^2)]\n'.format(cons['intensity_sum']))
                            f.write('    Intensity per sphere:    {:15.3f}\n'.format(cons['intensity_per_sphere']))
                        if cons.get('energy_ratio') is not None:
                            f.write('    Energy ratio:            {:15.6f}  [sum(|E|^2)/sum(|E0|^2)]\n'.format(cons['energy_ratio']))
                            if cons.get('energy_ratio_per_sphere') is not None:
                                f.write('    Energy ratio per sphere: {:15.6f}\n'.format(cons['energy_ratio_per_sphere']))
                        f.write('    Valid points in region:  {:15d}\n'.format(cons['valid_points']))
                        f.write('    Excluded outliers:       {:15d}\n'.format(cons['excluded_outliers']))
                        if cons.get('excluded_artifacts', 0) > 0:
                            f.write('    Excluded edge artifacts: {:15d}\n'.format(cons['excluded_artifacts']))
                        f.write('    Mean enhancement:        {:15.3f}\n'.format(cons['enhancement_mean']))
                        if cons['intensity_mean'] is not None:
                            f.write('    Mean intensity:          {:15.3f}\n'.format(cons['intensity_mean']))
                        f.write('\n' + '-' * 80 + '\n')

            f.write('\n')

    def _write_integration_summary(self,
            f: Any,
            results: Dict[str, Any]) -> None:

        f.write('=' * 100 + '\n')
        f.write('Summary (Strict Filtering)\n')
        f.write('=' * 100 + '\n\n')

        f.write('{:<12} {:<15} {:<8} {:<15} {:<15} {:<15} {:<10}\n'.format(
            'Wavelength', 'Polarization', 'Depth', 'Enh.Sum', 'Int.Sum', 'Energy Ratio', 'Points'))
        f.write('-' * 100 + '\n')

        for wl in sorted(results.keys()):
            wl_results = results[wl]

            for pol_key in sorted(wl_results.keys()):
                pol_data = wl_results[pol_key]

                if pol_key == 'unpolarized':
                    pol_str = 'unpolarized'
                else:
                    pol_num = pol_key.split('_')[1]
                    pol_str = 'pol{}'.format(pol_num)

                if 'depths' in pol_data:
                    for depth in sorted(pol_data['depths'].keys()):
                        depth_data = pol_data['depths'][depth]
                        strict = depth_data['strict']

                        wl_str = '{:.1f} nm'.format(wl)
                        depth_str = '{:.1f}nm'.format(depth)
                        enh_sum = strict['enhancement_sum']
                        int_sum = strict['intensity_sum'] if strict['intensity_sum'] is not None else 0
                        energy_ratio = strict.get('energy_ratio')
                        energy_ratio_str = '{:.6f}'.format(energy_ratio) if energy_ratio is not None else 'N/A'
                        points = strict['valid_points']

                        f.write('{:<12} {:<15} {:<8} {:<15.3f} {:<15.3f} {:<15} {:<10d}\n'.format(
                            wl_str, pol_str, depth_str, enh_sum, int_sum, energy_ratio_str, points))
