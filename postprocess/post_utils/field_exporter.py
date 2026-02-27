import os
import sys
import json
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np


class FieldExporter(object):

    def __init__(self,
            output_dir: str,
            verbose: bool = False) -> None:

        self.output_dir = output_dir
        self.verbose = verbose

    def export_to_json(self,
            field_data_list: Union[Dict, List[Dict]],
            field_analysis_list: Union[Dict, List[Dict]]) -> str:

        if not isinstance(field_data_list, list):
            field_data_list = [field_data_list]
        if not isinstance(field_analysis_list, list):
            field_analysis_list = [field_analysis_list]

        json_data = {
            'metadata': {
                'num_polarizations': len(field_data_list),
                'description': 'Electromagnetic field distribution data'
            },
            'fields': []
        }

        for i, (field_data, analysis) in enumerate(zip(field_data_list, field_analysis_list)):
            wl = field_data.get('wavelength', 0)
            if np.iscomplexobj(wl):
                wl = np.abs(wl)

            pol = field_data.get('polarization')
            if pol is not None and hasattr(pol, 'tolist'):
                pol = pol.tolist()

            field_dict = {
                'polarization_index': i + 1,
                'polarization': pol,
                'wavelength_nm': float(wl),
                'grid': self._extract_grid_info(field_data),
                'analysis': analysis if analysis is not None else {},
                'note': 'Full field arrays (enhancement, intensity, E-field components) are in field_data.npz'
            }

            json_data['fields'].append(field_dict)

        filepath = os.path.join(self.output_dir, 'field_analysis.json')
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent = 2)

        if self.verbose:
            print('  Saved: {}'.format(filepath))

        return filepath

    def export_field_data_arrays(self,
            field_data_list: Union[Dict, List[Dict]]) -> str:

        if not isinstance(field_data_list, list):
            field_data_list = [field_data_list]

        downsample = 4

        json_data = {
            'metadata': {
                'warning': 'Data is downsampled by factor of ' + str(downsample),
                'note': 'Full resolution data is in field_data.npz'
            },
            'fields': []
        }

        for i, field_data in enumerate(field_data_list):
            enhancement = field_data['enhancement']
            x_grid = field_data['x_grid']
            y_grid = field_data['y_grid']
            z_grid = field_data['z_grid']

            if not isinstance(enhancement, np.ndarray):
                enhancement = np.array([[enhancement]])
            elif enhancement.ndim == 0:
                enhancement = np.array([[enhancement.item()]])
            elif enhancement.ndim == 1:
                enhancement = enhancement.reshape(1, -1)

            if np.iscomplexobj(enhancement):
                enhancement = np.abs(enhancement)

            if not isinstance(x_grid, np.ndarray):
                x_grid = np.array([[x_grid]])
                y_grid = np.array([[y_grid]])
                z_grid = np.array([[z_grid]])
            elif x_grid.ndim == 0:
                x_grid = np.array([[x_grid.item()]])
                y_grid = np.array([[y_grid.item()]])
                z_grid = np.array([[z_grid.item()]])
            elif x_grid.ndim == 1:
                x_grid = x_grid.reshape(1, -1)
                y_grid = y_grid.reshape(1, -1)
                z_grid = z_grid.reshape(1, -1)

            if self._is_data_concentrated(enhancement):
                if self.verbose:
                    print('  Warning: Data appears concentrated (possible bug). '
                          'Using adaptive downsampling for polarization {}'.format(i + 1))
                enhancement_ds, x_ds, y_ds, z_ds = self._adaptive_downsample(
                    enhancement, x_grid, y_grid, z_grid, max_points = 2500
                )
            else:
                if enhancement.shape[0] > downsample and enhancement.shape[1] > downsample:
                    enhancement_ds = enhancement[::downsample, ::downsample]
                    x_ds = x_grid[::downsample, ::downsample]
                    y_ds = y_grid[::downsample, ::downsample]
                    z_ds = z_grid[::downsample, ::downsample]
                else:
                    enhancement_ds = enhancement
                    x_ds = x_grid
                    y_ds = y_grid
                    z_ds = z_grid

            wl = field_data['wavelength']
            if np.iscomplexobj(wl):
                wl = np.abs(wl)

            field_dict = {
                'polarization_index': i + 1,
                'wavelength_nm': float(wl),
                'x_coordinates': x_ds.tolist(),
                'y_coordinates': y_ds.tolist(),
                'z_coordinates': z_ds.tolist(),
                'enhancement': enhancement_ds.tolist(),
            }

            json_data['fields'].append(field_dict)

        filepath = os.path.join(self.output_dir, 'field_data_downsampled.json')
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent = 2)

        if self.verbose:
            print('  Saved (downsampled): {}'.format(filepath))

        return filepath

    def _is_data_concentrated(self,
            data: np.ndarray,
            threshold: float = 0.8) -> bool:

        if data.ndim != 2:
            return False

        valid_mask = np.isfinite(data) & (data > 0)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return False

        h, w = data.shape
        if h < 5 or w < 5:
            return False

        corner_size_h = h // 5
        corner_size_w = w // 5

        corner_mask = valid_mask[:corner_size_h, :corner_size_w]
        n_corner = np.sum(corner_mask)

        concentration_ratio = n_corner / n_valid

        return concentration_ratio > threshold

    def _adaptive_downsample(self,
            data: np.ndarray,
            x_grid: np.ndarray,
            y_grid: np.ndarray,
            z_grid: np.ndarray,
            max_points: int = 2500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        data_flat = data.flatten()
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()

        valid_mask = np.isfinite(data_flat)

        data_valid = data_flat[valid_mask]
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]

        if len(data_valid) > max_points:
            high_value_mask = data_valid > 1.0

            low_value_indices = np.where(~high_value_mask)[0]
            n_keep_low = max_points - np.sum(high_value_mask)

            if n_keep_low > 0 and len(low_value_indices) > 0:
                keep_low = np.random.choice(
                    low_value_indices,
                    size = min(n_keep_low, len(low_value_indices)),
                    replace = False
                )
                keep_mask = high_value_mask.copy()
                keep_mask[keep_low] = True
            else:
                keep_mask = high_value_mask

            data_valid = data_valid[keep_mask]
            x_valid = x_valid[keep_mask]
            y_valid = y_valid[keep_mask]
            z_valid = z_valid[keep_mask]

        if len(data_valid) == 0:
            return (np.array([[0]]),
                    np.array([[x_grid[0, 0]]]),
                    np.array([[y_grid[0, 0]]]),
                    np.array([[z_grid[0, 0]]]))

        data_2d = data_valid.reshape(1, -1)
        x_2d = x_valid.reshape(1, -1)
        y_2d = y_valid.reshape(1, -1)
        z_2d = z_valid.reshape(1, -1)

        return data_2d, x_2d, y_2d, z_2d

    def _extract_grid_info(self,
            field_data: Dict[str, Any]) -> Dict[str, Any]:

        x_grid = field_data['x_grid']
        y_grid = field_data['y_grid']
        z_grid = field_data['z_grid']

        def to_2d_array(val):
            if val is None:
                return np.array([[0]])
            if not isinstance(val, np.ndarray):
                return np.array([[val]])
            if val.ndim == 0:
                return np.array([[val.item()]])
            if val.ndim == 1:
                return val.reshape(1, -1)
            return val

        x_grid = to_2d_array(x_grid)
        y_grid = to_2d_array(y_grid)
        z_grid = to_2d_array(z_grid)

        grid_info = {
            'shape': list(x_grid.shape),
            'x_range': [float(x_grid.min()), float(x_grid.max())],
            'y_range': [float(y_grid.min()), float(y_grid.max())],
            'z_range': [float(z_grid.min()), float(z_grid.max())],
        }

        if x_grid.ndim == 2 and x_grid.shape[1] > 1:
            grid_info['x_spacing'] = float(np.abs(x_grid[0, 1] - x_grid[0, 0]))
        if y_grid.ndim == 2 and y_grid.shape[0] > 1:
            grid_info['y_spacing'] = float(np.abs(y_grid[1, 0] - y_grid[0, 0]))

        return grid_info
