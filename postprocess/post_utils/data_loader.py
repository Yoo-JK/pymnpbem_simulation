import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np


class DataLoader(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose

        output_dir = config.get('output_dir')
        simulation_name = config.get('simulation_name')

        if output_dir is None:
            raise ValueError('[error] Config missing required key: <output_dir>')
        if simulation_name is None:
            raise ValueError('[error] Config missing required key: <simulation_name>')

        self.output_dir = os.path.join(output_dir, simulation_name)

    def load_simulation_results(self) -> Dict[str, Any]:

        # Try .npz first, then fall back to .mat
        npz_file = os.path.join(self.output_dir, 'simulation_results.npz')
        mat_file = os.path.join(self.output_dir, 'simulation_results.mat')

        if os.path.exists(npz_file):
            return self._load_npz_results(npz_file)
        elif os.path.exists(mat_file):
            return self._load_mat_results(mat_file)
        else:
            raise FileNotFoundError(
                '[error] Results file not found: {} or {}'.format(npz_file, mat_file))

    def _load_npz_results(self,
            npz_file: str) -> Dict[str, Any]:

        if self.verbose:
            print('[info] Loading results from: {}'.format(npz_file))

        npz_data = np.load(npz_file, allow_pickle = True)

        data = {
            'wavelength': self._extract_array(npz_data.get('wavelength')),
            'scattering': self._extract_array(npz_data.get('scattering')),
            'extinction': self._extract_array(npz_data.get('extinction')),
            'absorption': self._extract_array(npz_data.get('absorption')),
            'polarizations': self._extract_array(npz_data.get('polarizations')),
            'propagation_dirs': self._extract_array(npz_data.get('propagation_dirs')),
        }

        if 'calculation_time' in npz_data:
            data['calculation_time'] = float(npz_data['calculation_time'])

        # Ensure 2D arrays for cross sections
        for key in ['scattering', 'extinction', 'absorption']:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 1:
                data[key] = data[key].reshape(-1, 1)

        data['n_polarizations'] = data['scattering'].shape[1]

        if self.verbose:
            print('  Loaded {} wavelength points'.format(len(data['wavelength'])))
            print('  Polarizations: {}'.format(data['n_polarizations']))

        # Load field data if available
        if 'fields' in npz_data:
            fields_raw = npz_data['fields']
            # allow_pickle=True returns an object array; extract the item
            if isinstance(fields_raw, np.ndarray) and fields_raw.dtype == object:
                fields_raw = fields_raw.item() if fields_raw.ndim == 0 else fields_raw.tolist()
            data['fields'] = self._load_field_data_from_npz(fields_raw)
            if self.verbose and data['fields']:
                unique_wls = set(f.get('wavelength_idx', f.get('wavelength')) for f in data['fields'])
                unique_pols = set(f.get('polarization_idx', 0) for f in data['fields'])
                print('  Field data loaded: {} entries ({} wavelength(s), {} polarization(s))'.format(
                    len(data['fields']), len(unique_wls), len(unique_pols)))

        # Load surface charge data if available
        if 'surface_charge' in npz_data:
            sc_raw = npz_data['surface_charge']
            if isinstance(sc_raw, np.ndarray) and sc_raw.dtype == object:
                sc_raw = sc_raw.item() if sc_raw.ndim == 0 else sc_raw.tolist()
            data['surface_charge'] = self._load_surface_charge_from_npz(sc_raw)
            if self.verbose and data['surface_charge']:
                print('  Surface charge loaded: {} entries'.format(len(data['surface_charge'])))

        return data

    def _load_mat_results(self,
            mat_file: str) -> Dict[str, Any]:

        import scipy.io as sio

        if self.verbose:
            print('[info] Loading results from MAT file (fallback): {}'.format(mat_file))

        mat_data = sio.loadmat(mat_file, struct_as_record = False, squeeze_me = True)

        if 'results' not in mat_data:
            raise KeyError('[error] MAT file missing <results> structure: {}'.format(mat_file))
        results = mat_data['results']

        data = {
            'wavelength': self._extract_array(results.wavelength),
            'scattering': self._extract_array(results.scattering),
            'extinction': self._extract_array(results.extinction),
            'absorption': self._extract_array(results.absorption),
            'polarizations': self._extract_array(results.polarizations),
            'propagation_dirs': self._extract_array(results.propagation_dirs),
        }

        if hasattr(results, 'calculation_time'):
            data['calculation_time'] = float(results.calculation_time)

        # Ensure 2D arrays for cross sections
        for key in ['scattering', 'extinction', 'absorption']:
            if data[key].ndim == 1:
                data[key] = data[key].reshape(-1, 1)

        data['n_polarizations'] = data['scattering'].shape[1]

        if self.verbose:
            print('  Loaded {} wavelength points'.format(len(data['wavelength'])))
            print('  Polarizations: {}'.format(data['n_polarizations']))

        # Load field data if available
        if hasattr(results, 'fields'):
            data['fields'] = self._load_field_data_from_mat(results.fields)
            if self.verbose and data['fields']:
                unique_wls = set(f.get('wavelength_idx', f.get('wavelength')) for f in data['fields'])
                unique_pols = set(f.get('polarization_idx', 0) for f in data['fields'])
                print('  Field data loaded: {} entries ({} wavelength(s), {} polarization(s))'.format(
                    len(data['fields']), len(unique_wls), len(unique_pols)))

        # Load surface charge data if available
        if hasattr(results, 'surface_charge'):
            data['surface_charge'] = self._load_surface_charge_from_mat(results.surface_charge)
            if self.verbose and data['surface_charge']:
                print('  Surface charge loaded: {} entries'.format(len(data['surface_charge'])))

        return data

    def _load_field_data_from_npz(self,
            fields_raw: Any) -> List[Dict[str, Any]]:

        if fields_raw is None:
            return []

        # fields_raw should be a list of dicts (from allow_pickle)
        if not isinstance(fields_raw, (list, tuple)):
            fields_raw = [fields_raw]

        field_data_list = []

        for field_item in fields_raw:
            if isinstance(field_item, dict):
                field_dict = self._extract_field_dict_from_dict(field_item)
            else:
                # numpy structured array or object - try dict conversion
                field_dict = self._extract_field_dict_from_dict(
                    dict(field_item) if hasattr(field_item, 'keys') else {'wavelength': 0})

            field_data_list.append(field_dict)

        return field_data_list

    def _extract_field_dict_from_dict(self,
            d: Dict[str, Any]) -> Dict[str, Any]:

        required_fields = ['wavelength', 'polarization', 'x_grid', 'y_grid', 'z_grid']
        missing_fields = [f for f in required_fields if f not in d]
        if missing_fields:
            raise KeyError('[error] Field data missing required keys: {}'.format(missing_fields))

        field_dict = {
            'wavelength': float(d['wavelength']),
            'polarization': self._extract_array(d['polarization']),
            'x_grid': self._extract_array(d['x_grid']),
            'y_grid': self._extract_array(d['y_grid']),
            'z_grid': self._extract_array(d['z_grid']),
        }

        # Optional integer index fields
        if 'wavelength_idx' in d:
            field_dict['wavelength_idx'] = int(d['wavelength_idx'])
        if 'polarization_idx' in d:
            field_dict['polarization_idx'] = int(d['polarization_idx'])

        # Optional array fields
        optional_keys = [
            'e_total', 'e_induced', 'e_incoming',
            'enhancement', 'intensity',
            'enhancement_ext', 'enhancement_int',
            'intensity_ext', 'intensity_int',
            'e_sq', 'e0_sq', 'e_sq_ext', 'e_sq_int',
        ]
        for key in optional_keys:
            if key in d and d[key] is not None:
                field_dict[key] = self._extract_array(d[key])

        return field_dict

    def _load_surface_charge_from_npz(self,
            sc_raw: Any) -> List[Dict[str, Any]]:

        if sc_raw is None:
            return []

        if not isinstance(sc_raw, (list, tuple)):
            sc_raw = [sc_raw]

        surface_charge_list = []

        for sc in sc_raw:
            if not isinstance(sc, dict):
                if hasattr(sc, 'keys'):
                    sc = dict(sc)
                else:
                    continue

            sc_data = {
                'wavelength': float(sc['wavelength']) if 'wavelength' in sc else None,
                'wavelength_idx': int(sc['wavelength_idx']) if 'wavelength_idx' in sc else None,
                'polarization': self._extract_array(sc.get('polarization')),
                'polarization_idx': int(sc['polarization_idx']) if 'polarization_idx' in sc else None,
                'vertices': self._extract_array(sc.get('vertices')),
                'faces': self._extract_array(sc.get('faces')),
                'centroids': self._extract_array(sc.get('centroids')),
                'normals': self._extract_array(sc.get('normals')),
                'areas': self._extract_array(sc.get('areas')),
                'charge': self._extract_array(sc.get('charge')),
            }

            surface_charge_list.append(sc_data)

        return surface_charge_list

    def _load_field_data_from_mat(self,
            fields_struct: Any) -> List[Dict[str, Any]]:

        if fields_struct is None:
            return []

        if not isinstance(fields_struct, np.ndarray):
            fields_struct = [fields_struct]

        field_data_list = []

        for field_item in fields_struct:
            required_fields = ['wavelength', 'polarization', 'x_grid', 'y_grid', 'z_grid']
            missing_fields = [f for f in required_fields if not hasattr(field_item, f)]
            if missing_fields:
                raise AttributeError(
                    '[error] Field data missing required attributes: {}'.format(missing_fields))

            field_dict = {
                'wavelength': float(field_item.wavelength),
                'polarization': self._extract_array(field_item.polarization),
                'x_grid': self._extract_array(field_item.x_grid),
                'y_grid': self._extract_array(field_item.y_grid),
                'z_grid': self._extract_array(field_item.z_grid),
            }

            if hasattr(field_item, 'wavelength_idx'):
                field_dict['wavelength_idx'] = int(field_item.wavelength_idx)
            if hasattr(field_item, 'polarization_idx'):
                field_dict['polarization_idx'] = int(field_item.polarization_idx)

            optional_attrs = [
                'e_total', 'e_induced', 'e_incoming',
                'enhancement', 'intensity',
                'enhancement_ext', 'enhancement_int',
                'intensity_ext', 'intensity_int',
                'e_sq', 'e0_sq', 'e_sq_ext', 'e_sq_int',
            ]
            for attr in optional_attrs:
                if hasattr(field_item, attr):
                    field_dict[attr] = self._extract_array(getattr(field_item, attr))

            field_data_list.append(field_dict)

        return field_data_list

    def _load_surface_charge_from_mat(self,
            sc_struct: Any) -> List[Dict[str, Any]]:

        if sc_struct is None:
            return []

        if not isinstance(sc_struct, np.ndarray):
            sc_struct = [sc_struct]

        surface_charge_list = []

        for sc in sc_struct:
            sc_data = {
                'wavelength': float(sc.wavelength) if hasattr(sc, 'wavelength') else None,
                'wavelength_idx': int(sc.wavelength_idx) if hasattr(sc, 'wavelength_idx') else None,
                'polarization': self._extract_array(sc.polarization) if hasattr(sc, 'polarization') else None,
                'polarization_idx': int(sc.polarization_idx) if hasattr(sc, 'polarization_idx') else None,
                'vertices': self._extract_array(sc.vertices) if hasattr(sc, 'vertices') else None,
                'faces': self._extract_array(sc.faces) if hasattr(sc, 'faces') else None,
                'centroids': self._extract_array(sc.centroids) if hasattr(sc, 'centroids') else None,
                'normals': self._extract_array(sc.normals) if hasattr(sc, 'normals') else None,
                'areas': self._extract_array(sc.areas) if hasattr(sc, 'areas') else None,
                'charge': self._extract_array(sc.charge) if hasattr(sc, 'charge') else None,
            }

            surface_charge_list.append(sc_data)

        return surface_charge_list

    def _extract_array(self,
            val: Any) -> Any:

        if val is None:
            return np.array([])

        arr = np.array(val)

        if arr.ndim == 0:
            return arr.item()

        return arr

    def load_text_results(self) -> Dict[str, Any]:

        txt_file = os.path.join(self.output_dir, 'simulation_results.txt')

        if not os.path.exists(txt_file):
            raise FileNotFoundError('[error] Results file not found: {}'.format(txt_file))

        if self.verbose:
            print('[info] Loading results from text file: {}'.format(txt_file))

        data_array = np.loadtxt(txt_file, skiprows = 1)

        n_cols = data_array.shape[1]
        if (n_cols - 1) % 3 != 0:
            raise ValueError(
                '[error] Invalid text file format: expected (3*n_pol + 1) columns, got {}.'.format(n_cols))
        n_pol = (n_cols - 1) // 3

        if n_pol < 1:
            raise ValueError('[error] Invalid number of polarizations calculated: {}'.format(n_pol))

        data = {
            'wavelength': data_array[:, 0],
            'scattering': data_array[:, 1:n_pol + 1],
            'extinction': data_array[:, n_pol + 1:2 * n_pol + 1],
            'absorption': data_array[:, 2 * n_pol + 1:],
            'n_polarizations': n_pol,
        }

        if self.verbose:
            print('  Loaded {} wavelength points'.format(len(data['wavelength'])))
            print('  Polarizations: {}'.format(n_pol))

        return data
