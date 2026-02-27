import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
from scipy import signal


class SpectrumAnalyzer(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose

    def analyze(self,
            data: Dict[str, Any]) -> Dict[str, Any]:

        results = {}

        calculate_cross_sections = self.config.get('calculate_cross_sections', True)
        field_wavelength_idx = self.config.get('field_wavelength_idx', 'middle')

        skip_analysis = False
        skip_reason = None

        if not calculate_cross_sections:
            skip_analysis = True
            skip_reason = 'calculate_cross_sections=False'
        elif isinstance(field_wavelength_idx, list):
            skip_analysis = True
            skip_reason = 'field_wavelength_idx is a list ({} wavelengths specified)'.format(
                len(field_wavelength_idx))

        if skip_analysis:
            if self.verbose:
                print('  [!] {}, skipping spectrum analysis'.format(skip_reason))
            results['field_only_mode'] = True
            results['peak_wavelengths'] = []
            results['peak_values'] = []
            results['peak_indices'] = []
            results['fwhm'] = []
            return results

        peak_wavelengths, peak_values, peak_indices = self._find_peaks(
            data['wavelength'],
            data['scattering']
        )

        results['peak_wavelengths'] = peak_wavelengths
        results['peak_values'] = peak_values
        results['peak_indices'] = peak_indices

        fwhm_values = self._calculate_fwhm(
            data['wavelength'],
            data['scattering'],
            peak_indices
        )
        results['fwhm'] = fwhm_values

        if data['n_polarizations'] > 1:
            enhancement = self._calculate_enhancement(data['scattering'])
            results['enhancement_factors'] = enhancement

        results['avg_scattering'] = np.mean(data['scattering'], axis = 0)
        results['avg_extinction'] = np.mean(data['extinction'], axis = 0)
        results['avg_absorption'] = np.mean(data['absorption'], axis = 0)

        results['max_scattering'] = np.max(data['scattering'], axis = 0)
        results['max_extinction'] = np.max(data['extinction'], axis = 0)
        results['max_absorption'] = np.max(data['absorption'], axis = 0)

        # Unpolarized calculation (FDTD-style incoherent averaging)
        excitation_type = self.config.get('excitation_type', 'planewave')
        polarizations = data.get('polarizations', self.config.get('polarizations', []))

        unpolarized_info = self._check_unpolarized_conditions(
            polarizations, excitation_type, data['n_polarizations']
        )
        results['unpolarized'] = unpolarized_info

        if unpolarized_info['can_calculate']:
            unpol_data = self._calculate_unpolarized_spectrum(data, unpolarized_info)
            results['unpolarized_spectrum'] = unpol_data

            if self.verbose:
                print('  Unpolarized calculation: {}'.format(unpolarized_info['method']))
                print('    Peak wavelength: {:.1f} nm'.format(unpol_data['peak_wavelength']))
                print('    Peak absorption: {:.2f} nm^2'.format(unpol_data['peak_absorption']))

        if self.verbose:
            print('Spectrum analysis complete:')
            print('  Peak wavelengths: {}'.format(peak_wavelengths))
            print('  Peak values: {}'.format(peak_values))
            print('  FWHM: {}'.format(fwhm_values))

        return results

    def _check_unpolarized_conditions(self,
            polarizations: Any,
            excitation_type: str,
            n_polarizations: int) -> Dict[str, Any]:

        result = {
            'can_calculate': False,
            'method': None,
            'reason': None
        }

        if excitation_type == 'eels':
            result['reason'] = 'EELS excitation does not support unpolarized calculation'
            return result

        if isinstance(polarizations, list):
            polarizations = np.array(polarizations)

        if len(polarizations) == 0:
            result['reason'] = 'No polarization data available'
            return result

        if excitation_type == 'planewave':
            if n_polarizations != 2:
                result['reason'] = 'Plane wave unpolarized requires exactly 2 polarizations (got {})'.format(
                    n_polarizations)
                return result

            if self._are_orthogonal(polarizations[0], polarizations[1]):
                result['can_calculate'] = True
                result['method'] = 'orthogonal_2pol_average'
                result['reason'] = 'Two orthogonal polarizations detected'
            else:
                result['reason'] = 'Polarizations are not orthogonal'

        elif excitation_type == 'dipole':
            if n_polarizations != 3:
                result['reason'] = 'Dipole unpolarized requires exactly 3 directions (got {})'.format(
                    n_polarizations)
                return result

            if (self._are_orthogonal(polarizations[0], polarizations[1]) and
                self._are_orthogonal(polarizations[1], polarizations[2]) and
                self._are_orthogonal(polarizations[0], polarizations[2])):
                result['can_calculate'] = True
                result['method'] = 'orthogonal_3dir_average'
                result['reason'] = 'Three orthogonal directions detected'
            else:
                result['reason'] = 'Directions are not mutually orthogonal'

        return result

    def _are_orthogonal(self,
            v1: np.ndarray,
            v2: np.ndarray,
            tolerance: float = 1e-6) -> bool:

        v1 = np.array(v1, dtype = float)
        v2 = np.array(v2, dtype = float)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < tolerance or norm2 < tolerance:
            return False

        v1_normalized = v1 / norm1
        v2_normalized = v2 / norm2

        dot_product = np.abs(np.dot(v1_normalized, v2_normalized))
        return dot_product < tolerance

    def _calculate_unpolarized_spectrum(self,
            data: Dict[str, Any],
            unpolarized_info: Dict[str, Any]) -> Dict[str, Any]:

        n_pol = data['n_polarizations']

        unpol_scattering = np.mean(data['scattering'], axis = 1)
        unpol_extinction = np.mean(data['extinction'], axis = 1)
        unpol_absorption = np.mean(data['absorption'], axis = 1)

        peak_idx = np.argmax(unpol_absorption)
        peak_wavelength = data['wavelength'][peak_idx]
        peak_absorption = unpol_absorption[peak_idx]
        peak_extinction = unpol_extinction[peak_idx]
        peak_scattering = unpol_scattering[peak_idx]

        return {
            'wavelength': data['wavelength'],
            'scattering': unpol_scattering,
            'extinction': unpol_extinction,
            'absorption': unpol_absorption,
            'peak_wavelength': peak_wavelength,
            'peak_wavelength_idx': peak_idx,
            'peak_absorption': peak_absorption,
            'peak_extinction': peak_extinction,
            'peak_scattering': peak_scattering,
            'method': unpolarized_info['method'],
            'n_averaged': n_pol
        }

    def _find_peaks(self,
            wavelength: np.ndarray,
            cross_sections: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_pol = cross_sections.shape[1]

        if np.iscomplexobj(cross_sections):
            if self.verbose:
                print('  Note: Converting complex cross_sections to magnitude')
            cross_sections = np.abs(cross_sections)

        peak_wavelengths = np.zeros(n_pol)
        peak_values = np.zeros(n_pol)
        peak_indices = np.zeros(n_pol, dtype = int)

        for i in range(n_pol):
            peaks, properties = signal.find_peaks(
                cross_sections[:, i],
                prominence = 0.1 * np.max(cross_sections[:, i])
            )

            if len(peaks) > 0:
                max_peak_idx = peaks[np.argmax(cross_sections[peaks, i])]
                peak_indices[i] = max_peak_idx
                peak_wavelengths[i] = wavelength[max_peak_idx]
                peak_values[i] = cross_sections[max_peak_idx, i]
            else:
                max_idx = np.argmax(cross_sections[:, i])
                peak_indices[i] = max_idx
                peak_wavelengths[i] = wavelength[max_idx]
                peak_values[i] = cross_sections[max_idx, i]

        return peak_wavelengths, peak_values, peak_indices

    def _calculate_fwhm(self,
            wavelength: np.ndarray,
            cross_sections: np.ndarray,
            peak_indices: np.ndarray) -> np.ndarray:

        n_pol = cross_sections.shape[1]

        if np.iscomplexobj(cross_sections):
            cross_sections = np.abs(cross_sections)

        fwhm_values = np.zeros(n_pol)

        for i in range(n_pol):
            peak_idx = peak_indices[i]
            peak_value = cross_sections[peak_idx, i]
            half_max = peak_value / 2

            above_half = cross_sections[:, i] > half_max

            left_indices = np.where(above_half[:peak_idx])[0]
            if len(left_indices) > 0:
                left_idx = left_indices[0]
            else:
                left_idx = 0

            right_indices = np.where(above_half[peak_idx:])[0]
            if len(right_indices) > 0:
                right_idx = peak_idx + right_indices[-1]
            else:
                right_idx = len(wavelength) - 1

            fwhm_values[i] = wavelength[right_idx] - wavelength[left_idx]

        return fwhm_values

    def _calculate_enhancement(self,
            cross_sections: np.ndarray) -> Dict[str, float]:

        n_pol = cross_sections.shape[1]
        enhancement = {}

        if n_pol > 1:
            max_vals = np.max(cross_sections, axis = 0)
            enhancement['pol1_vs_pol2'] = max_vals[0] / max_vals[1] if max_vals[1] > 0 else 0

            if n_pol > 2:
                enhancement['pol1_vs_pol3'] = max_vals[0] / max_vals[2] if max_vals[2] > 0 else 0
                enhancement['pol2_vs_pol3'] = max_vals[1] / max_vals[2] if max_vals[2] > 0 else 0

        return enhancement
