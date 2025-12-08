"""
Spectrum Analyzer

Analyzes optical spectra to extract key features.
Includes unpolarized light calculation (FDTD-style incoherent averaging).
"""

import numpy as np
from scipy import signal


class SpectrumAnalyzer:
    """Analyzes optical spectrum data."""

    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose

    def analyze(self, data):
        """
        Perform comprehensive spectrum analysis.

        Args:
            data (dict): Data dictionary with wavelength, scattering, etc.

        Returns:
            dict: Analysis results
        """
        results = {}

        # Find peaks for each polarization
        peak_wavelengths, peak_values, peak_indices = self._find_peaks(
            data['wavelength'],
            data['scattering']
        )

        results['peak_wavelengths'] = peak_wavelengths
        results['peak_values'] = peak_values
        results['peak_indices'] = peak_indices

        # Calculate FWHM (Full Width at Half Maximum)
        fwhm_values = self._calculate_fwhm(
            data['wavelength'],
            data['scattering'],
            peak_indices
        )
        results['fwhm'] = fwhm_values

        # Calculate enhancement factors (if multiple polarizations)
        if data['n_polarizations'] > 1:
            enhancement = self._calculate_enhancement(data['scattering'])
            results['enhancement_factors'] = enhancement

        # Calculate average cross sections
        results['avg_scattering'] = np.mean(data['scattering'], axis=0)
        results['avg_extinction'] = np.mean(data['extinction'], axis=0)
        results['avg_absorption'] = np.mean(data['absorption'], axis=0)

        # Calculate max cross sections
        results['max_scattering'] = np.max(data['scattering'], axis=0)
        results['max_extinction'] = np.max(data['extinction'], axis=0)
        results['max_absorption'] = np.max(data['absorption'], axis=0)

        # ============================================================
        # UNPOLARIZED CALCULATION (FDTD-style incoherent averaging)
        # ============================================================
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
                print(f"  Unpolarized calculation: {unpolarized_info['method']}")
                print(f"    Peak wavelength: {unpol_data['peak_wavelength']:.1f} nm")
                print(f"    Peak absorption: {unpol_data['peak_absorption']:.2f} nm²")

        if self.verbose:
            print("Spectrum analysis complete:")
            print(f"  Peak wavelengths: {peak_wavelengths}")
            print(f"  Peak values: {peak_values}")
            print(f"  FWHM: {fwhm_values}")

        return results

    def _check_unpolarized_conditions(self, polarizations, excitation_type, n_polarizations):
        """
        Check if unpolarized calculation is possible.

        FDTD-style unpolarized calculation requires:
        - Plane wave: 2 orthogonal polarizations
        - Dipole: 3 orthogonal directions

        Returns:
            dict: Information about unpolarized calculation possibility
        """
        result = {
            'can_calculate': False,
            'method': None,
            'reason': None
        }

        # EELS doesn't support unpolarized
        if excitation_type == 'eels':
            result['reason'] = 'EELS excitation does not support unpolarized calculation'
            return result

        # Convert polarizations to numpy array
        if isinstance(polarizations, list):
            polarizations = np.array(polarizations)

        if len(polarizations) == 0:
            result['reason'] = 'No polarization data available'
            return result

        # Plane wave: need exactly 2 orthogonal polarizations
        if excitation_type == 'planewave':
            if n_polarizations != 2:
                result['reason'] = f'Plane wave unpolarized requires exactly 2 polarizations (got {n_polarizations})'
                return result

            if self._are_orthogonal(polarizations[0], polarizations[1]):
                result['can_calculate'] = True
                result['method'] = 'orthogonal_2pol_average'
                result['reason'] = 'Two orthogonal polarizations detected'
            else:
                result['reason'] = 'Polarizations are not orthogonal'

        # Dipole: need exactly 3 orthogonal directions
        elif excitation_type == 'dipole':
            if n_polarizations != 3:
                result['reason'] = f'Dipole unpolarized requires exactly 3 directions (got {n_polarizations})'
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

    def _are_orthogonal(self, v1, v2, tolerance=1e-6):
        """Check if two vectors are orthogonal."""
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)

        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < tolerance or norm2 < tolerance:
            return False

        v1_normalized = v1 / norm1
        v2_normalized = v2 / norm2

        dot_product = np.abs(np.dot(v1_normalized, v2_normalized))
        return dot_product < tolerance

    def _calculate_unpolarized_spectrum(self, data, unpolarized_info):
        """
        Calculate unpolarized spectrum using FDTD-style incoherent averaging.

        For plane wave (2 orthogonal polarizations):
            σ_unpol = (σ_pol1 + σ_pol2) / 2

        For dipole (3 orthogonal directions):
            σ_unpol = (σ_x + σ_y + σ_z) / 3
        """
        n_pol = data['n_polarizations']

        # Incoherent average (arithmetic mean of cross sections)
        unpol_scattering = np.mean(data['scattering'], axis=1)
        unpol_extinction = np.mean(data['extinction'], axis=1)
        unpol_absorption = np.mean(data['absorption'], axis=1)

        # Find peak in unpolarized absorption spectrum
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
    
    def _find_peaks(self, wavelength, cross_sections):
        """
        Find peak positions in spectrum.
        
        FIXED: Now handles complex cross_sections properly.
        """
        n_pol = cross_sections.shape[1]
        
        # ✅ FIX: Handle complex data by taking magnitude
        if np.iscomplexobj(cross_sections):
            if self.verbose:
                print("  Note: Converting complex cross_sections to magnitude")
            cross_sections = np.abs(cross_sections)
        
        peak_wavelengths = np.zeros(n_pol)
        peak_values = np.zeros(n_pol)
        peak_indices = np.zeros(n_pol, dtype=int)
        
        for i in range(n_pol):
            # Find peaks
            peaks, properties = signal.find_peaks(
                cross_sections[:, i],
                prominence=0.1 * np.max(cross_sections[:, i])
            )
            
            if len(peaks) > 0:
                # Get the highest peak
                max_peak_idx = peaks[np.argmax(cross_sections[peaks, i])]
                peak_indices[i] = max_peak_idx
                peak_wavelengths[i] = wavelength[max_peak_idx]
                peak_values[i] = cross_sections[max_peak_idx, i]
            else:
                # No peak found, use maximum value
                max_idx = np.argmax(cross_sections[:, i])
                peak_indices[i] = max_idx
                peak_wavelengths[i] = wavelength[max_idx]
                peak_values[i] = cross_sections[max_idx, i]
        
        return peak_wavelengths, peak_values, peak_indices
    
    def _calculate_fwhm(self, wavelength, cross_sections, peak_indices):
        """
        Calculate Full Width at Half Maximum.
        
        FIXED: Now handles complex cross_sections properly.
        """
        n_pol = cross_sections.shape[1]
        
        # ✅ FIX: Handle complex data
        if np.iscomplexobj(cross_sections):
            cross_sections = np.abs(cross_sections)
        
        fwhm_values = np.zeros(n_pol)
        
        for i in range(n_pol):
            peak_idx = peak_indices[i]
            peak_value = cross_sections[peak_idx, i]
            half_max = peak_value / 2
            
            # Find points where spectrum crosses half maximum
            above_half = cross_sections[:, i] > half_max
            
            # Find left edge
            left_indices = np.where(above_half[:peak_idx])[0]
            if len(left_indices) > 0:
                left_idx = left_indices[0]
            else:
                left_idx = 0
            
            # Find right edge
            right_indices = np.where(above_half[peak_idx:])[0]
            if len(right_indices) > 0:
                right_idx = peak_idx + right_indices[-1]
            else:
                right_idx = len(wavelength) - 1
            
            # Calculate FWHM
            fwhm_values[i] = wavelength[right_idx] - wavelength[left_idx]
        
        return fwhm_values
    
    def _calculate_enhancement(self, cross_sections):
        """Calculate enhancement factors between polarizations."""
        n_pol = cross_sections.shape[1]
        enhancement = {}
        
        # Calculate enhancement of first polarization relative to others
        if n_pol > 1:
            max_vals = np.max(cross_sections, axis=0)
            enhancement['pol1_vs_pol2'] = max_vals[0] / max_vals[1] if max_vals[1] > 0 else 0
            
            if n_pol > 2:
                enhancement['pol1_vs_pol3'] = max_vals[0] / max_vals[2] if max_vals[2] > 0 else 0
                enhancement['pol2_vs_pol3'] = max_vals[1] / max_vals[2] if max_vals[2] > 0 else 0
        
        return enhancement
