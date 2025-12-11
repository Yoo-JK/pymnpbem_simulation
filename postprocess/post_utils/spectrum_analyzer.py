"""
Spectrum Analyzer for pyMNPBEM simulation results.

Provides analysis of optical spectra:
- Peak finding (resonance wavelengths)
- FWHM calculation
- Enhancement factors
- Unpolarized spectrum calculation
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from typing import Dict, List, Any, Optional, Tuple


class SpectrumAnalyzer:
    """
    Analyzes optical spectra from plasmonic simulations.

    Features:
    - Peak detection and characterization
    - Full Width at Half Maximum (FWHM)
    - Quality factor estimation
    - Polarization analysis
    """

    def __init__(self, spectrum_data: Dict[str, np.ndarray]):
        """
        Initialize the spectrum analyzer.

        Args:
            spectrum_data: Dictionary with wavelengths and cross-sections
        """
        self.wavelengths = spectrum_data['wavelengths']
        self.scattering = spectrum_data.get('scattering')
        self.absorption = spectrum_data.get('absorption')
        self.extinction = spectrum_data.get('extinction')

        self.n_wavelengths = len(self.wavelengths)
        # Safely get number of polarizations with dimension check
        if self.extinction is not None and self.extinction.ndim > 1:
            self.n_polarizations = self.extinction.shape[1]
        elif self.extinction is not None:
            self.n_polarizations = 1
        else:
            self.n_polarizations = 0

    def find_peaks(self, spectrum_type: str = 'extinction',
                   pol_idx: int = 0,
                   prominence: float = 0.1,
                   min_distance: int = 5) -> List[Dict[str, float]]:
        """
        Find peaks in the spectrum.

        Args:
            spectrum_type: 'extinction', 'scattering', or 'absorption'
            pol_idx: Polarization index
            prominence: Minimum prominence of peaks (fraction of max)
            min_distance: Minimum distance between peaks (in points)

        Returns:
            List of peak dictionaries with wavelength, value, and index
        """
        spectrum = self._get_spectrum(spectrum_type, pol_idx)

        if spectrum is None:
            return []

        # Normalize spectrum for peak finding
        max_val = np.max(spectrum)
        if max_val == 0 or np.isnan(max_val):
            return []
        spectrum_norm = spectrum / max_val

        # Find peaks
        peak_indices, properties = find_peaks(
            spectrum_norm,
            prominence=prominence,
            distance=min_distance
        )

        peaks = []
        for i, idx in enumerate(peak_indices):
            # Safely get prominence - use enumerate index directly instead of np.where
            prominence_val = None
            if 'prominences' in properties and i < len(properties['prominences']):
                prominence_val = float(properties['prominences'][i])
            peaks.append({
                'wavelength': float(self.wavelengths[idx]),
                'value': float(spectrum[idx]),
                'index': int(idx),
                'prominence': prominence_val
            })

        # Sort by value (highest first)
        peaks.sort(key=lambda x: x['value'], reverse=True)

        return peaks

    def calculate_fwhm(self, spectrum_type: str = 'extinction',
                       pol_idx: int = 0,
                       peak_idx: int = 0) -> Optional[Dict[str, float]]:
        """
        Calculate Full Width at Half Maximum for a peak.

        Args:
            spectrum_type: 'extinction', 'scattering', or 'absorption'
            pol_idx: Polarization index
            peak_idx: Index of peak to analyze (from find_peaks)

        Returns:
            Dictionary with FWHM, peak wavelength, and half-max wavelengths
        """
        spectrum = self._get_spectrum(spectrum_type, pol_idx)
        if spectrum is None:
            return None

        peaks = self.find_peaks(spectrum_type, pol_idx)
        if peak_idx >= len(peaks):
            return None

        peak = peaks[peak_idx]
        peak_wavelength = peak['wavelength']
        peak_value = peak['value']
        peak_index = peak['index']

        half_max = peak_value / 2

        # Find left half-max point
        left_idx = peak_index
        while left_idx > 0 and spectrum[left_idx] > half_max:
            left_idx -= 1

        # Interpolate for precise left wavelength
        if left_idx > 0:
            left_wl = np.interp(half_max,
                                [spectrum[left_idx], spectrum[left_idx + 1]],
                                [self.wavelengths[left_idx], self.wavelengths[left_idx + 1]])
        else:
            left_wl = self.wavelengths[0]

        # Find right half-max point
        right_idx = peak_index
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
            right_idx += 1

        # Interpolate for precise right wavelength
        if right_idx < len(spectrum) - 1:
            right_wl = np.interp(half_max,
                                 [spectrum[right_idx], spectrum[right_idx - 1]],
                                 [self.wavelengths[right_idx], self.wavelengths[right_idx - 1]])
        else:
            right_wl = self.wavelengths[-1]

        fwhm = right_wl - left_wl

        # Quality factor
        q_factor = peak_wavelength / fwhm if fwhm > 0 else float('inf')

        return {
            'fwhm': float(fwhm),
            'peak_wavelength': float(peak_wavelength),
            'half_max_left': float(left_wl),
            'half_max_right': float(right_wl),
            'quality_factor': float(q_factor),
            'peak_value': float(peak_value),
        }

    def calculate_unpolarized(self) -> Dict[str, np.ndarray]:
        """
        Calculate unpolarized spectrum from orthogonal polarizations.

        Uses FDTD-style incoherent averaging:
        sigma_unpol = (sigma_pol1 + sigma_pol2) / 2

        Returns:
            Dictionary with unpolarized spectra
        """
        if self.n_polarizations < 2:
            # Helper to safely get first column or 1D array
            def get_first(arr):
                if arr is None:
                    return None
                if arr.ndim == 1:
                    return arr
                if arr.shape[1] > 0:
                    return arr[:, 0]
                return None
            return {
                'wavelengths': self.wavelengths,
                'scattering_unpolarized': get_first(self.scattering),
                'absorption_unpolarized': get_first(self.absorption),
                'extinction_unpolarized': get_first(self.extinction),
            }

        result = {'wavelengths': self.wavelengths}

        # Helper to safely average across polarizations
        def safe_mean(arr):
            if arr is None:
                return None
            if arr.ndim > 1:
                return np.mean(arr, axis=1)
            return arr  # Already 1D

        if self.scattering is not None:
            result['scattering_unpolarized'] = safe_mean(self.scattering)

        if self.absorption is not None:
            result['absorption_unpolarized'] = safe_mean(self.absorption)

        if self.extinction is not None:
            result['extinction_unpolarized'] = safe_mean(self.extinction)

        return result

    def get_statistics(self, spectrum_type: str = 'extinction',
                       pol_idx: int = 0) -> Dict[str, float]:
        """
        Get statistical measures of a spectrum.

        Args:
            spectrum_type: Type of spectrum
            pol_idx: Polarization index

        Returns:
            Dictionary with statistics
        """
        spectrum = self._get_spectrum(spectrum_type, pol_idx)

        if spectrum is None:
            return {}

        return {
            'max': float(np.max(spectrum)),
            'min': float(np.min(spectrum)),
            'mean': float(np.mean(spectrum)),
            'std': float(np.std(spectrum)),
            'integral': float(np.trapz(spectrum, self.wavelengths)),
        }

    def get_resonance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all resonances in the spectra.

        Returns:
            Dictionary with resonance information for all polarizations
        """
        # Safe wavelength range access
        if len(self.wavelengths) > 0:
            wl_range = [float(self.wavelengths[0]), float(self.wavelengths[-1])]
        else:
            wl_range = [0.0, 0.0]

        summary = {
            'n_polarizations': self.n_polarizations,
            'wavelength_range': wl_range,
            'resonances': {}
        }

        for pol_idx in range(self.n_polarizations):
            peaks = self.find_peaks('extinction', pol_idx)

            pol_summary = {
                'n_peaks': len(peaks),
                'peaks': []
            }

            for i, peak in enumerate(peaks[:3]):  # Top 3 peaks
                fwhm_data = self.calculate_fwhm('extinction', pol_idx, i)
                pol_summary['peaks'].append({
                    'wavelength': peak['wavelength'],
                    'extinction': peak['value'],
                    'fwhm': fwhm_data['fwhm'] if fwhm_data else None,
                    'quality_factor': fwhm_data['quality_factor'] if fwhm_data else None,
                })

            summary['resonances'][f'pol_{pol_idx+1}'] = pol_summary

        return summary

    def _get_spectrum(self, spectrum_type: str, pol_idx: int) -> Optional[np.ndarray]:
        """Get spectrum array by type and polarization."""
        if spectrum_type == 'extinction':
            data = self.extinction
        elif spectrum_type == 'scattering':
            data = self.scattering
        elif spectrum_type == 'absorption':
            data = self.absorption
        else:
            return None

        if data is None:
            return None

        # Handle both 1D and 2D arrays
        if data.ndim == 1:
            return data if pol_idx == 0 else None

        if pol_idx >= data.shape[1]:
            return None

        return data[:, pol_idx]

    def interpolate_spectrum(self, wavelengths_new: np.ndarray,
                             spectrum_type: str = 'extinction',
                             pol_idx: int = 0) -> np.ndarray:
        """
        Interpolate spectrum to new wavelength grid.

        Args:
            wavelengths_new: New wavelength array
            spectrum_type: Type of spectrum
            pol_idx: Polarization index

        Returns:
            Interpolated spectrum array
        """
        spectrum = self._get_spectrum(spectrum_type, pol_idx)

        if spectrum is None:
            return np.zeros_like(wavelengths_new)

        interp_func = interp1d(self.wavelengths, spectrum,
                               kind='cubic', bounds_error=False,
                               fill_value='extrapolate')

        return interp_func(wavelengths_new)
