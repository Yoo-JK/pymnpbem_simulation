import os
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


class RefractiveIndexLoader(object):

    def __init__(self,
            filepath: str,
            verbose: bool = False) -> None:

        self.filepath = Path(filepath)
        self.verbose = verbose

        if not self.filepath.exists():
            raise FileNotFoundError(
                '[error] Refractive index file not found: {}'.format(filepath))

        self.wavelengths, self.n_values, self.k_values = self._load_file()

        if self.verbose:
            print('[info] Loaded refractive index data from: {}'.format(self.filepath))
            print('  Wavelength range: {:.1f} - {:.1f} nm'.format(
                self.wavelengths[0], self.wavelengths[-1]))
            print('  Number of points: {}'.format(len(self.wavelengths)))

    def _load_file(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            data = np.loadtxt(self.filepath, comments = '#')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            n_cols = data.shape[1]

            if n_cols == 2:
                # wavelength(nm), n
                wavelengths = data[:, 0]
                n_values = data[:, 1]
                k_values = np.zeros_like(n_values)

                if self.verbose:
                    print('  Format: wavelength(nm), n (no absorption)')

            elif n_cols == 3:
                # wavelength(nm), n, k
                wavelengths = data[:, 0]
                n_values = data[:, 1]
                k_values = data[:, 2]

                if self.verbose:
                    print('  Format: wavelength(nm), n, k')

            else:
                raise ValueError(
                    '[error] Unsupported file format: {} columns found. '
                    'Expected 2 or 3.'.format(n_cols))

            # sort by wavelength (required for interpolation)
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            n_values = n_values[sort_idx]
            k_values = k_values[sort_idx]

            return wavelengths, n_values, k_values

        except Exception as e:
            raise RuntimeError(
                '[error] Error loading refractive index file: {}'.format(e))

    def interpolate(self,
            target_wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        target_wavelengths = np.asarray(target_wavelengths)

        cs_n = CubicSpline(self.wavelengths, self.n_values)
        n_interp = cs_n(target_wavelengths)

        cs_k = CubicSpline(self.wavelengths, self.k_values)
        k_interp = cs_k(target_wavelengths)

        if self.verbose:
            print('[info] Interpolated refractive index at {} wavelengths'.format(
                len(target_wavelengths)))
            print('  Interpolation method: Cubic Spline')
            if (target_wavelengths.min() < self.wavelengths.min()
                    or target_wavelengths.max() > self.wavelengths.max()):
                print('  WARNING: Some wavelengths outside data range (extrapolation used)')

        return n_interp, k_interp

    def get_epsilon(self,
            target_wavelengths: np.ndarray) -> np.ndarray:

        n_interp, k_interp = self.interpolate(target_wavelengths)

        # epsilon = (n + ik)^2
        refractive_index = n_interp + 1j * k_interp
        epsilon = refractive_index ** 2

        return epsilon


def load_and_interpolate(
        filepath: str,
        target_wavelengths: np.ndarray,
        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    loader = RefractiveIndexLoader(filepath, verbose = verbose)
    n_interp, k_interp = loader.interpolate(target_wavelengths)
    epsilon = loader.get_epsilon(target_wavelengths)

    return n_interp, k_interp, epsilon
