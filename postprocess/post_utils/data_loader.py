"""
Data Loader for pyMNPBEM simulation results.

Loads simulation results from:
- NumPy .npy files
- JSON summary files
- Text data files
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class DataLoader:
    """
    Loads simulation results from a run folder.

    Supports loading:
    - Spectrum data (wavelengths, scattering, absorption, extinction)
    - Field data (enhancement, coordinates)
    - Surface charge data (positions, values, mesh)
    - Configuration and summary data
    """

    def __init__(self, run_folder: str):
        """
        Initialize the data loader.

        Args:
            run_folder: Path to simulation run folder
        """
        self.run_folder = run_folder
        self.data_dir = os.path.join(run_folder, 'data')

        if not os.path.exists(run_folder):
            raise FileNotFoundError(f"Run folder not found: {run_folder}")

    def load_all(self) -> Dict[str, Any]:
        """
        Load all available data from the run folder.

        Returns:
            Dictionary with all loaded data
        """
        data = {}

        # Load configuration
        data['config'] = self.load_config()

        # Load summary
        data['summary'] = self.load_summary()

        # Load spectrum data
        spectrum = self.load_spectrum()
        if spectrum:
            data['spectrum'] = spectrum

        # Load field data
        field = self.load_field_data()
        if field:
            data['field'] = field

        # Load surface charge data
        charges = self.load_surface_charges()
        if charges:
            data['surface_charges'] = charges

        return data

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = os.path.join(self.run_folder, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def load_summary(self) -> Dict[str, Any]:
        """Load summary from summary.json."""
        summary_path = os.path.join(self.run_folder, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        return {}

    def load_spectrum(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load spectrum data.

        Returns:
            Dictionary with wavelengths and cross-sections, or None if not available
        """
        wavelengths_path = os.path.join(self.data_dir, 'wavelengths.npy')

        if not os.path.exists(wavelengths_path):
            return None

        data = {
            'wavelengths': np.load(wavelengths_path),
        }

        # Load cross-section data
        for name in ['scattering', 'absorption', 'extinction']:
            path = os.path.join(self.data_dir, f'{name}.npy')
            if os.path.exists(path):
                data[name] = np.load(path)

        # Load unpolarized if available
        unpol_path = os.path.join(self.data_dir, 'extinction_unpolarized.npy')
        if os.path.exists(unpol_path):
            data['extinction_unpolarized'] = np.load(unpol_path)

        return data

    def load_field_data(self) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Load field data for all polarizations.

        Returns:
            Dictionary mapping polarization index to field data
        """
        field_data = {}

        # Find all field files
        for filename in os.listdir(self.data_dir):
            if filename.startswith('field_pol') and filename.endswith('_enhancement.npy'):
                # Extract polarization index
                parts = filename.split('_')
                pol_str = parts[1]  # 'pol1', 'pol2', etc.
                pol_idx = int(pol_str[3:]) - 1  # Convert to 0-indexed

                prefix = f'field_pol{pol_idx+1}'

                field_data[pol_idx] = {
                    'enhancement': np.load(os.path.join(self.data_dir, f'{prefix}_enhancement.npy')),
                    'x': np.load(os.path.join(self.data_dir, f'{prefix}_x.npy')),
                    'y': np.load(os.path.join(self.data_dir, f'{prefix}_y.npy')),
                    'z': np.load(os.path.join(self.data_dir, f'{prefix}_z.npy')),
                }

        return field_data if field_data else None

    def load_surface_charges(self) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Load surface charge data for all polarizations.

        Returns:
            Dictionary mapping polarization index to charge data
        """
        charge_data = {}

        for filename in os.listdir(self.data_dir):
            if filename.startswith('charges_pol') and filename.endswith('_values.npy'):
                parts = filename.split('_')
                pol_str = parts[1]
                pol_idx = int(pol_str[3:]) - 1

                prefix = f'charges_pol{pol_idx+1}'

                charges = np.load(os.path.join(self.data_dir, f'{prefix}_values.npy'))

                # Compute derived charge values (real, imag, magnitude, phase)
                charge_real = np.real(charges)
                charge_imag = np.imag(charges)
                charge_magnitude = np.abs(charges)
                charge_phase = np.angle(charges)

                charge_data[pol_idx] = {
                    'positions': np.load(os.path.join(self.data_dir, f'{prefix}_positions.npy')),
                    'charges': charges,
                    'charge_real': charge_real,
                    'charge_imag': charge_imag,
                    'charge_magnitude': charge_magnitude,
                    'charge_phase': charge_phase,
                    'vertices': np.load(os.path.join(self.data_dir, f'{prefix}_vertices.npy')),
                    'faces': np.load(os.path.join(self.data_dir, f'{prefix}_faces.npy')),
                }

        return charge_data if charge_data else None

    def load_spectrum_txt(self, pol_idx: int = 0) -> Optional[Dict[str, np.ndarray]]:
        """
        Load spectrum data from text file.

        Args:
            pol_idx: Polarization index (0-indexed)

        Returns:
            Dictionary with spectrum data
        """
        filepath = os.path.join(self.data_dir, f'spectrum_pol{pol_idx+1}.txt')

        if not os.path.exists(filepath):
            return None

        data = np.loadtxt(filepath, skiprows=1)

        return {
            'wavelengths': data[:, 0],
            'scattering': data[:, 1],
            'absorption': data[:, 2],
            'extinction': data[:, 3],
        }

    def get_polarization_count(self) -> int:
        """Get the number of polarizations in the data."""
        spectrum = self.load_spectrum()
        if spectrum and 'scattering' in spectrum:
            return spectrum['scattering'].shape[1]
        return 0

    def get_wavelength_range(self) -> Optional[tuple]:
        """Get the wavelength range of the simulation."""
        spectrum = self.load_spectrum()
        if spectrum and 'wavelengths' in spectrum:
            wavelengths = spectrum['wavelengths']
            return (float(wavelengths[0]), float(wavelengths[-1]))
        return None
