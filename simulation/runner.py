"""
Simulation Runner for pyMNPBEM-based plasmonic simulations.

Main orchestrator that coordinates:
- Geometry building
- Material setup
- BEM solver execution
- Field calculations
- Surface charge analysis
- Data saving
"""

import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

from .sim_utils import (
    GeometryBuilder,
    MaterialBuilder,
    BEMSolver,
    FieldCalculator,
    SurfaceChargeCalculator
)


class SimulationRunner:
    """
    Main simulation runner for pyMNPBEM plasmonic calculations.

    Manages the complete simulation workflow:
    1. Load configurations
    2. Build particle geometry
    3. Set up materials
    4. Run BEM simulation
    5. Compute optical spectra
    6. Calculate field distributions
    7. Analyze surface charges
    8. Save results
    """

    def __init__(self, structure_config: Dict[str, Any], sim_config: Dict[str, Any],
                 pymnpbem_path: Optional[str] = None):
        """
        Initialize the simulation runner.

        Args:
            structure_config: Dictionary with structure parameters
            sim_config: Dictionary with simulation parameters
            pymnpbem_path: Path to pyMNPBEM installation
        """
        self.structure_config = structure_config
        self.sim_config = sim_config

        # Determine pyMNPBEM path
        if pymnpbem_path is None:
            # Try common locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'pyMNPBEM'),
                os.path.expanduser('~/pyMNPBEM'),
                '/opt/pyMNPBEM',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pymnpbem_path = path
                    break

        self.pymnpbem_path = pymnpbem_path
        if pymnpbem_path:
            sys.path.insert(0, pymnpbem_path)

        # Initialize components
        self.geometry_builder = GeometryBuilder(structure_config, pymnpbem_path)
        self.material_builder = MaterialBuilder(structure_config, pymnpbem_path)
        self.bem_solver = BEMSolver(sim_config, pymnpbem_path)
        self.field_calculator = FieldCalculator(sim_config, pymnpbem_path)
        self.surface_charge_calculator = SurfaceChargeCalculator(sim_config, pymnpbem_path)

        # Results storage
        self.results = {}
        self.run_folder = None

    def create_run_folder(self) -> str:
        """
        Create a unique folder for this simulation run.

        Returns:
            Path to the run folder
        """
        output_dir = self.sim_config.get('output_dir', './results')
        sim_name = self.sim_config.get('simulation_name', 'simulation')
        structure_name = self.structure_config.get('structure_name', 'structure')

        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create folder name
        folder_name = f"{sim_name}_{structure_name}_{timestamp}"
        self.run_folder = os.path.join(output_dir, folder_name)

        # Create directories
        os.makedirs(self.run_folder, exist_ok=True)
        os.makedirs(os.path.join(self.run_folder, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.run_folder, 'data'), exist_ok=True)

        # Save configuration
        self._save_config()

        return self.run_folder

    def _save_config(self):
        """Save configuration to run folder."""
        if self.run_folder is None:
            return

        config = {
            'structure': self.structure_config,
            'simulation': self.sim_config,
            'timestamp': datetime.now().isoformat(),
        }

        config_path = os.path.join(self.run_folder, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def run(self, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run the complete simulation.

        Args:
            show_progress: Whether to show progress information

        Returns:
            Dictionary with all simulation results
        """
        if show_progress:
            print("=" * 60)
            print("Starting pyMNPBEM Simulation")
            print("=" * 60)

        # Create run folder
        if self.run_folder is None:
            self.create_run_folder()

        # Step 1: Build geometry
        if show_progress:
            print("\n[1/6] Building particle geometry...")

        particles, inout = self.geometry_builder.build()
        self.results['geometry'] = {
            'n_particles': len(particles),
            'inout': inout,
            'bounds': self.geometry_builder.get_particle_bounds(particles),
            'info': self.geometry_builder.get_structure_info(),
        }

        if show_progress:
            print(f"      Created {len(particles)} particle(s)")

        # Step 2: Build materials
        if show_progress:
            print("\n[2/6] Setting up materials...")

        epstab = self.material_builder.build()
        substrate = self.material_builder.build_substrate()
        self.results['materials'] = {
            'n_materials': len(epstab),
            'info': self.material_builder.get_material_info(),
            'has_substrate': substrate is not None,
        }

        if show_progress:
            print(f"      Set up {len(epstab)} material(s)")

        # Step 3: Set up BEM solver
        if show_progress:
            print("\n[3/6] Setting up BEM solver...")

        self.bem_solver.setup(particles, inout, epstab, substrate)

        if show_progress:
            solver_info = self.bem_solver.get_solver_info()
            print(f"      Solver type: {solver_info['simulation_type']}")
            print(f"      Wavelength range: {solver_info['wavelength_range']} nm")
            print(f"      Number of excitations: {solver_info['n_excitations']}")

        # Step 4: Compute spectra
        if self.sim_config.get('calculate_cross_sections', True):
            if show_progress:
                print("\n[4/6] Computing optical spectra...")

            spectrum_data = self.bem_solver.compute_spectrum(show_progress)
            self.results['spectrum'] = spectrum_data

            # Compute unpolarized if multiple polarizations
            if spectrum_data['scattering'].shape[1] > 1:
                unpolarized = self.bem_solver.compute_unpolarized(spectrum_data)
                self.results['spectrum_unpolarized'] = unpolarized

            if show_progress:
                # Find and report peaks
                for i in range(spectrum_data['scattering'].shape[1]):
                    peak_wl = self.bem_solver.get_peak_wavelength(spectrum_data, i, 'extinction')
                    print(f"      Peak extinction (pol {i+1}): {peak_wl:.1f} nm")
        else:
            if show_progress:
                print("\n[4/6] Skipping spectrum calculation...")

        # Step 5: Calculate fields
        if self.sim_config.get('calculate_fields', False):
            if show_progress:
                print("\n[5/6] Computing electric field distributions...")

            if 'spectrum' in self.results:
                field_data = self.field_calculator.compute_field_at_peaks(
                    self.bem_solver, self.results['spectrum'], show_progress
                )
                self.results['field'] = field_data

                # Compute unpolarized field if multiple polarizations
                if len(field_data) > 1:
                    unpolarized_field = self.field_calculator.compute_unpolarized_field(field_data)
                    self.results['field_unpolarized'] = unpolarized_field

                # Find hotspots
                for exc_idx, data in field_data.items():
                    hotspots = self.field_calculator.find_hotspots(data)
                    self.results.setdefault('hotspots', {})[exc_idx] = hotspots
                    stats = self.field_calculator.get_field_statistics(data)
                    self.results.setdefault('field_stats', {})[exc_idx] = stats

                    if show_progress:
                        print(f"      Max enhancement (pol {exc_idx+1}): {stats['max']:.1f}")
        else:
            if show_progress:
                print("\n[5/6] Skipping field calculation...")

        # Step 6: Surface charge analysis
        if self.sim_config.get('calculate_surface_charges', True):
            if show_progress:
                print("\n[6/6] Analyzing surface charge distributions...")

            if 'spectrum' in self.results:
                charge_data = self.surface_charge_calculator.compute_charges_at_peaks(
                    self.bem_solver, self.results['spectrum'], show_progress
                )
                self.results['surface_charges'] = charge_data

                # Mode analysis
                for exc_idx, data in charge_data.items():
                    mode_info = self.surface_charge_calculator.identify_mode(data)
                    self.results.setdefault('mode_analysis', {})[exc_idx] = mode_info

                    if show_progress:
                        print(f"      Dominant mode (pol {exc_idx+1}): {mode_info['dominant_mode']}")
        else:
            if show_progress:
                print("\n[6/6] Skipping surface charge analysis...")

        # Save results
        if show_progress:
            print("\n" + "=" * 60)
            print("Saving results...")

        self._save_results()

        if show_progress:
            print(f"Results saved to: {self.run_folder}")
            print("=" * 60)

        return self.results

    def _save_results(self):
        """Save all results to files."""
        if self.run_folder is None:
            return

        data_dir = os.path.join(self.run_folder, 'data')

        # Save spectrum data
        if 'spectrum' in self.results:
            spectrum = self.results['spectrum']

            # Save as numpy files
            np.save(os.path.join(data_dir, 'wavelengths.npy'), spectrum['wavelengths'])
            np.save(os.path.join(data_dir, 'scattering.npy'), spectrum['scattering'])
            np.save(os.path.join(data_dir, 'absorption.npy'), spectrum['absorption'])
            np.save(os.path.join(data_dir, 'extinction.npy'), spectrum['extinction'])

            # Save as text file
            self._save_spectrum_txt(spectrum)

        # Save unpolarized spectrum
        if 'spectrum_unpolarized' in self.results:
            unpol = self.results['spectrum_unpolarized']
            np.save(os.path.join(data_dir, 'extinction_unpolarized.npy'),
                    unpol['extinction_unpolarized'])

        # Save field data
        if 'field' in self.results:
            for exc_idx, field_data in self.results['field'].items():
                prefix = f'field_pol{exc_idx+1}'
                np.save(os.path.join(data_dir, f'{prefix}_enhancement.npy'),
                        field_data['enhancement'])
                np.save(os.path.join(data_dir, f'{prefix}_x.npy'), field_data['x'])
                np.save(os.path.join(data_dir, f'{prefix}_y.npy'), field_data['y'])
                np.save(os.path.join(data_dir, f'{prefix}_z.npy'), field_data['z'])

        # Save surface charge data
        if 'surface_charges' in self.results:
            for exc_idx, charge_data in self.results['surface_charges'].items():
                prefix = f'charges_pol{exc_idx+1}'
                np.save(os.path.join(data_dir, f'{prefix}_positions.npy'),
                        charge_data['positions'])
                np.save(os.path.join(data_dir, f'{prefix}_values.npy'),
                        charge_data['charges'])
                np.save(os.path.join(data_dir, f'{prefix}_vertices.npy'),
                        charge_data['vertices'])
                np.save(os.path.join(data_dir, f'{prefix}_faces.npy'),
                        charge_data['faces'])

        # Save summary JSON
        summary = {
            'geometry': self.results.get('geometry', {}),
            'materials': self.results.get('materials', {}),
            'hotspots': {str(k): v for k, v in self.results.get('hotspots', {}).items()},
            'field_stats': {str(k): v for k, v in self.results.get('field_stats', {}).items()},
            'mode_analysis': {str(k): v for k, v in self.results.get('mode_analysis', {}).items()},
        }

        with open(os.path.join(self.run_folder, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def _save_spectrum_txt(self, spectrum: Dict[str, np.ndarray]):
        """Save spectrum data as text file."""
        data_dir = os.path.join(self.run_folder, 'data')
        wavelengths = spectrum['wavelengths']
        n_exc = spectrum['scattering'].shape[1]

        for i in range(n_exc):
            filepath = os.path.join(data_dir, f'spectrum_pol{i+1}.txt')
            with open(filepath, 'w') as f:
                f.write("# Wavelength(nm)\tScattering(nm^2)\tAbsorption(nm^2)\tExtinction(nm^2)\n")
                for j, wl in enumerate(wavelengths):
                    sca = spectrum['scattering'][j, i]
                    abs_ = spectrum['absorption'][j, i]
                    ext = spectrum['extinction'][j, i]
                    f.write(f"{wl:.2f}\t{sca:.6e}\t{abs_:.6e}\t{ext:.6e}\n")

    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        return self.results

    def get_run_folder(self) -> Optional[str]:
        """Get the run folder path."""
        return self.run_folder
