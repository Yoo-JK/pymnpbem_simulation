"""
Simulation Manager for pyMNPBEM-based plasmonic simulations.

This module provides the SimulationManager class that orchestrates
BEM simulations using pyMNPBEM (Python implementation of MNPBEM).

Replaces the original MATLAB code generation with direct Python execution.
"""

import numpy as np
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from .sim_utils import (
    GeometryBuilder,
    MaterialBuilder,
    BEMSolver,
    FieldCalculator,
    SurfaceChargeCalculator,
    NonlocalHandler
)


class SimulationManager:
    """
    Manager class for pyMNPBEM simulations.

    Replaces MATLAB code generation with direct Python pyMNPBEM execution.
    Maintains the same interface as the original MATLAB-based system.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the simulation manager.

        Args:
            config: Merged configuration dictionary (structure + simulation settings)
            verbose: Whether to print progress information
        """
        self.config = config
        self.verbose = verbose
        self.run_folder = None
        self.results = {}

        # Get pyMNPBEM path from config
        self.pymnpbem_path = config.get('pymnpbem_path')
        if self.pymnpbem_path:
            sys.path.insert(0, self.pymnpbem_path)

        # Initialize component builders
        self.geometry_builder = GeometryBuilder(config, self.pymnpbem_path)
        self.material_builder = MaterialBuilder(config, self.pymnpbem_path)
        self.bem_solver = BEMSolver(config, self.pymnpbem_path)
        self.field_calculator = FieldCalculator(config, self.pymnpbem_path)
        self.surface_charge_calculator = SurfaceChargeCalculator(config, self.pymnpbem_path)
        self.nonlocal_handler = NonlocalHandler(config, self.pymnpbem_path)

    def create_run_folder(self) -> str:
        """
        Create the output folder for this simulation run.

        Returns:
            Path to the run folder
        """
        output_dir = self.config.get('output_dir', './results')

        # Get run folder name from config or generate one
        run_name = self.config.get('run_name')
        if not run_name:
            structure = self.config.get('structure', 'particle')
            sim_type = self.config.get('simulation_type', 'stat')
            run_name = f"{structure}_{sim_type}"

        self.run_folder = os.path.join(output_dir, run_name)

        # Create directories (overwrite if exists, matching original behavior)
        os.makedirs(self.run_folder, exist_ok=True)
        os.makedirs(os.path.join(self.run_folder, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.run_folder, 'plots'), exist_ok=True)

        if self.verbose:
            print(f"Created run folder: {self.run_folder}")

        return self.run_folder

    def save_config_snapshot(self):
        """Save configuration snapshot to run folder."""
        if self.run_folder is None:
            raise RuntimeError("Must call create_run_folder() first")

        config_path = os.path.join(self.run_folder, 'config.json')

        # Prepare config for JSON serialization
        config_copy = self._prepare_config_for_json(self.config)
        config_copy['_timestamp'] = datetime.now().isoformat()
        config_copy['_pymnpbem_version'] = 'python'

        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2, default=str)

        if self.verbose:
            print(f"Saved configuration to: {config_path}")

    def _prepare_config_for_json(self, config: Dict) -> Dict:
        """Convert config to JSON-serializable format."""
        result = {}
        for key, value in config.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, dict):
                result[key] = self._prepare_config_for_json(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def run_simulation(self):
        """
        Run the complete BEM simulation.

        This replaces MATLAB code generation and execution with direct
        Python pyMNPBEM calls.
        """
        if self.verbose:
            print("=" * 60)
            print("Starting pyMNPBEM Simulation")
            print("=" * 60)

        # Step 1: Build particle geometry
        if self.verbose:
            print("\n[1/6] Building particle geometry...")

        particles, inout = self.geometry_builder.build()
        self.results['geometry'] = {
            'n_particles': len(particles),
            'inout': inout,
            'bounds': self.geometry_builder.get_particle_bounds(particles),
            'info': self.geometry_builder.get_structure_info(),
        }

        if self.verbose:
            print(f"      Created {len(particles)} particle(s)")
            structure_info = self.results['geometry']['info']
            print(f"      Structure: {structure_info.get('structure_type', 'unknown')}")

        # Step 2: Build materials/dielectric functions
        if self.verbose:
            print("\n[2/6] Setting up materials...")

        epstab = self.material_builder.build()
        substrate = self.material_builder.build_substrate()
        self.results['materials'] = {
            'n_materials': len(epstab),
            'info': self.material_builder.get_material_info(),
            'has_substrate': substrate is not None,
        }

        if self.verbose:
            mat_info = self.results['materials']['info']
            print(f"      Materials: {mat_info.get('materials', [])}")
            print(f"      Medium: {mat_info.get('medium', 'unknown')}")

        # Step 2.5: Apply nonlocal corrections if enabled
        if self.nonlocal_handler.is_needed():
            if self.verbose:
                print("\n[2.5/6] Applying nonlocal quantum corrections...")

            materials_list = self.config.get('materials', [])
            particles, inout, extra_eps = self.nonlocal_handler.modify_particles_for_nonlocal(
                particles, inout, materials_list
            )

            # Add artificial epsilon functions to epstab
            epstab.extend(extra_eps)

            self.results['nonlocal'] = {
                'enabled': True,
                'model': self.nonlocal_handler.model,
                'cover_thickness_nm': self.nonlocal_handler.cover_thickness,
                'n_cover_layers': len(extra_eps),
            }

            if self.verbose:
                print(f"      Model: {self.nonlocal_handler.model}")
                print(f"      Cover layer thickness: {self.nonlocal_handler.cover_thickness} nm")
                print(f"      Added {len(extra_eps)} artificial layer(s)")
        else:
            self.results['nonlocal'] = {'enabled': False}

        # Step 3: Set up BEM solver
        if self.verbose:
            print("\n[3/6] Setting up BEM solver...")

        self.bem_solver.setup(particles, inout, epstab, substrate)

        if self.verbose:
            solver_info = self.bem_solver.get_solver_info()
            print(f"      Solver type: {solver_info['simulation_type']}")
            wl_range = solver_info.get('wavelength_range', [])
            if wl_range:
                print(f"      Wavelength range: {wl_range[0]:.0f} - {wl_range[1]:.0f} nm")
            print(f"      Excitations: {solver_info['n_excitations']}")

        # Step 4: Compute optical spectra
        if self.config.get('calculate_cross_sections', True):
            if self.verbose:
                print("\n[4/6] Computing optical spectra...")

            spectrum_data = self.bem_solver.compute_spectrum(show_progress=self.verbose)
            self.results['spectrum'] = spectrum_data

            # Compute unpolarized if multiple polarizations
            if spectrum_data['scattering'].shape[1] > 1:
                unpolarized = self.bem_solver.compute_unpolarized(spectrum_data)
                self.results['spectrum_unpolarized'] = unpolarized

            if self.verbose:
                # Report peak wavelengths
                for i in range(spectrum_data['scattering'].shape[1]):
                    peak_wl = self.bem_solver.get_peak_wavelength(spectrum_data, i, 'extinction')
                    print(f"      Peak extinction (pol {i+1}): {peak_wl:.1f} nm")
        else:
            if self.verbose:
                print("\n[4/6] Skipping spectrum calculation...")

        # Step 5: Calculate electric fields
        if self.config.get('calculate_fields', False):
            if self.verbose:
                print("\n[5/6] Computing electric field distributions...")

            if 'spectrum' in self.results:
                field_data = self.field_calculator.compute_field_at_peaks(
                    self.bem_solver, self.results['spectrum'], show_progress=self.verbose
                )
                self.results['field'] = field_data

                # Compute field statistics
                for exc_idx, data in field_data.items():
                    stats = self.field_calculator.get_field_statistics(data)
                    self.results.setdefault('field_stats', {})[exc_idx] = stats
                    hotspots = self.field_calculator.find_hotspots(data)
                    self.results.setdefault('hotspots', {})[exc_idx] = hotspots

                    if self.verbose:
                        print(f"      Max enhancement (pol {exc_idx+1}): {stats['max']:.1f}")
        else:
            if self.verbose:
                print("\n[5/6] Skipping field calculation...")

        # Step 6: Surface charge analysis
        if self.config.get('calculate_surface_charges', True):
            if self.verbose:
                print("\n[6/6] Analyzing surface charge distributions...")

            if 'spectrum' in self.results:
                charge_data = self.surface_charge_calculator.compute_charges_at_peaks(
                    self.bem_solver, self.results['spectrum'], show_progress=self.verbose
                )
                self.results['surface_charges'] = charge_data

                # Mode analysis
                for exc_idx, data in charge_data.items():
                    mode_info = self.surface_charge_calculator.identify_mode(data)
                    self.results.setdefault('mode_analysis', {})[exc_idx] = mode_info

                    if self.verbose:
                        print(f"      Dominant mode (pol {exc_idx+1}): {mode_info['dominant_mode']}")
        else:
            if self.verbose:
                print("\n[6/6] Skipping surface charge analysis...")

        if self.verbose:
            print("\n" + "=" * 60)
            print("Simulation completed successfully")
            print("=" * 60)

    def save_results(self):
        """Save all simulation results to files."""
        if self.run_folder is None:
            raise RuntimeError("Must call create_run_folder() first")

        data_dir = os.path.join(self.run_folder, 'data')

        if self.verbose:
            print("Saving results...")

        # Save spectrum data
        if 'spectrum' in self.results:
            spectrum = self.results['spectrum']

            # Numpy files
            np.save(os.path.join(data_dir, 'wavelengths.npy'), spectrum['wavelengths'])
            np.save(os.path.join(data_dir, 'scattering.npy'), spectrum['scattering'])
            np.save(os.path.join(data_dir, 'absorption.npy'), spectrum['absorption'])
            np.save(os.path.join(data_dir, 'extinction.npy'), spectrum['extinction'])

            # Text files (per polarization)
            self._save_spectrum_txt(spectrum)

            if self.verbose:
                print(f"      Saved spectrum data ({len(spectrum['wavelengths'])} wavelengths)")

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

            if self.verbose:
                print(f"      Saved field data ({len(self.results['field'])} polarization(s))")

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

            if self.verbose:
                print(f"      Saved surface charge data ({len(self.results['surface_charges'])} polarization(s))")

        # Save summary JSON
        summary = self._build_summary()
        summary_path = os.path.join(self.run_folder, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        if self.verbose:
            print(f"      Saved summary to: {summary_path}")

    def _save_spectrum_txt(self, spectrum: Dict[str, np.ndarray]):
        """Save spectrum data as text files."""
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

    def _build_summary(self) -> Dict[str, Any]:
        """Build summary dictionary for JSON export."""
        summary = {
            'geometry': self.results.get('geometry', {}),
            'materials': self.results.get('materials', {}),
            'nonlocal': self.results.get('nonlocal', {'enabled': False}),
            'simulation': {
                'type': self.config.get('simulation_type', 'stat'),
                'excitation': self.config.get('excitation_type', 'planewave'),
                'wavelength_range': self.config.get('wavelength_range', []),
            },
        }

        # Add field statistics
        if 'field_stats' in self.results:
            summary['field_stats'] = {
                str(k): v for k, v in self.results['field_stats'].items()
            }

        # Add hotspots
        if 'hotspots' in self.results:
            summary['hotspots'] = {
                str(k): v for k, v in self.results['hotspots'].items()
            }

        # Add mode analysis
        if 'mode_analysis' in self.results:
            summary['mode_analysis'] = {
                str(k): v for k, v in self.results['mode_analysis'].items()
            }

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation.

        Returns:
            Dictionary with summary information
        """
        wl_range = self.config.get('wavelength_range', [400, 800, 100])

        return {
            'structure': self.config.get('structure', 'unknown'),
            'simulation_type': self.config.get('simulation_type', 'stat'),
            'excitation_type': self.config.get('excitation_type', 'planewave'),
            'wavelength_range': f"{wl_range[0]}-{wl_range[1]} nm ({wl_range[2]} points)",
            'run_folder': self.run_folder,
            'materials': self.config.get('materials', []),
            'medium': self.config.get('medium', 'air'),
            'use_nonlocality': self.config.get('use_nonlocality', False),
        }

    def get_results(self) -> Dict[str, Any]:
        """Get all simulation results."""
        return self.results

    def get_run_folder(self) -> Optional[str]:
        """Get the run folder path."""
        return self.run_folder
