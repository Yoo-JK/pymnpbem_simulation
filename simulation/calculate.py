import os
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

from .sim_utils.solver import BEMSolver


class SimulationManager(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose
        self.run_folder = None
        self.results = None

        # Sub-managers will be initialized lazily when run() is called
        self.geometry_gen = None
        self.material_mgr = None
        self.solver = BEMSolver(config, verbose)

        if verbose:
            print('[info] SimulationManager initialized')

    def create_run_folder(self) -> Path:
        base_output_dir = Path(self.config['output_dir'])
        sim_name = self.config.get('simulation_name', 'simulation')
        folder_name = sim_name

        self.run_folder = base_output_dir / folder_name

        if self.run_folder.exists():
            if self.verbose:
                print('[info] Using existing folder (preserving previous results): {}'.format(
                    self.run_folder))
        else:
            self.run_folder.mkdir(parents = True, exist_ok = True)
            if self.verbose:
                print('[info] Created run folder: {}'.format(self.run_folder))

        # Create logs subfolder
        (self.run_folder / 'logs').mkdir(exist_ok = True)

        # Update config to use this run folder
        self.config['output_dir'] = str(self.run_folder)

        return self.run_folder

    def save_config_snapshot(self) -> str:
        if self.run_folder is None:
            raise RuntimeError(
                '[error] Run folder not created. Call create_run_folder() first.')

        config_file = self.run_folder / 'config_snapshot.py'

        with open(config_file, 'w') as f:
            f.write('# Configuration snapshot\n')
            f.write('# Generated: {}\n\n'.format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write('args = {\n')
            for key, value in self.config.items():
                if isinstance(value, str):
                    f.write("    '{}': '{}',\n".format(key, value))
                else:
                    f.write("    '{}': {},\n".format(key, repr(value)))
            f.write('}\n')

        if self.verbose:
            print('[info] Saved config snapshot: {}'.format(config_file))

        return str(config_file)

    def run(self) -> Dict[str, Any]:
        t_start = time.time()

        if self.verbose:
            print('')
            print('================================================================')
            print('         Starting Python MNPBEM Simulation')
            print('================================================================')
            print('Structure: {}'.format(self.config.get('structure', 'unknown')))
            print('Simulation type: {}'.format(self.config.get('simulation_type', 'stat')))
            print('Excitation: {}'.format(self.config.get('excitation_type', 'planewave')))

        # 1. Create run folder
        if self.run_folder is None:
            self.create_run_folder()

        # 2. Save config snapshot
        self.save_config_snapshot()

        # 3. Generate particles via GeometryGenerator
        if self.verbose:
            print('')
            print('[info] Generating geometry...')

        particles = self._generate_particles()

        # 4. Generate materials via MaterialManager
        if self.verbose:
            print('[info] Generating materials...')

        materials = self._generate_materials()

        # 5. Run BEM simulation
        if self.verbose:
            print('')
            print('[info] Running BEM simulation...')

        self.results = self.solver.run(particles, materials)

        # 6. Save results
        if self.verbose:
            print('')
            print('[info] Saving results...')

        output_path = str(self.run_folder)
        self.solver.save_results(self.results, output_path)

        t_elapsed = time.time() - t_start

        if self.verbose:
            print('')
            print('================================================================')
            print('  Simulation completed in {:.2f} seconds ({:.2f} minutes)'.format(
                t_elapsed, t_elapsed / 60.0))
            print('  Results saved to: {}'.format(output_path))
            print('================================================================')

        return self.results

    def _generate_particles(self) -> List[Any]:
        # Lazy import to avoid circular dependencies
        from .sim_utils.geometry_generator import GeometryGenerator

        if self.geometry_gen is None:
            self.geometry_gen = GeometryGenerator(self.config, self.verbose)

        return self.geometry_gen.generate()

    def _generate_materials(self) -> Dict[str, Any]:
        # Lazy import to avoid circular dependencies
        from .sim_utils.material_manager import MaterialManager

        if self.material_mgr is None:
            self.material_mgr = MaterialManager(self.config, self.verbose)

        return self.material_mgr.generate()

    def get_run_folder(self) -> Optional[Path]:
        return self.run_folder

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'structure': self.config.get('structure', 'unknown'),
            'simulation_type': self.config.get('simulation_type', 'stat'),
            'excitation': self.config.get('excitation_type', 'planewave'),
            'wavelength_range': self.config.get('wavelength_range', [400, 800, 100]),
            'materials': self.config.get('materials', []),
            'run_folder': str(self.run_folder) if self.run_folder else None,
            'calculate_cross_sections': self.config.get('calculate_cross_sections', True),
            'calculate_fields': self.config.get('calculate_fields', False),
        }

        if self.results is not None:
            summary['calculation_time'] = self.results.get('calculation_time', 0)
            summary['n_wavelengths'] = len(self.results.get('wavelength', []))
            summary['n_polarizations'] = self.results.get('extinction', np.empty(0)).shape[0]

            # Peak wavelengths
            ext = self.results.get('extinction', None)
            wl = self.results.get('wavelength', None)
            if ext is not None and wl is not None and ext.size > 0:
                import numpy as np
                peak_info = []
                for j in range(ext.shape[0]):
                    idx = int(np.argmax(ext[j, :]))
                    peak_info.append({
                        'pol_index': j,
                        'peak_wavelength': float(wl[idx]),
                        'peak_extinction': float(ext[j, idx]),
                    })
                summary['peaks'] = peak_info

        return summary
