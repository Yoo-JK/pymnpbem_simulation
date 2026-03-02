import os
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

        # 3.5 Visualize geometry
        self._visualize_geometry(particles)

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

    def _visualize_geometry(self, particles: List[Any]) -> Optional[str]:
        output_dir = str(self.run_folder)
        plot_format = self.config.get('plot_format', ['png'])
        if isinstance(plot_format, str):
            plot_format = [plot_format]
        plot_dpi = self.config.get('plot_dpi', 300)

        if self.verbose:
            print('[info] Visualizing geometry...')

        # Collect all vertices/faces from particles
        all_verts_list = []
        all_faces_list = []
        particle_ids = []  # which particle each face belongs to
        vert_offset = 0

        for pid, p in enumerate(particles):
            verts = p.verts
            faces = p.faces
            if len(faces) == 0:
                continue

            all_verts_list.append(verts)

            # Offset face indices by accumulated vertex count
            faces_shifted = faces.copy()
            valid_mask = ~np.isnan(faces_shifted)
            faces_shifted[valid_mask] = faces_shifted[valid_mask] + vert_offset

            all_faces_list.append(faces_shifted)
            particle_ids.extend([pid] * len(faces))
            vert_offset += len(verts)

        if len(all_verts_list) == 0:
            if self.verbose:
                print('[info] No geometry to visualize (empty particles)')
            return None

        all_verts = np.vstack(all_verts_list)
        all_faces = np.vstack(all_faces_list)
        particle_ids = np.array(particle_ids)

        total_faces = len(all_faces)
        total_verts = len(all_verts)

        # Split quads into triangles (faces are 0-based)
        tri_faces = []
        tri_particle_ids = []
        for i, face in enumerate(all_faces):
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            tri_faces.append([v0, v1, v2])
            tri_particle_ids.append(particle_ids[i])

            if not np.isnan(face[3]):
                v3 = int(face[3])
                tri_faces.append([v0, v2, v3])
                tri_particle_ids.append(particle_ids[i])

        tri_faces = np.array(tri_faces)
        tri_particle_ids = np.array(tri_particle_ids)

        # Colors for different particles
        particle_colors = [
            '#DAA520',  # goldenrod
            '#B8860B',  # dark goldenrod
            '#FFD700',  # gold
            '#CD853F',  # peru
            '#D2691E',  # chocolate
        ]

        face_colors = []
        for pid in tri_particle_ids:
            face_colors.append(particle_colors[pid % len(particle_colors)])

        # 4 viewpoints: perspective, top, front, side
        views = [
            ('Perspective', 30, -60),
            ('Top (z+)', 90, -90),
            ('Front (y-)', 0, -90),
            ('Side (x+)', 0, 0),
        ]

        fig = plt.figure(figsize = (14, 12))
        fig.patch.set_facecolor('white')

        structure_name = self.config.get('structure', 'unknown')
        fig.suptitle('{} | {} faces, {} vertices'.format(
            structure_name, total_faces, total_verts),
            fontsize = 14, fontweight = 'bold', y = 0.98)

        verts_tri = all_verts[tri_faces]

        # Axis limits (equal aspect)
        x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
        y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
        z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)

        # Add some padding
        pad = max_range * 0.1
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2
        half = max_range / 2 + pad

        # Substrate
        use_substrate = self.config.get('use_substrate', False)
        z_interface = self.config.get('z_interface', 0.0)

        for idx, (title, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, idx + 1, projection = '3d')

            poly = Poly3DCollection(
                verts_tri, alpha = 0.85,
                edgecolor = 'k', linewidth = 0.2)
            poly.set_facecolor(face_colors)
            ax.add_collection3d(poly)

            # Substrate plane
            if use_substrate:
                substrate_half = half * 1.2
                sx = [x_mid - substrate_half, x_mid + substrate_half,
                      x_mid + substrate_half, x_mid - substrate_half]
                sy = [y_mid - substrate_half, y_mid - substrate_half,
                      y_mid + substrate_half, y_mid + substrate_half]
                sz = [z_interface] * 4
                substrate_verts = [list(zip(sx, sy, sz))]
                substrate_poly = Poly3DCollection(
                    substrate_verts, alpha = 0.2,
                    facecolor = '#4682B4', edgecolor = '#4682B4',
                    linewidth = 0.5, linestyle = '--')
                ax.add_collection3d(substrate_poly)

            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
            ax.set_zlim(z_mid - half, z_mid + half)

            ax.set_xlabel('x (nm)', fontsize = 9)
            ax.set_ylabel('y (nm)', fontsize = 9)
            ax.set_zlabel('z (nm)', fontsize = 9)
            ax.set_title(title, fontsize = 11)

            ax.view_init(elev = elev, azim = azim)

            if max_range > 0:
                ax.set_box_aspect([
                    x_range / max_range if x_range > 0 else 0.1,
                    y_range / max_range if y_range > 0 else 0.1,
                    z_range / max_range if z_range > 0 else 0.1,
                ])

        plt.tight_layout(rect = [0, 0, 1, 0.95])

        saved_files = []
        for fmt in plot_format:
            filepath = os.path.join(output_dir, 'geometry_preview.{}'.format(fmt))
            fig.savefig(filepath, dpi = plot_dpi, bbox_inches = 'tight')
            saved_files.append(filepath)

        plt.close(fig)

        if self.verbose:
            for f in saved_files:
                print('[info] Saved geometry preview: {}'.format(f))

        return saved_files[0] if saved_files else None

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
