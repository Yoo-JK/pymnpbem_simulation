from typing import Any, Dict, List, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from .advanced_monomer_cube import _resolve_roundings, _resolve_n_per_edge
from ..util import print_info


class AdvancedDimerCubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        core_size = float(self.cfg_struct.get('core_size', 30.0))
        shell_layers = list(self.cfg_struct.get('shell_layers', []))
        materials = list(self.cfg_struct.get('materials', []))
        gap = float(self.cfg_struct.get('gap', 5.0))
        offset = list(self.cfg_struct.get('offset', [0.0, 0.0, 0.0]))
        tilt_angle = float(self.cfg_struct.get('tilt_angle', 0.0))
        tilt_axis = list(self.cfg_struct.get('tilt_axis', [0.0, 1.0, 0.0]))
        rotation_angle = float(self.cfg_struct.get('rotation_angle', 0.0))
        # base_tilt_angle: tilt BOTH cubes about tilt_axis (the "base" orientation),
        # while tilt_angle/rotation_angle add the relative rotation to particle 2.
        # base_tilt = 45 - theta/2 + tilt_angle = theta tilts the dimer so both cubes
        # rest on edges (mirror-symmetric) on a substrate.
        base_tilt_angle = float(self.cfg_struct.get('base_tilt_angle', 0.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        n_layers = 1 + len(shell_layers)

        if not materials:
            base_particle = self.cfg_materials.get('particle', 'gold')
            base_shell = self.cfg_materials.get('shell', 'silver')
            materials = [base_particle] + [base_shell] * len(shell_layers)

        assert len(materials) == n_layers, \
            '[error] <materials> length must equal 1 (core) + len(shell_layers)'

        roundings = _resolve_roundings(self.cfg_struct, n_layers)

        sizes = [core_size]
        for thickness in shell_layers:
            sizes.append(sizes[-1] + 2.0 * float(thickness))

        # Per-layer n_per_edge: each layer uses its own size for mesh_density.
        # Without edge_override, _resolve_n_per_edge would use core_size for all
        # layers — making the shell mesh sparser than the core under the same
        # mesh_density (e.g. shell size 55nm at density 2nm should get n=28,
        # not n=24 derived from core size 47). connected_dimer_cube uses the
        # same per-layer pattern.
        n_per_edges = []
        for size in sizes:
            n_per_edges.append(
                    _resolve_n_per_edge(self.cfg_struct, 1, edge_override = size)[0])

        total_size = sizes[-1]
        shift_distance = (total_size + gap) / 2.0

        medium_name = self.cfg_materials.get('medium', 'water')
        eps_medium = _build_eps_medium(medium_name)
        rip = self.cfg_struct.get('refractive_index_paths', None)
        eps_layers = [_build_eps_particle(name, rip) for name in materials]
        epstab = [eps_medium] + eps_layers

        particles_p1 = []
        for size, n_e, e in zip(sizes, n_per_edges, roundings):
            cube = tricube(n_e, size, e = e)
            if base_tilt_angle != 0.0:
                cube.rot(base_tilt_angle, tilt_axis)
            cube.shift([-shift_distance, 0.0, 0.0])
            particles_p1.append(cube)

        particles_p2 = []
        for size, n_e, e in zip(sizes, n_per_edges, roundings):
            cube = tricube(n_e, size, e = e)
            if rotation_angle != 0.0:
                cube.rot(rotation_angle, [0.0, 0.0, 1.0])
            if base_tilt_angle != 0.0:
                cube.rot(base_tilt_angle, tilt_axis)
            if tilt_angle != 0.0:
                cube.rot(tilt_angle, tilt_axis)
            cube.shift([shift_distance + offset[0], offset[1], offset[2]])
            particles_p2.append(cube)

        all_particles = particles_p1 + particles_p2

        single_inout = []
        for i in range(n_layers):
            mat_idx = 2 + i
            if i == 0:
                if n_layers == 1:
                    single_inout.append([mat_idx, 1])
                else:
                    single_inout.append([mat_idx, mat_idx + 1])
            elif i == n_layers - 1:
                single_inout.append([mat_idx, 1])
            else:
                single_inout.append([mat_idx, mat_idx + 1])

        inout = single_inout + single_inout

        p = ComParticle(epstab, all_particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('AdvancedDimerCubeBuilder: core={}nm, shells={}, gap={}nm, tilt={}deg, rot={}deg, nfaces={}'.format(
            core_size, shell_layers, gap, tilt_angle, rotation_angle, nfaces))

        return p, epstab, nfaces
