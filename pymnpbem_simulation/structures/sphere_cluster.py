from typing import Any, Dict, List, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


def _cluster_positions(n_spheres: int, spacing: float) -> List[Tuple[float, float]]:
    dy_60 = spacing * np.sqrt(3.0) / 2.0

    hex_positions = []
    for i in range(6):
        angle = i * 60.0 * np.pi / 180.0
        x = spacing * np.cos(angle)
        y = spacing * np.sin(angle)
        hex_positions.append((x, y))

    table = {
        1: [(0.0, 0.0)],
        2: [(-spacing / 2.0, 0.0),
            (+spacing / 2.0, 0.0)],
        3: [(-spacing / 2.0, 0.0),
            (+spacing / 2.0, 0.0),
            (0.0, dy_60)],
        4: [(0.0, 0.0)] + hex_positions[0:3],
        5: [(0.0, 0.0)] + hex_positions[0:4],
        6: [(0.0, 0.0)] + hex_positions[0:5],
        7: [(0.0, 0.0)] + hex_positions[0:6]}

    if n_spheres not in table:
        raise ValueError('[error] <n_spheres> must be 1-7, got <{}>'.format(n_spheres))

    return table[n_spheres]


class SphereClusterBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trisphere, ComParticle

        n_spheres = int(self.cfg_struct.get('n_spheres', 1))
        diameter = float(self.cfg_struct.get('diameter', 50.0))
        gap = float(self.cfg_struct.get('gap', -0.1))
        n_verts = int(self.cfg_struct.get('n_verts',
                self.cfg_struct.get('mesh_density', 144)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        spacing = diameter + gap

        positions = _cluster_positions(n_spheres, spacing)

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
        epstab = [eps_medium, eps_particle]

        particles = []
        for x, y in positions:
            sph = trisphere(n_verts, diameter)
            sph.shift([x, y, 0.0])
            particles.append(sph)

        inout = [[2, 1] for _ in particles]

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('SphereClusterBuilder: n_spheres={}, diameter={}nm, gap={}nm, nfaces={}'.format(
            n_spheres, diameter, gap, nfaces))

        return p, epstab, nfaces
