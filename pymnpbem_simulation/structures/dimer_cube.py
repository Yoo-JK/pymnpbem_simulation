from typing import Any, Dict, Tuple

import numpy as np

from .advanced_monomer_cube import _resolve_n_per_edge
from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


_DIMER_CUBE_DEFAULT_N = 24


class DimerCubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        edge = float(self.cfg_struct.get('edge', 47.0))
        gap = float(self.cfg_struct.get('gap', 0.6))
        if self.cfg_struct.get('mesh_density') is None and self.cfg_struct.get('n_per_edge') is None:
            n_per_edge = _DIMER_CUBE_DEFAULT_N
        else:
            n_per_edge = _resolve_n_per_edge(self.cfg_struct, 1, edge_override = edge)[0]
        e = float(self.cfg_struct.get('e', 0.2))
        refine = int(self.cfg_struct.get('refine', 3))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
        epstab = [eps_medium, eps_particle]

        half = edge / 2 + gap / 2

        cube1 = tricube(n_per_edge, edge, e = e, refine = refine)
        cube1.shift([-half, 0, 0])

        cube2 = tricube(n_per_edge, edge, e = e, refine = refine)
        cube2.shift([+half, 0, 0])

        p = ComParticle(epstab, [cube1, cube2], [[2, 1], [2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('DimerCubeBuilder: edge={}, gap={}, n={}, e={}, refine={}, nfaces={}'.format(
            edge, gap, n_per_edge, e, refine, nfaces))

        return p, epstab, nfaces
