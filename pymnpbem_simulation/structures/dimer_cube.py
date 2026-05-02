from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


class DimerCubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        edge = float(self.cfg_struct.get('edge', 47.0))
        gap = float(self.cfg_struct.get('gap', 0.6))
        n_per_edge = int(self.cfg_struct.get('n_per_edge', 24))
        e = float(self.cfg_struct.get('e', 0.2))
        refine = int(self.cfg_struct.get('refine', 3))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name)
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
