from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


class DimerSphereBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trisphere, ComParticle

        diameter = float(self.cfg_struct.get('diameter', 50.0))
        gap = float(self.cfg_struct.get('gap', 5.0))
        n_verts = int(self.cfg_struct.get('n_verts',
                self.cfg_struct.get('mesh_density', 256)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name)
        epstab = [eps_medium, eps_particle]

        shift = (diameter + gap) / 2.0

        s1 = trisphere(n_verts, diameter)
        s1.shift([-shift, 0.0, 0.0])

        s2 = trisphere(n_verts, diameter)
        s2.shift([+shift, 0.0, 0.0])

        p = ComParticle(epstab, [s1, s2], [[2, 1], [2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('DimerSphereBuilder: diameter={}nm, gap={}nm, n_verts={}, nfaces={}'.format(
            diameter, gap, n_verts, nfaces))

        return p, epstab, nfaces
