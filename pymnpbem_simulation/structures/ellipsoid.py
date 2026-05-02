from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


class EllipsoidBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import ComParticle
        from mnpbem.geometry.mesh_generators import triellipsoid

        axes = self.cfg_struct.get('axes', [10.0, 15.0, 20.0])
        axes = [float(v) for v in axes]
        assert len(axes) == 3, '[error] <axes> must be [a, b, c]'

        n_verts = int(self.cfg_struct.get('n_verts',
                self.cfg_struct.get('mesh_density', 256)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name)
        epstab = [eps_medium, eps_particle]

        ell = triellipsoid(n_verts, axes)

        p = ComParticle(epstab, [ell], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('EllipsoidBuilder: axes={}, n_verts={}, nfaces={}'.format(
            axes, n_verts, nfaces))

        return p, epstab, nfaces
