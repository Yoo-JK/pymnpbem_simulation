from typing import Any, Dict, Tuple

import numpy as np

from .advanced_monomer_cube import _resolve_n_per_edge
from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


class CubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        size = float(self.cfg_struct.get('size', self.cfg_struct.get('edge', 20.0)))
        n_per_edge = _resolve_n_per_edge(self.cfg_struct, 1, edge_override = size)[0]
        e = float(self.cfg_struct.get('e', self.cfg_struct.get('rounding', 0.25)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
        epstab = [eps_medium, eps_particle]

        cube = tricube(n_per_edge, size, e = e)

        p = ComParticle(epstab, [cube], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('CubeBuilder: size={}nm, n={}, e={}, refine={}, nfaces={}'.format(
            size, n_per_edge, e, refine, nfaces))

        return p, epstab, nfaces
