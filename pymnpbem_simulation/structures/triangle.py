from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


class TriangleBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import (
            tripolygon, Polygon, EdgeProfile, ComParticle)

        side_length = float(self.cfg_struct.get('side_length', 30.0))
        thickness = float(self.cfg_struct.get('thickness', 5.0))
        nz = int(self.cfg_struct.get('nz', 11))
        rounding_radius = self.cfg_struct.get('rounding_radius', None)
        nrad = int(self.cfg_struct.get('nrad', 5))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
        epstab = [eps_medium, eps_particle]

        poly = Polygon(3, size = [side_length, side_length * 2.0 / np.sqrt(3.0)])
        if rounding_radius is not None:
            poly.round_(rad = float(rounding_radius), nrad = nrad)
        else:
            poly.round_(nrad = nrad)

        edge = EdgeProfile(thickness, nz)

        tri = tripolygon(poly, edge)

        p = ComParticle(epstab, [tri], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('TriangleBuilder: side={}nm, thickness={}nm, nfaces={}'.format(
            side_length, thickness, nfaces))

        return p, epstab, nfaces
