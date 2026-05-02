from typing import Any, Dict, List, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


def _resolve_rod_mesh(cfg: Dict[str, Any], diameter: float, height: float) -> List[int]:
    if 'rod_mesh' in cfg:
        n = list(cfg['rod_mesh'])
        assert len(n) == 3, '[error] rod_mesh must have 3 elements [nphi, ntheta, nz]'
        return [int(v) for v in n]

    if 'nphi' in cfg or 'ntheta' in cfg or 'nz' in cfg:
        nphi = int(cfg.get('nphi', 15))
        ntheta = int(cfg.get('ntheta', 20))
        nz = int(cfg.get('nz', 20))
        return [nphi, ntheta, nz]

    element_size = float(cfg.get('mesh_density', 2.0))
    nphi = max(8, int(np.ceil(np.pi * diameter / element_size)))
    ntheta = max(6, int(np.ceil(0.5 * diameter / element_size)))
    nz = max(2, int(np.ceil(max(0.0, height - diameter) / element_size)))
    return [nphi, ntheta, nz]


class RodBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trirod, ComParticle

        diameter = float(self.cfg_struct.get('diameter', 10.0))
        height = float(self.cfg_struct.get('height', 50.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')
        horizontal = bool(self.cfg_struct.get('horizontal', True))

        n_mesh = _resolve_rod_mesh(self.cfg_struct, diameter, height)

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name)
        epstab = [eps_medium, eps_particle]

        rod = trirod(diameter, height, n_mesh, triangles = True)

        if horizontal:
            rod.rot(90, [0, 1, 0])

        p = ComParticle(epstab, [rod], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('RodBuilder: diameter={}nm, height={}nm, mesh={}, nfaces={}'.format(
            diameter, height, n_mesh, nfaces))

        return p, epstab, nfaces
