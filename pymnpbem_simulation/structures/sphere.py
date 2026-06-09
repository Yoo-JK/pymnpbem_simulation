from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from ..util import print_info


_MATERIAL_DEFAULTS = {
    'water': 1.33 ** 2,
    'vacuum': 1.0,
    'air': 1.0,
    'glass': 1.5 ** 2}


class SphereBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.materials import EpsConst, EpsTable
        from mnpbem.geometry import trisphere, ComParticle

        diameter = float(self.cfg_struct.get('diameter', 50.0))
        n = int(self.cfg_struct.get('mesh_density', 256))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name)
        epstab = [eps_medium, eps_particle]

        sphere = trisphere(n, diameter)

        p = ComParticle(epstab, [sphere], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('SphereBuilder: diameter={}nm, n={}, nfaces={}'.format(
            diameter, n, nfaces))

        return p, epstab, nfaces


def _build_eps_medium(name: str) -> Any:
    from mnpbem.materials import EpsConst, EpsTable

    name_l = name.lower()

    if name_l in _MATERIAL_DEFAULTS:
        return EpsConst(_MATERIAL_DEFAULTS[name_l])

    if name.endswith('.dat'):
        return EpsTable(name)

    return EpsConst(float(name))


def _build_eps_particle(name: str, custom: Any = None) -> Any:
    from mnpbem.materials import EpsTable, EpsDrude

    name_l = name.lower()

    if name_l in {'gold', 'au'}:
        return EpsTable('gold.dat')

    if name_l in {'silver', 'ag'}:
        return EpsTable('silver.dat')

    if name.endswith('.dat'):
        return EpsTable(name)

    # custom material from refractive_index_paths (e.g. agcl, ito):
    #   {'agcl': {'type': 'constant', 'epsilon': 2.02}}  ->  EpsConst(2.02)
    #   {'foo':  {'type': 'table', 'path': 'foo.dat'}}   ->  EpsTable('foo.dat')
    if custom:
        cmap = {str(k).lower(): v for k, v in custom.items()}
        if name_l in cmap:
            from mnpbem.materials import EpsConst
            m = cmap[name_l]
            mtype = str(m.get('type', 'constant')).lower()
            if mtype == 'constant':
                return EpsConst(float(m['epsilon']))
            return EpsTable(m.get('path', m.get('file', name)))

    raise ValueError('[error] Unsupported <particle> = <{}>!'.format(name))


def _count_faces(p: Any) -> int:
    if hasattr(p, 'pfull'):
        return int(p.pfull.nfaces)

    if hasattr(p, 'nfaces'):
        return int(p.nfaces)

    if hasattr(p, 'pos'):
        return int(len(p.pos))

    return -1
