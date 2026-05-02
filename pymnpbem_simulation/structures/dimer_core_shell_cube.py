from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


class DimerCoreShellCubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        core_size = float(self.cfg_struct.get('core_size', 30.0))
        shell_thickness = float(self.cfg_struct.get('shell_thickness', 5.0))
        gap = float(self.cfg_struct.get('gap', 5.0))
        n_per_edge = int(self.cfg_struct.get('n_per_edge', 16))
        e = float(self.cfg_struct.get('e', self.cfg_struct.get('rounding', 0.25)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        shell_size = core_size + 2.0 * shell_thickness
        shift = (shell_size + gap) / 2.0

        medium_name = self.cfg_materials.get('medium', 'water')
        core_name = self.cfg_materials.get('core', self.cfg_materials.get('particle', 'gold'))
        shell_name = self.cfg_materials.get('shell', 'silver')

        eps_medium = _build_eps_medium(medium_name)
        eps_core = _build_eps_particle(core_name)
        eps_shell = _build_eps_particle(shell_name)
        epstab = [eps_medium, eps_core, eps_shell]

        c1 = tricube(n_per_edge, core_size, e = e)
        c1.shift([-shift, 0.0, 0.0])
        s1 = tricube(n_per_edge, shell_size, e = e)
        s1.shift([-shift, 0.0, 0.0])

        c2 = tricube(n_per_edge, core_size, e = e)
        c2.shift([+shift, 0.0, 0.0])
        s2 = tricube(n_per_edge, shell_size, e = e)
        s2.shift([+shift, 0.0, 0.0])

        p = ComParticle(epstab, [c1, s1, c2, s2],
                [[2, 3], [3, 1], [2, 3], [3, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('DimerCoreShellCubeBuilder: core={}nm, shell_t={}nm, gap={}nm, nfaces={}'.format(
            core_size, shell_thickness, gap, nfaces))

        return p, epstab, nfaces
