from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from .rod import _resolve_rod_mesh
from ..util import print_info


class CoreShellRodBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trirod, ComParticle

        core_diameter = float(self.cfg_struct.get('core_diameter', 15.0))
        shell_thickness = float(self.cfg_struct.get('shell_thickness', 5.0))
        height = float(self.cfg_struct.get('height', 80.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')
        horizontal = bool(self.cfg_struct.get('horizontal', True))

        shell_diameter = core_diameter + 2.0 * shell_thickness
        shell_height = height
        core_height = max(core_diameter, height - 2.0 * shell_thickness)

        n_core = _resolve_rod_mesh(self.cfg_struct, core_diameter, core_height)
        n_shell = _resolve_rod_mesh(self.cfg_struct, shell_diameter, shell_height)

        medium_name = self.cfg_materials.get('medium', 'water')
        core_name = self.cfg_materials.get('core', self.cfg_materials.get('particle', 'gold'))
        shell_name = self.cfg_materials.get('shell', 'silver')

        eps_medium = _build_eps_medium(medium_name)
        eps_core = _build_eps_particle(core_name)
        eps_shell = _build_eps_particle(shell_name)
        epstab = [eps_medium, eps_core, eps_shell]

        p_core = trirod(core_diameter, core_height, n_core, triangles = True)
        p_shell = trirod(shell_diameter, shell_height, n_shell, triangles = True)

        if horizontal:
            p_core.rot(90, [0, 1, 0])
            p_shell.rot(90, [0, 1, 0])

        p = ComParticle(epstab, [p_core, p_shell],
                [[2, 3], [3, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('CoreShellRodBuilder: core_d={}nm, shell_t={}nm, height={}nm, nfaces={}'.format(
            core_diameter, shell_thickness, height, nfaces))

        return p, epstab, nfaces
