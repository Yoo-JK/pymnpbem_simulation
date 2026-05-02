from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


class CoreShellSphereBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trisphere, ComParticle

        core_diameter = float(self.cfg_struct.get('core_diameter', 30.0))
        shell_thickness = float(self.cfg_struct.get('shell_thickness', 5.0))
        n_core = int(self.cfg_struct.get('n_core',
                self.cfg_struct.get('mesh_density', 256)))
        n_shell = int(self.cfg_struct.get('n_shell',
                self.cfg_struct.get('mesh_density', 256)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        shell_diameter = core_diameter + 2.0 * shell_thickness

        medium_name = self.cfg_materials.get('medium', 'water')
        core_name = self.cfg_materials.get('core', self.cfg_materials.get('particle', 'gold'))
        shell_name = self.cfg_materials.get('shell', 'silver')

        eps_medium = _build_eps_medium(medium_name)
        eps_core = _build_eps_particle(core_name)
        eps_shell = _build_eps_particle(shell_name)
        epstab = [eps_medium, eps_core, eps_shell]

        p_core = trisphere(n_core, core_diameter)
        p_shell = trisphere(n_shell, shell_diameter)

        p = ComParticle(epstab, [p_core, p_shell],
                [[2, 3], [3, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('CoreShellSphereBuilder: core_d={}nm, shell_t={}nm, nfaces={}'.format(
            core_diameter, shell_thickness, nfaces))

        return p, epstab, nfaces
