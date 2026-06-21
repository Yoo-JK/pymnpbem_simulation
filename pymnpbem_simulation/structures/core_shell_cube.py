from typing import Any, Dict, Tuple

import numpy as np

from .advanced_monomer_cube import _resolve_n_per_edge
from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from .core_shell_sphere import _normalize_shells, _build_inout_table
from ..util import print_info


class CoreShellCubeBuilder(StructureBuilder):
    """Multi-shell core_shell cube builder (1+ shells).

    YAML config (single-shell, legacy)::

        structure:
          type: core_shell_cube
          core_size: 30
          shell_thickness: 5
          n_per_edge: 16

    YAML config (N shells, v1.5+)::

        structure:
          type: core_shell_cube
          core_size: 30
          n_per_edge: 16
          shells:
            - thickness: 3.0
              material: silver
            - thickness: 2.0
              material: silica
    """

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        core_size = float(self.cfg_struct.get('core_size', 30.0))

        cum_size_outer = core_size
        shells_raw = self.cfg_struct.get('shells', None)
        if shells_raw:
            for sh in shells_raw:
                cum_size_outer = cum_size_outer + 2.0 * float(sh['thickness'])
        elif 'shell_thickness' in self.cfg_struct:
            cum_size_outer = cum_size_outer + 2.0 * float(self.cfg_struct['shell_thickness'])

        n_per_edge = _resolve_n_per_edge(self.cfg_struct, 1,
                edge_override = cum_size_outer)[0]
        e = float(self.cfg_struct.get('e', self.cfg_struct.get('rounding', 0.25)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        shells = _normalize_shells(self.cfg_struct, self.cfg_materials,
                default_n = n_per_edge)

        if len(shells) == 0:
            raise ValueError(
                '[error] CoreShellCubeBuilder: no shells specified '
                '(set <shell_thickness> or <shells>)')

        medium_name = self.cfg_materials.get('medium', 'water')
        core_name = self.cfg_materials.get('core',
                self.cfg_materials.get('particle', 'gold'))

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_core = _build_eps_particle(core_name, rip)

        epstab = [eps_medium, eps_core]
        for sh in shells:
            epstab.append(_build_eps_particle(sh['material'], rip))

        cum_size = core_size
        particles = [tricube(n_per_edge, core_size, e = e)]
        for sh in shells:
            cum_size = cum_size + 2.0 * float(sh['thickness'])
            n_edge = int(sh.get('n_per_edge', sh['n']))
            particles.append(tricube(n_edge, cum_size, e = e))

        inout = _build_inout_table(len(shells))

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info(
            'CoreShellCubeBuilder: core={}nm, n_shells={}, nfaces={}'.format(
                core_size, len(shells), nfaces))

        return p, epstab, nfaces
