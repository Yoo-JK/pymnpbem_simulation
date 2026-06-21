from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from .rod import _resolve_rod_mesh
from .core_shell_sphere import _normalize_shells, _build_inout_table
from ..util import print_info


class CoreShellRodBuilder(StructureBuilder):
    """Multi-shell core_shell rod builder (1+ shells).

    YAML config (single-shell, legacy)::

        structure:
          type: core_shell_rod
          core_diameter: 15
          shell_thickness: 5
          height: 80

    YAML config (N shells, v1.5+)::

        structure:
          type: core_shell_rod
          core_diameter: 15
          height: 80
          shells:
            - thickness: 3.0
              material: silver
            - thickness: 2.0
              material: silica
    """

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trirod, ComParticle

        core_diameter = float(self.cfg_struct.get('core_diameter', 15.0))
        height = float(self.cfg_struct.get('height', 80.0))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')
        horizontal = bool(self.cfg_struct.get('horizontal', True))

        # Rod meshes are 3-tuples [nphi, ntheta, nz] (list); per-shell n is
        # not directly used by trirod (we recompute via _resolve_rod_mesh
        # against the cumulative dimensions). Pass `0` as the placeholder
        # default so legacy `n_shell` (if numeric) still parses.
        shells = _normalize_shells(self.cfg_struct, self.cfg_materials,
                default_n = 0)

        if len(shells) == 0:
            raise ValueError(
                '[error] CoreShellRodBuilder: no shells specified '
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

        # Cumulative dimensions: each shell extends radius (and possibly
        # height) by `thickness`. We grow both diameter and height; height
        # never shrinks below core_diameter.
        cum_d = core_diameter
        cum_h = max(core_diameter, height)
        # Base rod is shrunken so that final outermost rod equals the
        # legacy (height, core_diameter + 2 * total_shell_thickness).
        total_shell_thickness = sum(float(sh['thickness']) for sh in shells)
        core_height_eff = max(core_diameter, height - 2.0 * total_shell_thickness)

        p_core = trirod(core_diameter, core_height_eff,
                _resolve_rod_mesh(self.cfg_struct, core_diameter, core_height_eff),
                triangles = True)
        if horizontal:
            p_core.rot(90, [0, 1, 0])

        particles = [p_core]
        for sh in shells:
            cum_d = cum_d + 2.0 * float(sh['thickness'])
            cum_h = cum_h + 2.0 * float(sh['thickness'])
            n_sh = _resolve_rod_mesh(self.cfg_struct, cum_d, cum_h)
            p_sh = trirod(cum_d, cum_h, n_sh, triangles = True)
            if horizontal:
                p_sh.rot(90, [0, 1, 0])
            particles.append(p_sh)

        inout = _build_inout_table(len(shells))

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info(
            'CoreShellRodBuilder: core_d={}nm, height={}nm, n_shells={}, nfaces={}'.format(
                core_diameter, height, len(shells), nfaces))

        return p, epstab, nfaces
