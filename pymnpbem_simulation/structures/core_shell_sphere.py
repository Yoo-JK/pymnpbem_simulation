from typing import Any, Dict, List, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


class CoreShellSphereBuilder(StructureBuilder):
    """Multi-shell core_shell sphere builder (1+ shells).

    YAML config (single-shell, legacy)::

        structure:
          type: core_shell_sphere
          core_diameter: 30
          shell_thickness: 5
          n_core: 256
          n_shell: 256
        materials:
          medium: water
          core: gold       # alias: particle
          shell: silver

    YAML config (N shells, v1.5+)::

        structure:
          type: core_shell_sphere
          core_diameter: 30
          n_core: 256
          shells:
            - thickness: 3.0
              material: silver
              n: 256
            - thickness: 2.0
              material: silica
              n: 256
            - thickness: 1.0
              material: tio2
              n: 256

    Single-shell legacy keys (`shell_thickness`, `shell`, `n_shell`) remain
    supported and are normalized into a 1-element shells list internally.
    """

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import trisphere, ComParticle

        core_diameter = float(self.cfg_struct.get('core_diameter', 30.0))
        n_core = int(self.cfg_struct.get('n_core',
                self.cfg_struct.get('mesh_density', 256)))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        shells = _normalize_shells(self.cfg_struct, self.cfg_materials,
                default_n = n_core)

        if len(shells) == 0:
            raise ValueError(
                '[error] CoreShellSphereBuilder: no shells specified '
                '(set <shell_thickness> or <shells>)')

        medium_name = self.cfg_materials.get('medium', 'water')
        core_name = self.cfg_materials.get('core',
                self.cfg_materials.get('particle', 'gold'))

        eps_medium = _build_eps_medium(medium_name)
        eps_core = _build_eps_particle(core_name)

        epstab = [eps_medium, eps_core]
        for sh in shells:
            epstab.append(_build_eps_particle(sh['material']))

        # Build cumulative diameters for shells.
        cum_d = core_diameter
        particles = [trisphere(n_core, core_diameter)]
        for sh in shells:
            cum_d = cum_d + 2.0 * float(sh['thickness'])
            particles.append(trisphere(int(sh['n']), cum_d))

        inout = _build_inout_table(len(shells))

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info(
            'CoreShellSphereBuilder: core_d={}nm, n_shells={}, nfaces={}'.format(
                core_diameter, len(shells), nfaces))

        return p, epstab, nfaces


def _normalize_shells(cfg_struct: Dict[str, Any],
        cfg_materials: Dict[str, Any],
        default_n: int) -> List[Dict[str, Any]]:
    """Resolve N-shell list from either new (`shells`) or legacy keys.

    Returns a list of dicts each containing keys ``thickness``, ``material``,
    ``n``. Empty list means "no shell" (caller must error).
    """
    shells_cfg = cfg_struct.get('shells', None)

    if shells_cfg is not None and len(shells_cfg) > 0:
        out = []
        for i, sh in enumerate(shells_cfg):
            thickness = float(sh['thickness'])
            material = sh.get('material',
                    cfg_materials.get('shell', 'silver'))
            n_sh = int(sh.get('n', default_n))
            out.append({
                'thickness': thickness,
                'material': material,
                'n': n_sh})
        return out

    # Legacy single-shell path
    if 'shell_thickness' in cfg_struct:
        thickness = float(cfg_struct['shell_thickness'])
        material = cfg_materials.get('shell', 'silver')
        n_sh = int(cfg_struct.get('n_shell', default_n))
        return [{'thickness': thickness, 'material': material, 'n': n_sh}]

    return []


def _build_inout_table(n_shells: int) -> List[List[int]]:
    """Construct ComParticle inout for [medium, core, shell_1, ..., shell_N].

    epstab indexing (1-based, MATLAB):
        eps[1] = medium
        eps[2] = core
        eps[3] = shell_1
        ...
        eps[N + 2] = shell_N

    Particle list (in build order):
        p[0] = core         inout = [core, shell_1] = [2, 3]
        p[1] = shell_1      inout = [shell_1, shell_2] = [3, 4]
        ...
        p[N - 1] = shell_N  inout = [shell_N, medium] = [N + 2, 1]
    """
    if n_shells == 1:
        return [[2, 3], [3, 1]]

    inout = [[2, 3]]
    for i in range(1, n_shells):
        inout.append([2 + i, 3 + i])
    inout.append([2 + n_shells, 1])  # outermost shell -> medium
    return inout
