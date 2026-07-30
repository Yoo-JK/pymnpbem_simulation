from typing import Any, Dict, Optional, Tuple

import numpy as np

from .adaptive_cube_mesh import build_adaptive_cube
from .advanced_monomer_cube import _resolve_n_per_edge
from .base import StructureBuilder
from .sphere import (_build_eps_medium, _build_eps_particle, _count_faces,
        _resolve_materials_list, _resolve_rip)
from ..util import print_info


def _resolve_face_densities(cfg: Dict) -> Optional[Dict[str, int]]:
    """Return per-face density dict or None (uniform path).

    Config key ``face_densities`` accepts a dict with any subset of
    ``'+x', '-x', '+y', '-y', '+z', '-z'`` → int.
    """
    fd = cfg.get('face_densities', None)
    if fd is None:
        return None
    return {str(k): int(v) for k, v in fd.items()}


def _resolve_edge_profile_kwargs(cfg: Dict) -> Optional[Dict]:
    """Return EdgeProfile kwargs dict or None.

    Config key ``edge_profile``: dict with optional keys ``e``, ``dz``, ``mode``,
    ``nz``.  If the key is present but falsy, returns None.
    """
    ep = cfg.get('edge_profile', None)
    if not ep:
        return None
    return dict(ep)


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

        face_densities = _resolve_face_densities(self.cfg_struct)
        edge_profile_kw = _resolve_edge_profile_kwargs(self.cfg_struct)

        if face_densities is not None or edge_profile_kw is not None:
            # Per-face density / edge-profile path
            cube = build_adaptive_cube(
                size = size,
                n_default = n_per_edge,
                face_densities = face_densities,
                e = e,
                edge_profile_kwargs = edge_profile_kw,
                interp = interp)
            print_info('CubeBuilder (adaptive): size={}nm, n_default={}, e={}, '
                       'face_densities={}, edge_profile={}'.format(
                           size, n_per_edge, e, face_densities, edge_profile_kw))
        else:
            # Uniform path (original)
            cube = tricube(n_per_edge, size, e = e)
            print_info('CubeBuilder: size={}nm, n={}, e={}, refine={}'.format(
                size, n_per_edge, e, refine))

        p = ComParticle(epstab, [cube], [[2, 1]],
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('CubeBuilder: nfaces={}'.format(nfaces))

        return p, epstab, nfaces
