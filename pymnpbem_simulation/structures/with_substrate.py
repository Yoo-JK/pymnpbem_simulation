from typing import Any, Dict, Tuple, Optional

import numpy as np

from .base import StructureBuilder
from ..util import print_info


_SUBSTRATE_PRESETS = {
    'glass': 1.5 ** 2,
    'silica': 1.45 ** 2,
    'silicon': 3.5 ** 2,
    'water': 1.33 ** 2,
    'vacuum': 1.0,
    'air': 1.0}


class WithSubstrateBuilder(StructureBuilder):
    """Wrap an arbitrary base structure on top of a planar dielectric substrate.

    The base structure is built first via the standard registry, then the
    particle is shifted so its lowest point sits ``z_shift`` nm above the
    substrate interface (``z_position``, default 0.0). A LayerStructure is
    constructed using the medium (above) + substrate (below). The substrate
    eps is appended to ``epstab`` so MNPBEM can resolve the layered Green
    function.

    Returned tuple is ``(p, epstab, nfaces)`` to remain compatible with the
    standard build_structure contract; the LayerStructure is attached on
    the particle as the attribute ``_mnpbem_layer`` so simulation runners
    can pick it up without changing the dispatch signature.

    Example YAML::

        structure:
          type: with_substrate
          base:
            type: sphere
            diameter: 20
            mesh_density: 144
          substrate:
            eps: 2.25            # n=1.5 glass; or 'glass'/'silicon'/...
            z_position: 0.0      # interface at z=0
            z_shift: 1.0         # particle bottom 1nm above substrate

        materials:
          medium: vacuum
          particle: gold
    """

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.materials import EpsConst, EpsTable
        from mnpbem.geometry import LayerStructure

        from . import build_structure

        cfg_base = self.cfg_struct.get('base', None)

        if cfg_base is None:
            raise ValueError(
                '[error] <structure.base> required for type=with_substrate')

        cfg_sub = self.cfg_struct.get('substrate', dict())

        z_position = float(cfg_sub.get('z_position', 0.0))
        z_shift = float(cfg_sub.get('z_shift', 1.0))
        eps_sub_spec = cfg_sub.get('eps', 'glass')

        # Build the base particle using the existing registry
        p, epstab_base, nfaces_base = build_structure(cfg_base, self.cfg_materials)

        # Construct the substrate dielectric
        eps_sub = _build_eps_substrate(eps_sub_spec)

        # Append substrate eps to the table; particle inout assignments stay
        # valid (they reference indices 1, 2 = medium, particle).
        epstab = list(epstab_base) + [eps_sub]
        sub_idx = len(epstab)  # 1-indexed for LayerStructure ind argument

        # Determine medium index in epstab. Convention used by other builders:
        # epstab[0] = medium (index 1), epstab[1] = particle (index 2).
        # LayerStructure.ind uses 1-based MATLAB indexing.
        medium_idx = 1

        # Build the LayerStructure: top layer = medium, bottom layer = substrate.
        layer = LayerStructure(epstab, [medium_idx, sub_idx], [z_position])

        # ComParticle was built by the base builder with only [medium, particle]
        # in its eps list. Spectrum/far-field code paths reference
        # ``p.eps[layer.ind[-1] - 1]`` to look up the substrate refractive
        # index, so the extended epstab (including substrate) must be visible
        # on the particle. Splice the full table in-place.
        try:
            p.eps = list(epstab)
        except Exception:
            pass

        if hasattr(p, 'pc') and p.pc is not None:
            try:
                p.pc.eps = list(epstab)
            except Exception:
                pass

        # Shift the base particle so its lowest face sits z_shift above the
        # substrate interface.
        try:
            zmin = float(_get_particle_pos(p)[:, 2].min())
        except Exception as e:
            raise RuntimeError(
                '[error] WithSubstrateBuilder: cannot read particle z positions: {}'.format(e))

        dz = z_position + z_shift - zmin
        _shift_particle(p, [0.0, 0.0, dz])

        # Stash the layer on the particle for the simulation runner to pick up.
        # We use a dedicated attribute name to avoid colliding with mnpbem internals.
        try:
            setattr(p, '_mnpbem_layer', layer)
        except Exception:
            # ComParticle may use slots; fall back to an attribute on pfull.
            if hasattr(p, 'pfull'):
                setattr(p.pfull, '_mnpbem_layer', layer)

        nfaces = _count_faces(p, fallback = nfaces_base)

        print_info(
            'WithSubstrate: base={}, substrate eps={}, z_position={}, z_shift={}, dz={:.3f}'.format(
                cfg_base.get('type'), _eps_repr(eps_sub_spec),
                z_position, z_shift, dz))
        print_info(
            'WithSubstrate: layer ind=[{}, {}] (medium, substrate), nfaces={}'.format(
                medium_idx, sub_idx, nfaces))

        return p, epstab, nfaces


def _build_eps_substrate(spec: Any) -> Any:
    from mnpbem.materials import EpsConst, EpsTable

    if isinstance(spec, (int, float)):
        return EpsConst(float(spec))

    if isinstance(spec, str):
        spec_l = spec.lower()
        if spec_l in _SUBSTRATE_PRESETS:
            return EpsConst(_SUBSTRATE_PRESETS[spec_l])
        if spec.endswith('.dat'):
            return EpsTable(spec)
        try:
            return EpsConst(float(spec))
        except ValueError:
            raise ValueError(
                '[error] Unsupported <substrate.eps>=<{}>!'.format(spec))

    raise ValueError(
        '[error] Unsupported <substrate.eps> type: <{}>!'.format(type(spec).__name__))


def _eps_repr(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    return repr(spec)


def _get_particle_pos(p: Any) -> np.ndarray:
    if hasattr(p, 'pos') and getattr(p, 'pos') is not None:
        return np.asarray(p.pos)
    if hasattr(p, 'pfull') and hasattr(p.pfull, 'pos'):
        return np.asarray(p.pfull.pos)
    raise AttributeError('[error] particle has no <pos> attribute')


def _shift_particle(p: Any, dxyz) -> None:
    if hasattr(p, 'shift'):
        p.shift(list(dxyz))
        return
    if hasattr(p, 'pfull') and hasattr(p.pfull, 'shift'):
        p.pfull.shift(list(dxyz))
        return
    raise AttributeError('[error] particle has no <shift> method')


def _count_faces(p: Any, fallback: int = -1) -> int:
    if hasattr(p, 'pfull') and hasattr(p.pfull, 'nfaces'):
        return int(p.pfull.nfaces)
    if hasattr(p, 'nfaces'):
        return int(p.nfaces)
    if hasattr(p, 'pos') and p.pos is not None:
        return int(len(p.pos))
    return int(fallback)
