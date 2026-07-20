from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from ..util import print_info


_MATERIAL_DEFAULTS = {
    'water': 1.33 ** 2,
    'vacuum': 1.0,
    'air': 1.0,
    'glass': 1.5 ** 2}

# Available trisphere vertex counts (same list used by MATLAB trisphere.m)
_TRISPHERE_AVAILABLE = [
    32, 60, 144, 169, 225, 256, 289, 324, 361, 400,
    441, 484, 529, 576, 625, 676, 729, 784, 841, 900,
    961, 1024, 1225, 1444]


def _resolve_sphere_n(cfg: Dict) -> int:
    """Resolve trisphere vertex count from config.

    Priority:
    1. ``nphi`` (legacy): ``n = round(((diameter+1)*pi/nphi)^2 / 2)``
       snapped to nearest available count — mirrors OLD geometry_generator.py.
    2. ``n_verts`` / ``mesh_density`` (existing NEW key): used directly.
    3. Default: 256.
    """
    diameter = float(cfg.get('diameter', 50.0))

    if 'nphi' in cfg:
        nphi = float(cfg['nphi'])
        target = int(round(((diameter + 1) * np.pi / nphi) ** 2 / 2))
        n = min(_TRISPHERE_AVAILABLE,
                key = lambda x: abs(x - target))
        return n

    # existing keys: n_verts or mesh_density (float → treated as n directly)
    n = cfg.get('n_verts', cfg.get('mesh_density', 256))
    return int(n)


class SphereBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.materials import EpsConst, EpsTable
        from mnpbem.geometry import trisphere, ComParticle

        diameter = float(self.cfg_struct.get('diameter', 50.0))
        n = _resolve_sphere_n(self.cfg_struct)
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        medium_name = self.cfg_materials.get('medium', 'water')
        particle_name = self.cfg_materials.get('particle', 'gold')

        rip = _resolve_rip(self.cfg_struct, self.cfg_materials)
        eps_medium = _build_eps_medium(medium_name)
        eps_particle = _build_eps_particle(particle_name, rip)
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
    from mnpbem.materials import EpsConst, EpsTable, EpsDrude

    name_l = name.lower()

    if name_l in {'gold', 'au'}:
        return EpsTable('gold.dat')

    if name_l in {'silver', 'ag'}:
        return EpsTable('silver.dat')

    if name.endswith('.dat'):
        return EpsTable(name)

    # custom material from refractive_index_paths (e.g. agcl, ito).
    # Supports both descriptor dicts and runtime-resolved values:
    #   {'agcl': {'type': 'constant', 'epsilon': 2.02}} -> EpsConst(2.02)
    #   {'foo':  {'type': 'table', 'file': 'foo.dat'}}  -> EpsTable('foo.dat')
    #   {'foo': 'foo.dat'}                               -> EpsTable('foo.dat')
    #   {'foo': 2.02}                                    -> EpsConst(2.02)
    #   {'foo': <callable>}                              -> <callable>
    if isinstance(custom, dict) and custom:
        cmap = {str(k).lower(): v for k, v in custom.items()}
        if name_l in cmap:
            m = cmap[name_l]

            # Runtime value resolved by material_descriptor.py.
            if callable(m):
                return m

            if isinstance(m, (int, float)):
                return EpsConst(float(m))

            if isinstance(m, str):
                if m.endswith('.dat'):
                    return EpsTable(m)
                try:
                    return EpsConst(float(m))
                except ValueError:
                    return EpsTable(m)
            if isinstance(m, dict):
                mtype = str(m.get('type', 'constant')).lower()
                if mtype == 'constant':
                    if 'epsilon' not in m:
                        raise ValueError('[error] Missing <epsilon> for custom '
                                         'material <{}>!'.format(name))
                    return EpsConst(float(m['epsilon']))

                if mtype == 'table':
                    path = m.get('path', m.get('file', name))
                    return EpsTable(str(path))

                if mtype == 'python_module':
                    # If resolver is bypassed, allow direct callable injection.
                    fn = m.get('callable', None)
                    if callable(fn):
                        return fn
                    raise ValueError('[error] Unsupported unresolved '
                                     '<python_module> descriptor for material '
                                     '<{}>; run descriptor resolver first!'.format(name))

                # Legacy path-like dicts without explicit type.
                if 'path' in m or 'file' in m:
                    return EpsTable(str(m.get('path', m.get('file'))))

            raise ValueError('[error] Unsupported custom material spec for '
                             '<{}>: <{}>!'.format(name, type(m).__name__))

    raise ValueError('[error] Unsupported <particle> = <{}>!'.format(name))


def _resolve_materials_list(cfg_struct: Any, cfg_materials: Any) -> list:
    """Per-layer materials list, tolerating both config layouts.

    Direct .py configs carry the list under ``structure.materials``; the yaml
    migration (py_to_yaml) routes it to ``materials.particle_list``. Read the
    structure section first, then fall back to the materials section so CLI
    runs (migrated) and direct builder calls resolve to the same materials.
    """
    mats = (cfg_struct or {}).get('materials')
    if not mats:
        mats = (cfg_materials or {}).get('particle_list')
    return list(mats) if mats else []


def _resolve_rip(cfg_struct: Any, cfg_materials: Any) -> Any:
    """``refractive_index_paths`` from either config section (custom eps).

    Same rationale as :func:`_resolve_materials_list` — the migration routes
    ``refractive_index_paths`` into the materials section, so a builder reading
    only ``cfg_struct`` would silently lose custom dielectrics under the CLI.
    """
    return ((cfg_struct or {}).get('refractive_index_paths')
            or (cfg_materials or {}).get('refractive_index_paths')
            or None)


def _count_faces(p: Any) -> int:
    if hasattr(p, 'pfull'):
        return int(p.pfull.nfaces)

    if hasattr(p, 'nfaces'):
        return int(p.nfaces)

    if hasattr(p, 'pos'):
        return int(len(p.pos))

    return -1
