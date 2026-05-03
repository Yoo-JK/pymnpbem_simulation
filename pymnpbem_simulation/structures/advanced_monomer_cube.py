import warnings

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import StructureBuilder
from .sphere import _build_eps_medium, _build_eps_particle, _count_faces
from ..util import print_info


_DEFAULT_N_PER_EDGE = 16


def _resolve_roundings(cfg: Dict[str, Any], n_layers: int) -> List[float]:
    if 'roundings' in cfg:
        roundings = list(cfg['roundings'])
        assert len(roundings) == n_layers, \
            '[error] <roundings> length must match number of layers (core + shells)'
        return [float(r) for r in roundings]

    e = float(cfg.get('e', cfg.get('rounding', 0.25)))
    return [e] * n_layers


def _layer_sizes(cfg: Dict[str, Any], n_layers: int) -> List[float]:
    core_size = float(cfg.get('core_size',
            cfg.get('size', cfg.get('edge', 30.0))))
    shell_layers = list(cfg.get('shell_layers', []))

    sizes = [core_size]
    for shell in shell_layers:
        if isinstance(shell, dict):
            t = float(shell.get('thickness', 0.0))
        else:
            t = float(shell)
        sizes.append(sizes[-1] + 2.0 * t)

    if len(sizes) < n_layers:
        sizes = sizes + [sizes[-1]] * (n_layers - len(sizes))

    return sizes[:n_layers]


def _n_per_edge_from_density(edge: float, mesh_density: float) -> int:
    n = int(round(float(edge) / float(mesh_density)))
    return max(2, n)


def _resolve_n_per_edge(cfg: Dict[str, Any],
        n_layers: int,
        edge_override: Optional[float] = None) -> List[int]:
    if 'n_per_edges' in cfg:
        nps = list(cfg['n_per_edges'])
        assert len(nps) == n_layers, \
            '[error] <n_per_edges> length must match number of layers'
        return [int(n) for n in nps]

    mesh_density = cfg.get('mesh_density', None)
    explicit_n = cfg.get('n_per_edge', None)

    if mesh_density is not None:
        if edge_override is not None:
            sizes = [float(edge_override)] * n_layers
        else:
            sizes = _layer_sizes(cfg, n_layers)

        outermost = sizes[-1]
        n_outer = _n_per_edge_from_density(outermost, mesh_density)

        if explicit_n is not None:
            try:
                explicit_int = int(explicit_n)
            except (TypeError, ValueError):
                explicit_int = None
            if explicit_int is not None and explicit_int != n_outer:
                warnings.warn(
                    'mesh_density={} (-> n_per_edge={}) overrides explicit '
                    'n_per_edge={}. mesh_density takes priority since v1.6.0.'.format(
                        mesh_density, n_outer, explicit_int),
                    stacklevel = 2)

        return [n_outer] * n_layers

    if explicit_n is None:
        return [_DEFAULT_N_PER_EDGE] * n_layers

    return [int(explicit_n)] * n_layers


class AdvancedMonomerCubeBuilder(StructureBuilder):

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import tricube, ComParticle

        core_size = float(self.cfg_struct.get('core_size', 30.0))
        shell_layers = list(self.cfg_struct.get('shell_layers', []))
        materials = list(self.cfg_struct.get('materials', []))
        refine = int(self.cfg_struct.get('refine', 2))
        interp = self.cfg_struct.get('interp', 'curv')

        n_layers = 1 + len(shell_layers)

        if not materials:
            base_particle = self.cfg_materials.get('particle', 'gold')
            base_shell = self.cfg_materials.get('shell', 'silver')
            materials = [base_particle] + [base_shell] * len(shell_layers)

        assert len(materials) == n_layers, \
            '[error] <materials> length must equal 1 (core) + len(shell_layers)'

        roundings = _resolve_roundings(self.cfg_struct, n_layers)
        n_per_edges = _resolve_n_per_edge(self.cfg_struct, n_layers)

        sizes = [core_size]
        for thickness in shell_layers:
            sizes.append(sizes[-1] + 2.0 * float(thickness))

        medium_name = self.cfg_materials.get('medium', 'water')
        eps_medium = _build_eps_medium(medium_name)
        eps_layers = [_build_eps_particle(name) for name in materials]
        epstab = [eps_medium] + eps_layers

        particles = []
        for size, n_e, e in zip(sizes, n_per_edges, roundings):
            cube = tricube(n_e, size, e = e)
            particles.append(cube)

        inout = []
        for i in range(n_layers):
            mat_idx = 2 + i
            if i == 0:
                if n_layers == 1:
                    inout.append([mat_idx, 1])
                else:
                    inout.append([mat_idx, mat_idx + 1])
            elif i == n_layers - 1:
                inout.append([mat_idx, 1])
            else:
                inout.append([mat_idx, mat_idx + 1])

        p = ComParticle(epstab, particles, inout,
                interp = interp, refine = refine)

        nfaces = _count_faces(p)
        print_info('AdvancedMonomerCubeBuilder: core={}nm, shells={}, mats={}, nfaces={}'.format(
            core_size, shell_layers, materials, nfaces))

        return p, epstab, nfaces
