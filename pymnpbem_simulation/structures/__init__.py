from typing import Any, Dict, Tuple
from ..material.material_descriptor import resolve_refractive_index_paths
import copy

from .base import StructureBuilder
from .sphere import SphereBuilder
from .cube import CubeBuilder
from .rod import RodBuilder
from .ellipsoid import EllipsoidBuilder
from .triangle import TriangleBuilder
from .dimer_sphere import DimerSphereBuilder
from .dimer_cube import DimerCubeBuilder
from .core_shell_sphere import CoreShellSphereBuilder
from .core_shell_cube import CoreShellCubeBuilder
from .core_shell_rod import CoreShellRodBuilder
from .dimer_core_shell_cube import DimerCoreShellCubeBuilder
from .advanced_monomer_cube import AdvancedMonomerCubeBuilder
from .advanced_dimer_cube import AdvancedDimerCubeBuilder
from .connected_dimer_cube import ConnectedDimerCubeBuilder
from .sphere_cluster import SphereClusterBuilder
from .from_shape import FromShapeBuilder
from .with_substrate import WithSubstrateBuilder
from .with_mirror import WithMirrorBuilder
from .with_nonlocal import WithNonlocalBuilder


REGISTRY = {
        'sphere': SphereBuilder,
        'cube': CubeBuilder,
        'rod': RodBuilder,
        'ellipsoid': EllipsoidBuilder,
        'triangle': TriangleBuilder,
        'dimer_sphere': DimerSphereBuilder,
        'dimer_cube': DimerCubeBuilder,
        'core_shell_sphere': CoreShellSphereBuilder,
        'core_shell_cube': CoreShellCubeBuilder,
        'core_shell_rod': CoreShellRodBuilder,
        'dimer_core_shell_cube': DimerCoreShellCubeBuilder,
        'advanced_monomer_cube': AdvancedMonomerCubeBuilder,
        'advanced_dimer_cube': AdvancedDimerCubeBuilder,
        'connected_dimer_cube': ConnectedDimerCubeBuilder,
        'sphere_cluster_aggregate': SphereClusterBuilder,
        'sphere_cluster': SphereClusterBuilder,
        'from_shape': FromShapeBuilder,
        'with_substrate': WithSubstrateBuilder,
        'with_mirror': WithMirrorBuilder,
        'with_nonlocal': WithNonlocalBuilder}


def build_structure(cfg_struct: Dict[str, Any],
        cfg_materials: Dict[str, Any]) -> Tuple[Any, Any, int]:
    stype = str(cfg_struct.get('type', '')).lower()

    if stype not in REGISTRY:
        raise ValueError('[error] Invalid <structure.type> = <{}>; '
                'available: {}'.format(stype, sorted(REGISTRY.keys())))

    cls = REGISTRY[stype]
    cfg_materials_local = copy.deepcopy(cfg_materials) if isinstance(cfg_materials, dict) else {}
    ri_paths = cfg_materials_local.get("refractive_index_paths", {})
    if isinstance(ri_paths, dict) and ri_paths:
        cfg_materials_local["refractive_index_paths"] = resolve_refractive_index_paths(ri_paths)
    builder = cls(cfg_struct, cfg_materials_local)
    # builder = cls(cfg_struct, cfg_materials)
    return builder.build()


__all__ = [
        'StructureBuilder',
        'build_structure',
        'REGISTRY',
        'SphereBuilder',
        'CubeBuilder',
        'RodBuilder',
        'EllipsoidBuilder',
        'TriangleBuilder',
        'DimerSphereBuilder',
        'DimerCubeBuilder',
        'CoreShellSphereBuilder',
        'CoreShellCubeBuilder',
        'CoreShellRodBuilder',
        'DimerCoreShellCubeBuilder',
        'AdvancedMonomerCubeBuilder',
        'AdvancedDimerCubeBuilder',
        'ConnectedDimerCubeBuilder',
        'SphereClusterBuilder',
        'FromShapeBuilder',
        'WithSubstrateBuilder',
        'WithMirrorBuilder',
        'WithNonlocalBuilder']
