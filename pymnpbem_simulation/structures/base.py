from typing import Any, Dict, Tuple


class StructureBuilder(object):

    def __init__(self,
            cfg_struct: Dict[str, Any],
            cfg_materials: Dict[str, Any]) -> None:
        self.cfg_struct = cfg_struct
        self.cfg_materials = cfg_materials

    def build(self) -> Tuple[Any, Any, int]:
        raise NotImplementedError('[error] Subclass must implement build()')

    @property
    def name(self) -> str:
        return self.cfg_struct.get('type', 'unknown')


def build_structure(cfg_struct: Dict[str, Any],
        cfg_materials: Dict[str, Any]) -> Tuple[Any, Any, int]:

    from . import sphere, dimer_cube

    stype = cfg_struct.get('type', '').lower()

    match stype:

        case 'sphere':

            builder = sphere.SphereBuilder(cfg_struct, cfg_materials)

        case 'dimer_cube' | 'advanced_dimer_cube':

            builder = dimer_cube.DimerCubeBuilder(cfg_struct, cfg_materials)

        case _:

            raise ValueError('[error] Invalid <structure.type> = <{}>!'.format(stype))

    return builder.build()
