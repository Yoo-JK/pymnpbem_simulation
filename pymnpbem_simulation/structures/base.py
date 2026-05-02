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


