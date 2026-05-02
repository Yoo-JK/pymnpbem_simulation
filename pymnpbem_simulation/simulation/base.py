from typing import Any, Dict, Tuple

import numpy as np


class SimulationRunner(object):

    def __init__(self,
            cfg: Dict[str, Any],
            p: Any,
            epstab: Any) -> None:
        self.cfg = cfg
        self.p = p
        self.epstab = epstab

    def build_excitation(self) -> Any:
        raise NotImplementedError('[error] Subclass must implement build_excitation()')

    def build_solver(self) -> Any:
        raise NotImplementedError('[error] Subclass must implement build_solver()')

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError('[error] Subclass must implement run()')


def run_simulation(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    sim_type = cfg['simulation']['type']
    exc_type = cfg['simulation']['excitation']

    match (sim_type, exc_type):

        case ('ret', 'planewave'):

            from . import planewave_ret
            runner = planewave_ret.PlaneWaveRetRunner(cfg, p, epstab)

        case ('ret_layer', 'planewave'):

            from . import planewave_ret_layer
            runner = planewave_ret_layer.PlaneWaveRetLayerRunner(cfg, p, epstab)

        case ('ret_layer', 'dipole'):

            from . import dipole_ret_layer
            runner = dipole_ret_layer.DipoleRetLayerRunner(cfg, p, epstab)

        case ('ret_layer', 'eels'):

            from . import eels_ret_layer
            runner = eels_ret_layer.EelsRetLayerRunner(cfg, p, epstab)

        case _:

            raise ValueError(
                '[error] Unsupported (simulation.type, excitation) = (<{}>, <{}>)!'.format(
                    sim_type, exc_type))

    return runner.run(enei)
