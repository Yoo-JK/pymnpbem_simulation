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


def _get_registry() -> Dict[str, Any]:
    from . import planewave_ret, planewave_stat
    from . import dipole_ret, dipole_stat
    from . import eels_ret, eels_stat

    return {
            ('ret', 'planewave'): planewave_ret.PlaneWaveRetRunner,
            ('stat', 'planewave'): planewave_stat.PlaneWaveStatRunner,
            ('ret', 'dipole'): dipole_ret.DipoleRetRunner,
            ('stat', 'dipole'): dipole_stat.DipoleStatRunner,
            ('ret', 'eels'): eels_ret.EELSRetRunner,
            ('stat', 'eels'): eels_stat.EELSStatRunner,
            'planewave_ret': planewave_ret.PlaneWaveRetRunner,
            'planewave_stat': planewave_stat.PlaneWaveStatRunner,
            'dipole_ret': dipole_ret.DipoleRetRunner,
            'dipole_stat': dipole_stat.DipoleStatRunner,
            'eels_ret': eels_ret.EELSRetRunner,
            'eels_stat': eels_stat.EELSStatRunner}


class _LazyRegistry(object):

    def __init__(self) -> None:
        self._cache = None

    def _ensure(self) -> Dict[str, Any]:

        if self._cache is None:
            self._cache = _get_registry()

        return self._cache

    def __getitem__(self, key: Any) -> Any:
        return self._ensure()[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._ensure()

    def keys(self) -> Any:
        return self._ensure().keys()

    def items(self) -> Any:
        return self._ensure().items()


REGISTRY = _LazyRegistry()


def build_simulation(p: Any,
        epstab: Any,
        cfg: Dict[str, Any]) -> Any:

    sim_type = cfg['simulation']['type']
    exc_type = cfg['simulation']['excitation']

    key = (sim_type, exc_type)

    if key not in REGISTRY:
        raise ValueError(
                '[error] Unsupported (simulation.type, excitation) = (<{}>, <{}>)!'.format(
                        sim_type, exc_type))

    cls = REGISTRY[key]

    return cls(cfg, p, epstab)


def run_simulation(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    sim_type = cfg['simulation']['type']
    exc_type = cfg['simulation']['excitation']

    key = (sim_type, exc_type)

    if key not in REGISTRY:
        raise ValueError(
                '[error] Unsupported (simulation.type, excitation) = (<{}>, <{}>)!'.format(
                        sim_type, exc_type))

    cls = REGISTRY[key]
    runner = cls(cfg, p, epstab)

    return runner.run(enei)
