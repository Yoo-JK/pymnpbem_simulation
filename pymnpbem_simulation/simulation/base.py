import inspect

from typing import Any, Dict, Tuple

import numpy as np


def _filter_kwargs_for(cls: Any,
        kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs that ``cls.__init__`` does not accept.

    Used so that v1.2.0 options (``schur``, future flags) degrade
    gracefully when running against an older mnpbem port that does not
    yet recognise them. Inspects ``cls.__init__``; if it accepts
    ``**kwargs`` we pass everything through unchanged.
    """
    if not kwargs:
        return kwargs
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return kwargs

    accepted = {name for name in params if name != 'self'}
    return {k: v for k, v in kwargs.items() if k in accepted}


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

    def _bem_options(self) -> Dict[str, Any]:
        """Optional BEM solver kwargs derived from particle metadata.

        Currently forwards:

        * the cover-layer refinement callable (``_mnpbem_refun`` set by
          WithNonlocalBuilder via ``coverlayer.refine``) to the
          underlying CompGreenStat. Only stat-path BEM solvers
          (BEMStat / BEMStatIter) consume ``refun``;
        * the v1.2.0 ``schur`` option for cover-layer Schur-complement
          elimination — auto-enabled when the particle exposes
          ``_mnpbem_refun`` (i.e. nonlocal cover-layer present), unless
          the user explicitly overrides via
          ``cfg['compute']['schur_complement']``.

        BEMRet does not yet expose ``refun`` in mnpbem. Nonlocal
        cover-layer simulations should therefore set
        ``simulation.type = 'stat'`` to match MATLAB MNPBEM
        demospecstat19.m.
        """
        opts: Dict[str, Any] = dict()
        refun = getattr(self.p, '_mnpbem_refun', None)
        if refun is None and hasattr(self.p, 'pfull'):
            refun = getattr(self.p.pfull, '_mnpbem_refun', None)
        if refun is not None:
            opts['refun'] = refun

        schur_setting = self._resolve_schur(refun is not None)
        if schur_setting is not None:
            opts['schur'] = schur_setting
        return opts

    def _construct_bem(self,
            cls: Any,
            *args: Any,
            **opts: Any) -> Any:
        """Instantiate a BEM solver class, gracefully dropping unknown kwargs.

        v1.2.0 introduces ``schur=...`` on BEMStat / BEMRet. v1.3.0
        introduces ``hmatrix=`` (and the companion ``htol`` / ``kmax`` /
        ``cleaf``) on BEMRetIter / BEMStatIter. When the installed
        mnpbem port (Agent α merge) does not yet recognise these flags
        we retry without them so the wrapper still functions during the
        rollout window.
        """
        try:
            return cls(*args, **opts)
        except TypeError as exc:
            msg = str(exc).lower()
            stripped = dict(opts)
            removed: list = []

            if 'schur' in stripped and 'schur' in msg:
                stripped.pop('schur', None)
                removed.append('schur')

            if 'hmatrix' in stripped and 'hmatrix' in msg:
                for v130_key in ('hmatrix', 'htol', 'kmax', 'cleaf'):
                    stripped.pop(v130_key, None)
                removed.append('hmatrix')

            if not removed:
                raise
            return cls(*args, **stripped)

    def _resolve_hmatrix(self,
            particle: Any,
            iter_cfg: Dict[str, Any],
            face_threshold: int = 5000) -> bool:
        """Resolve the v1.3.0 ``hmatrix`` flag from the iter block.

        * ``'auto'`` (default) -> ``True`` if particle has more than
          ``face_threshold`` faces, else ``False``.
        * explicit ``True`` / ``'true'`` / ``'True'`` -> ``True``.
        * explicit ``False`` / ``'false'`` / ``'False'`` -> ``False``.
        * missing key -> behaves like ``'auto'``.
        """
        if not isinstance(iter_cfg, dict):
            iter_cfg = dict()

        raw = iter_cfg.get('hmatrix', 'auto')

        if raw is None:
            raw = 'auto'

        if isinstance(raw, str):
            tag = raw.strip().lower()
        else:
            tag = raw

        if tag == 'auto':
            n = self._particle_face_count(particle)
            return n > face_threshold

        if tag is True or tag == 'true':
            return True

        if tag is False or tag == 'false':
            return False

        # Unknown value -> auto.
        n = self._particle_face_count(particle)
        return n > face_threshold

    @staticmethod
    def _particle_face_count(particle: Any) -> int:
        if particle is None:
            return 0

        for attr in ('n', 'nfaces'):
            v = getattr(particle, attr, None)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)

        pfull = getattr(particle, 'pfull', None)
        if pfull is not None:
            for attr in ('n', 'nfaces'):
                v = getattr(pfull, attr, None)
                if isinstance(v, (int, np.integer)) and v > 0:
                    return int(v)

        return 0

    def _resolve_schur(self,
            has_cover_layer: bool) -> Any:
        """Resolve the v1.2.0 Schur-complement option.

        Reads ``cfg['compute']['schur_complement']`` (default ``'auto'``).

        * ``'auto'`` (default) -> ``True`` if cover layer detected else ``None``
          (skip kwarg to keep backward compat with pre-v1.2.0 mnpbem).
        * explicit ``True`` / ``'true'`` -> ``True``
        * explicit ``False`` / ``'false'`` -> ``False``
        * ``None`` / missing key -> behaves like ``'auto'``
        """
        compute = self.cfg.get('compute', dict()) if isinstance(self.cfg, dict) else dict()
        raw = compute.get('schur_complement', 'auto')

        if raw is None:
            raw = 'auto'

        if isinstance(raw, str):
            tag = raw.strip().lower()
        else:
            tag = raw

        if tag == 'auto':
            return True if has_cover_layer else None
        if tag is True or tag == 'true':
            return True
        if tag is False or tag == 'false':
            return False
        # Unknown value -> treat as auto.
        return True if has_cover_layer else None


def _get_registry() -> Dict[str, Any]:
    from . import planewave_ret, planewave_stat
    from . import dipole_ret, dipole_stat
    from . import eels_ret, eels_stat
    from . import field_calculator
    from . import planewave_ret_layer, dipole_ret_layer, eels_ret_layer
    from . import planewave_ret_iter, planewave_stat_iter, planewave_ret_layer_iter
    from . import planewave_ret_mirror

    return {
            ('ret', 'planewave'): planewave_ret.PlaneWaveRetRunner,
            ('stat', 'planewave'): planewave_stat.PlaneWaveStatRunner,
            ('ret', 'dipole'): dipole_ret.DipoleRetRunner,
            ('stat', 'dipole'): dipole_stat.DipoleStatRunner,
            ('ret', 'eels'): eels_ret.EELSRetRunner,
            ('stat', 'eels'): eels_stat.EELSStatRunner,
            ('ret_layer', 'planewave'): planewave_ret_layer.PlaneWaveRetLayerRunner,
            ('ret_layer', 'dipole'): dipole_ret_layer.DipoleRetLayerRunner,
            ('ret_layer', 'eels'): eels_ret_layer.EelsRetLayerRunner,
            ('ret_iter', 'planewave'): planewave_ret_iter.PlaneWaveRetIterRunner,
            ('stat_iter', 'planewave'): planewave_stat_iter.PlaneWaveStatIterRunner,
            ('ret_layer_iter', 'planewave'): planewave_ret_layer_iter.PlaneWaveRetLayerIterRunner,
            ('ret_mirror', 'planewave'): planewave_ret_mirror.PlaneWaveRetMirrorRunner,
            'planewave_ret': planewave_ret.PlaneWaveRetRunner,
            'planewave_stat': planewave_stat.PlaneWaveStatRunner,
            'dipole_ret': dipole_ret.DipoleRetRunner,
            'dipole_stat': dipole_stat.DipoleStatRunner,
            'eels_ret': eels_ret.EELSRetRunner,
            'eels_stat': eels_stat.EELSStatRunner,
            'field': field_calculator.FieldCalculator,
            'planewave_ret_layer': planewave_ret_layer.PlaneWaveRetLayerRunner,
            'dipole_ret_layer': dipole_ret_layer.DipoleRetLayerRunner,
            'eels_ret_layer': eels_ret_layer.EelsRetLayerRunner,
            'planewave_ret_iter': planewave_ret_iter.PlaneWaveRetIterRunner,
            'planewave_stat_iter': planewave_stat_iter.PlaneWaveStatIterRunner,
            'planewave_ret_layer_iter': planewave_ret_layer_iter.PlaneWaveRetLayerIterRunner,
            'planewave_ret_mirror': planewave_ret_mirror.PlaneWaveRetMirrorRunner}


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
