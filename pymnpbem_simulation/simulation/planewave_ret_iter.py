import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class PlaneWaveRetIterRunner(SimulationRunner):
    """Iterative retarded plane-wave BEM (BEMRetIter, GMRES + ACA H-matrix).

    YAML config 예시::

        simulation:
          type: planewave_ret_iter      # 또는 ret_iter
          excitation: planewave
          enei_min: 500
          enei_max: 800
          n_wavelengths: 30
          polarizations: [[1, 0, 0], [0, 1, 0]]
          propagation_dirs: [[0, 0, 1], [0, 0, 1]]
          iter:
            solver: gmres               # gmres / cgs / bicgstab
            tol: 1.0e-6
            maxit: 200
            restart: null               # null = scipy default
            precond: hmat               # hmat / null
            output: 0
            hmatrix: auto               # auto | true | false (v1.3.0)
            htol: 1.0e-6
            kmax: [4, 100]
            cleaf: 200

    ``hmatrix: auto`` (default) -> mesh face count 가 5000 을 초과할 때만
    ACA H-matrix Green function 을 활성화한다. 작은 mesh 에서는 dense
    G/H 가 더 빠르므로 auto 가 합리적인 기본값.

    BEMRetIter 와 BEMRet 의 결과 차이는 GMRES tolerance (기본 1e-6) 만큼.
    매우 큰 mesh (수천 face 이상) 에서는 메모리/시간 모두 dense 보다 효율적.
    """

    def build_excitation(self) -> Any:
        from mnpbem.simulation import PlaneWaveRet

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])
        prop = self.cfg['simulation'].get('propagation_dirs',
                [[0, 0, 1], [0, 0, 1]])

        return PlaneWaveRet(pol, prop)

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMRetIter

        opts = _iter_options(self.cfg)
        bem_opts = self._bem_options()
        bem_opts.pop('refun', None)  # BEMRetIter does not yet consume refun
        opts.update(bem_opts)
        opts.update(_iter_hmatrix_options(self, self.p, self.cfg))
        return self._construct_bem(BEMRetIter, self.p, **opts)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_pol = self._infer_n_pol()

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        print_info('PlaneWaveRetIter: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        warm_s = time.time() - t_warm

        ev = self._extract(exc.extinction(sig), n_pol)
        sv = self._extract(exc.scattering(sig), n_pol)

        ext[0, :] = ev
        sca[0, :] = sv
        abs_[0, :] = ext[0, :] - sca[0, :]

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))
            ev = self._extract(exc.extinction(sig), n_pol)
            sv = self._extract(exc.scattering(sig), n_pol)

            ext[i, :] = ev
            sca[i, :] = sv
            abs_[i, :] = ext[i, :] - sca[i, :]

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                        i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        peak_idx = int(np.argmax(ext[:, 0]))
        peak_wl = float(enei[peak_idx])
        peak_ext_x = float(ext[peak_idx, 0])

        print_info('peak ext_x = {:.3f} at {:.2f} nm'.format(peak_ext_x, peak_wl))
        print_info('total wall = {:.2f} min'.format(wall_s / 60.0))

        return {
                'wavelength': enei,
                'ext': ext,
                'sca': sca,
                'abs': abs_,
                'wall_s': wall_s,
                'warmup_s': warm_s,
                'peak_idx': peak_idx,
                'peak_wl_nm': peak_wl,
                'peak_ext_x': peak_ext_x,
                'n_pol': n_pol,
                'solver_type': 'BEMRetIter'}

    def _infer_n_pol(self) -> int:
        pol = self.cfg['simulation'].get('polarizations', [[1, 0, 0], [0, 1, 0]])
        return len(pol)

    def _extract(self,
            raw: Any,
            n_pol: int) -> np.ndarray:
        if isinstance(raw, tuple):
            raw = raw[0]
        arr = np.atleast_1d(np.asarray(raw)).real.flatten()
        if arr.size < n_pol:
            arr = np.tile(arr, n_pol)[:n_pol]
        return arr[:n_pol]


def _iter_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """GMRES knobs only. The v1.3.0 ``hmatrix`` flag is resolved
    separately via :func:`_iter_hmatrix_options` because its default
    (``'auto'``) depends on the particle face count, which the bare cfg
    does not know about.
    """
    iter_cfg = cfg.get('simulation', {}).get('iter', {}) or {}

    out = {
            'solver':  iter_cfg.get('solver', 'gmres'),
            'tol':     float(iter_cfg.get('tol', 1.0e-6)),
            'maxit':   int(iter_cfg.get('maxit', iter_cfg.get('maxiter', 200))),
            'restart': iter_cfg.get('restart', None),
            'precond': iter_cfg.get('precond', 'hmat'),
            'output':  int(iter_cfg.get('output', 0))}

    return out


def _iter_hmatrix_options(runner: SimulationRunner,
        particle: Any,
        cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve v1.3.0 ``hmatrix`` flag and forward companion knobs.

    Returns an empty dict when H-matrix mode is OFF, so callers can
    ``opts.update(...)`` unconditionally.
    """
    iter_cfg = cfg.get('simulation', {}).get('iter', {}) or {}
    if not isinstance(iter_cfg, dict):
        iter_cfg = dict()

    active = runner._resolve_hmatrix(particle, iter_cfg)

    if not active:
        return dict()

    out: Dict[str, Any] = {'hmatrix': True}

    out['htol'] = float(iter_cfg.get('htol', 1.0e-6))
    out['cleaf'] = int(iter_cfg.get('cleaf', 200))

    kmax = iter_cfg.get('kmax', [4, 100])
    if isinstance(kmax, (list, tuple)):
        out['kmax'] = list(kmax)
    else:
        out['kmax'] = kmax

    return out
