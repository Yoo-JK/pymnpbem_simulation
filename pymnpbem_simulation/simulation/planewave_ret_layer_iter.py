import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from .planewave_ret_iter import _iter_options
from ..util import print_info


class PlaneWaveRetLayerIterRunner(SimulationRunner):
    """Iterative retarded plane-wave with substrate (BEMRetLayerIter, GMRES + ACA).

    YAML config 예시::

        structure:
          type: with_substrate
          base: { type: sphere, diameter: 30, mesh_density: 60 }
          substrate: { eps: glass, z_position: 0.0, z_shift: 1.0 }

        simulation:
          type: planewave_ret_layer_iter   # 또는 ret_layer_iter
          excitation: planewave
          enei_min: 500
          enei_max: 800
          n_wavelengths: 11
          polarizations: [[1, 0, 0]]
          propagation_dirs: [[0, 0, -1]]
          tab_n: 5
          iter:
            solver: gmres
            tol: 1.0e-6
            maxit: 200
            precond: hmat
            hmatrix: true
            htol: 1.0e-6
            kmax: [4, 100]

    BEMRetLayerIter 는 layer Green function 을 ACA 로 압축하므로 substrate 가 있는
    매우 큰 mesh 에서 메모리/시간을 모두 절약. 작은 mesh 에서는 dense BEMRetLayer
    가 더 빠를 수 있다.
    """

    def build_layer(self) -> Any:
        layer = getattr(self.p, '_mnpbem_layer', None)

        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)

        if layer is None:
            raise RuntimeError(
                    '[error] PlaneWaveRetLayerIterRunner: particle has no <_mnpbem_layer>; '
                    'use structure.type=with_substrate to enable substrate.')

        return layer

    def build_greentab(self,
            layer: Any,
            enei: np.ndarray) -> Any:
        from mnpbem.greenfun import GreenTabLayer

        tab_n = int(self.cfg['simulation'].get('tab_n', 5))
        tab_n = max(2, min(tab_n, len(enei)))

        tab = layer.tabspace(self.p)
        gt = GreenTabLayer(layer, tab = tab)

        enei_tab = np.linspace(float(enei[0]), float(enei[-1]), tab_n)
        gt.set(enei_tab)

        print_info(
                'GreenTabLayer (iter): tabulated at {} enei points ({:.1f}-{:.1f} nm)'.format(
                        tab_n, float(enei_tab[0]), float(enei_tab[-1])))

        return gt

    def build_excitation(self,
            layer: Any) -> Any:
        from mnpbem.simulation import PlaneWaveRetLayer

        pol = self.cfg['simulation'].get('polarizations', [[1, 0, 0]])
        prop = self.cfg['simulation'].get('propagation_dirs', [[0, 0, -1]])

        return PlaneWaveRetLayer(np.asarray(pol, dtype = float),
                np.asarray(prop, dtype = float), layer)

    def build_solver(self,
            layer: Any,
            greentab: Any) -> Any:
        from mnpbem.bem import BEMRetLayerIter

        opts = _iter_options(self.cfg)
        return BEMRetLayerIter(self.p, layer = layer, greentab = greentab, **opts)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        greentab = self.build_greentab(layer, enei)
        bem = self.build_solver(layer, greentab)
        exc = self.build_excitation(layer)

        n_wl = len(enei)
        n_pol = self._infer_n_pol()

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        print_info('PlaneWaveRetLayerIter: warming up at enei={:.1f} nm'.format(float(enei[0])))
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
                'solver_type': 'BEMRetLayerIter'}

    def _infer_n_pol(self) -> int:
        pol = self.cfg['simulation'].get('polarizations', [[1, 0, 0]])
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
