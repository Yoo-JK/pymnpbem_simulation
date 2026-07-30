import time

from typing import Any, Dict, List, Optional

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class PlaneWaveStatLayerRunner(SimulationRunner):
    """Quasistatic plane-wave excitation on a particle + planar substrate.

    Uses BEMStatLayer + PlaneWaveStatLayer.  Looks up the LayerStructure
    attached to the particle by WithSubstrateBuilder via ``_mnpbem_layer``.

    This runner also fixes the latent crash that occurred when config
    auto-promoted ``stat`` + substrate to ``stat_layer`` but no runner
    existed for that combination.

    YAML config (simulation section)::

        simulation:
          type: stat_layer
          excitation: planewave
          polarizations: [[1, 0, 0], [0, 1, 0]]
          propagation_dirs: [[0, 0, -1]]
          enei_min: 450
          enei_max: 750
          n_wavelengths: 11
    """

    def build_layer(self) -> Any:
        layer = getattr(self.p, '_mnpbem_layer', None)

        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)

        if layer is None:
            raise RuntimeError(
                '[error] PlaneWaveStatLayerRunner: particle has no <_mnpbem_layer>; '
                'use structure.type=with_substrate to enable substrate.')

        return layer

    def build_excitation(self,
            layer: Any) -> Any:
        from mnpbem.simulation import PlaneWaveStatLayer

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])
        prop = self.cfg['simulation'].get('propagation_dirs',
                [[0, 0, -1], [0, 0, -1]])

        pol = np.asarray(pol, dtype = float)
        prop = np.asarray(prop, dtype = float)

        return PlaneWaveStatLayer(pol, prop, layer)

    def build_solver(self,
            layer: Any) -> Any:
        from mnpbem.bem import BEMStatLayer

        return BEMStatLayer(self.p, layer)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        bem = self.build_solver(layer)
        exc = self.build_excitation(layer)

        n_wl = len(enei)
        n_pol = self._infer_n_pol()

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        print_info('PlaneWaveStatLayer: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        warm_s = time.time() - t_warm

        ext[0, :] = self._extract(exc.extinction(sig), n_pol)
        sca[0, :] = self._extract(exc.scattering(sig), n_pol)
        abs_[0, :] = self._extract(exc.absorption(sig), n_pol)

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))

            ext[i, :] = self._extract(exc.extinction(sig), n_pol)
            sca[i, :] = self._extract(exc.scattering(sig), n_pol)
            abs_[i, :] = self._extract(exc.absorption(sig), n_pol)

            self.save_sigma_for_wavelength(sig, float(enei[i]))

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
            'solver_type': 'BEMStatLayer'}

    def _infer_n_pol(self) -> int:
        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])
        return len(pol)

    @staticmethod
    def _extract(val: Any,
            n_pol: int) -> np.ndarray:
        if isinstance(val, tuple):
            val = val[0]
        arr = np.atleast_1d(np.asarray(val)).real.flatten()
        if arr.size < n_pol:
            out = np.zeros(n_pol)
            out[:arr.size] = arr
            return out
        return arr[:n_pol]
