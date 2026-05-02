import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class PlaneWaveStatRunner(SimulationRunner):

    def build_excitation(self) -> Any:
        from mnpbem.simulation import PlaneWaveStat

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])

        return PlaneWaveStat(pol)

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMStat

        return BEMStat(self.p)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_pol = self._infer_n_pol()

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        print_info('PlaneWaveStat: warming up at enei={:.1f} nm'.format(enei[0]))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, enei[0]))
        warm_s = time.time() - t_warm

        ext[0, :] = self._scalar_or_arr(exc.extinction(sig), n_pol)
        sca[0, :] = self._scalar_or_arr(exc.scattering(sig), n_pol)
        abs_[0, :] = self._scalar_or_arr(exc.absorption(sig), n_pol)

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, enei[i]))

            ext[i, :] = self._scalar_or_arr(exc.extinction(sig), n_pol)
            sca[i, :] = self._scalar_or_arr(exc.scattering(sig), n_pol)
            abs_[i, :] = self._scalar_or_arr(exc.absorption(sig), n_pol)

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
            'n_pol': n_pol}

    def _infer_n_pol(self) -> int:
        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])

        return len(pol)

    @staticmethod
    def _scalar_or_arr(val: Any,
            n_pol: int) -> np.ndarray:

        if isinstance(val, tuple):
            val = val[0]

        arr = np.asarray(val).real.flatten()

        if arr.size < n_pol:
            out = np.zeros(n_pol)
            out[:arr.size] = arr

            return out

        return arr[:n_pol]
