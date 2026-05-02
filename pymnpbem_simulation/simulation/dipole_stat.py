import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class DipoleStatRunner(SimulationRunner):

    def build_excitation(self) -> Any:
        from mnpbem.simulation import DipoleStat
        from mnpbem.geometry import ComPoint

        dip_cfg = self.cfg['simulation'].get('dipole', dict())
        pos = np.atleast_2d(np.asarray(
                dip_cfg.get('position', [10.0, 0.0, 5.0]),
                dtype = np.float64))

        orient = dip_cfg.get('orientation', None)

        if orient is None:
            dip = None
        else:
            dip = np.asarray(orient, dtype = np.float64)

        medium_idx = int(dip_cfg.get('medium', 1))

        pt = ComPoint(self.p, pos, medium = medium_idx)

        if dip is None:

            return DipoleStat(pt)

        return DipoleStat(pt, dip)

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMStat

        return BEMStat(self.p, **self._bem_options())

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_dip = self._infer_n_pol(exc)

        decay_total = np.zeros((n_wl, n_dip))
        decay_radiative = np.zeros((n_wl, n_dip))
        decay_free = np.zeros((n_wl, n_dip))

        print_info('DipoleStat: warming up at enei={:.1f} nm'.format(enei[0]))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, enei[0]))
        warm_s = time.time() - t_warm

        tot, rad, rad0 = exc.decayrate(sig)
        decay_total[0, :] = self._flatten_decay(tot, n_dip)
        decay_radiative[0, :] = self._flatten_decay(rad, n_dip)
        decay_free[0, :] = self._flatten_decay(rad0, n_dip)

        print_info('warmup done in {:.1f}s'.format(warm_s))
        print_info('  decay_total[0] = {}'.format(decay_total[0, :].tolist()))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, enei[i]))

            tot, rad, rad0 = exc.decayrate(sig)
            decay_total[i, :] = self._flatten_decay(tot, n_dip)
            decay_radiative[i, :] = self._flatten_decay(rad, n_dip)
            decay_free[i, :] = self._flatten_decay(rad0, n_dip)

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                    i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        peak_idx = int(np.argmax(decay_total[:, 0]))
        peak_wl = float(enei[peak_idx])
        peak_decay = float(decay_total[peak_idx, 0])

        print_info('peak decay_total = {:.3f} at {:.2f} nm'.format(peak_decay, peak_wl))
        print_info('total wall = {:.2f} min'.format(wall_s / 60.0))

        return {
            'wavelength': enei,
            'ext': decay_total,
            'sca': decay_radiative,
            'abs': decay_total - decay_radiative,
            'decay_total': decay_total,
            'decay_radiative': decay_radiative,
            'decay_free': decay_free,
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_decay,
            'n_pol': n_dip}

    def _infer_n_pol(self,
            exc: Any) -> int:
        return int(exc.dip.shape[0]) * int(exc.dip.shape[2])

    @staticmethod
    def _flatten_decay(arr: Any,
            n_dip: int) -> np.ndarray:

        flat = np.asarray(arr).real.flatten()

        if flat.size < n_dip:
            out = np.zeros(n_dip)
            out[:flat.size] = flat

            return out

        return flat[:n_dip]
