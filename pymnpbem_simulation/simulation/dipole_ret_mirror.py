import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class DipoleRetMirrorRunner(SimulationRunner):
    """Retarded dipole excitation near a mirror-symmetric particle (BEMRetMirror).

    YAML config example::

        structure:
          type: with_mirror
          base: { type: sphere, diameter: 30, mesh_density: 60 }
          mirror: { sym: xy }

        simulation:
          type: ret_mirror
          excitation: dipole
          dipole_position: [0, 0, 25]
          dipole_moment: [0, 0, 1]
          enei_min: 500
          enei_max: 800
          n_wavelengths: 11

    Constraints:
      - Dipole position z must be above the mirror plane (z > 0 for xy symmetry).
      - DipoleRetMirror requires the particle to have mirror symmetry (sym attribute).
    """

    def build_excitation(self) -> Any:
        from mnpbem.geometry import ComPoint
        from mnpbem.simulation import DipoleRetMirror

        pos = self.cfg['simulation'].get('dipole_position', [0.0, 0.0, 25.0])
        mom = self.cfg['simulation'].get('dipole_moment', [0.0, 0.0, 1.0])

        pos = np.asarray(pos, dtype = float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)

        mom = np.asarray(mom, dtype = float)
        if mom.ndim == 1:
            mom = mom.reshape(1, -1)

        pt = ComPoint(self.p, pos)
        return DipoleRetMirror(pt, dip = mom)

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMRetMirror

        return BEMRetMirror(self.p)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        dip = self.build_excitation()

        mom = self.cfg['simulation'].get('dipole_moment', [0.0, 0.0, 1.0])
        mom = np.asarray(mom, dtype = float)
        if mom.ndim == 1:
            mom = mom.reshape(1, -1)
        n_dip = mom.shape[0]

        n_wl = len(enei)

        tot = np.zeros((n_wl, n_dip))
        rad = np.zeros((n_wl, n_dip))

        print_info('DipoleRetMirror: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(dip(self.p, float(enei[0])))
        t_val, r_val, _ = dip.decayrate(sig)
        warm_s = time.time() - t_warm

        tot[0, :] = self._flatten_decay(t_val, n_dip)
        rad[0, :] = self._flatten_decay(r_val, n_dip)

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(dip(self.p, float(enei[i])))
            t_val, r_val, _ = dip.decayrate(sig)

            tot[i, :] = self._flatten_decay(t_val, n_dip)
            rad[i, :] = self._flatten_decay(r_val, n_dip)

            self.save_sigma_for_wavelength(sig, float(enei[i]))

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                    i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        ext = tot.copy()
        sca = rad.copy()
        abs_ = ext - sca

        peak_idx = int(np.argmax(ext[:, 0]))
        peak_wl = float(enei[peak_idx])
        peak_ext_x = float(ext[peak_idx, 0])

        print_info('peak total decay = {:.3f} at {:.2f} nm'.format(peak_ext_x, peak_wl))
        print_info('total wall = {:.2f} min'.format(wall_s / 60.0))

        return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'tot_decay': tot,
            'rad_decay': rad,
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_dip,
            'solver_type': 'BEMRetMirror'}

    def _flatten_decay(self,
            val: Any,
            n_dip: int) -> np.ndarray:
        v = np.atleast_1d(np.asarray(val)).real.flatten()
        if v.size < n_dip:
            v = np.tile(v, n_dip)[:n_dip]
        return v[:n_dip]
