import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class PlaneWaveRetMirrorRunner(SimulationRunner):
    """Retarded plane-wave on a mirror-symmetric particle (BEMRetMirror).

    YAML config example (예시)::

        structure:
          type: with_mirror
          base: { type: sphere, diameter: 30, mesh_density: 60 }
          mirror: { sym: xy }

        simulation:
          type: ret_mirror
          excitation: planewave
          enei_min: 500
          enei_max: 800
          n_wavelengths: 11
          polarizations: [[1, 0, 0], [0, 1, 0]]    # z-pol forbidden (mirror constraint) / z-pol 금지 (mirror 제약)
          propagation_dirs: [[0, 0, 1], [0, 0, 1]]

    Constraints (제약):
      - PlaneWaveRetMirror only supports pol[:, 2] == 0 (in-plane polarization)
        (PlaneWaveRetMirror 는 pol[:, 2] == 0 만 지원 — in-plane polarization)
      - dir is only meaningful as [0, 0, ±1]; other directions break the mirror symmetry.
        (dir 는 [0, 0, ±1] 형태만 의미 — 다른 방향은 mirror 대칭을 깨뜨림.)
    """

    def build_excitation(self) -> Any:
        from mnpbem.simulation import PlaneWaveRetMirror

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])
        prop = self.cfg['simulation'].get('propagation_dirs',
                [[0, 0, 1], [0, 0, 1]])

        return PlaneWaveRetMirror(np.asarray(pol, dtype = float),
                np.asarray(prop, dtype = float))

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMRetMirror

        return BEMRetMirror(self.p)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_pol = self._infer_n_pol()

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        print_info('PlaneWaveRetMirror: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        warm_s = time.time() - t_warm

        ev = self._extract(exc.extinction(sig), n_pol)
        sv = self._extract(exc.scattering(sig), n_pol)

        ext[0, :] = ev
        sca[0, :] = sv
        abs_[0, :] = ext[0, :] - sca[0, :]

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))
            ev = self._extract(exc.extinction(sig), n_pol)
            sv = self._extract(exc.scattering(sig), n_pol)

            ext[i, :] = ev
            sca[i, :] = sv
            abs_[i, :] = ext[i, :] - sca[i, :]

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
                'solver_type': 'BEMRetMirror'}

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
