import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from .planewave_ret_iter import (_iter_options, _iter_hmatrix_options,
        _iter_preconditioner_options, _iter_schur_options)
from ..util import print_info


class PlaneWaveStatIterRunner(SimulationRunner):
    """Iterative quasistatic plane-wave BEM (BEMStatIter, GMRES + ACA H-matrix).

    YAML config example (예시)::

        simulation:
          type: planewave_stat_iter   # or stat_iter (또는 stat_iter)
          excitation: planewave
          enei_min: 500
          enei_max: 800
          n_wavelengths: 30
          polarizations: [[1, 0, 0], [0, 1, 0]]
          iter:
            solver: gmres
            tol: 1.0e-6
            maxit: 200
            precond: hmat
            hmatrix: auto             # auto | true | false (v1.3.0)
            htol: 1.0e-6
            kmax: [4, 100]
            cleaf: 200
            # new in v1.5.0 (v1.5.0 신규)
            preconditioner: auto      # auto | none | hlu_dense | hlu_tree
            htol_precond: 1.0e-4
            schur: auto               # auto | true | false (cover-layer auto / 자동)
            schur_g_ss_solver: auto   # auto | lu_dense | gmres
            schur_inner_tol: 1.0e-8

    ``hmatrix: auto`` activates ACA H-matrix Green functions only when
    the particle has more than 5000 faces.

    ``preconditioner: auto`` (v1.5.0) -> auto-select the H-matrix LU
    preconditioner. ``schur: auto`` (v1.5.0) -> auto-enable iter Schur
    reduction when a nonlocal cover-layer is detected (BEMStatIter components=1).
    (preconditioner: auto → H-matrix LU preconditioner 자동 선택.
     schur: auto → nonlocal cover-layer 감지 시 iter Schur reduction 자동 활성화.)
    """

    def build_excitation(self) -> Any:
        from mnpbem.simulation import PlaneWaveStat

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])

        return PlaneWaveStat(pol)

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMStatIter

        opts = _iter_options(self.cfg)
        opts.update(self._bem_options())

        hmatrix_opts = _iter_hmatrix_options(self, self.p, self.cfg)
        opts.update(hmatrix_opts)

        opts.update(_iter_preconditioner_options(self, self.cfg))

        schur_iter_opts = _iter_schur_options(self, self.cfg)
        if schur_iter_opts:
            opts.update(schur_iter_opts)

        return self._construct_bem(BEMStatIter, self.p, **opts)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_pol = self._infer_n_pol()

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        print_info('PlaneWaveStatIter: warming up at enei={:.1f} nm'.format(float(enei[0])))
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
                'solver_type': 'BEMStatIter'}

    def _infer_n_pol(self) -> int:
        pol = self.cfg['simulation'].get('polarizations', [[1, 0, 0], [0, 1, 0]])
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
