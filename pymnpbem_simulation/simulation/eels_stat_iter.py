import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from .planewave_ret_iter import (_iter_options, _iter_hmatrix_options,
        _iter_preconditioner_options, _iter_schur_options)
from ..util import print_info


class EELSStatIterRunner(SimulationRunner):
    """Iterative quasistatic EELS BEM (BEMStatIter, GMRES + ACA H-matrix).

    YAML config example::

        simulation:
          type: stat_iter
          excitation: eels
          electron:
            impact: [[15, 0]]
            energy_kev: 200
            width: 0.5
          enei_min: 450
          enei_max: 750
          n_wavelengths: 21
          iter:
            solver: gmres
            tol: 1.0e-6
            maxit: 200
            precond: hmat
            hmatrix: auto
    """

    def build_excitation(self) -> Any:
        from mnpbem.simulation import EELSStat

        elec_cfg = self.cfg['simulation'].get('electron', dict())

        impact = np.atleast_2d(np.asarray(
                elec_cfg.get('impact', [15.0, 0.0]),
                dtype = np.float64))

        if impact.shape[1] != 2:
            raise ValueError(
                '[error] electron.impact must have shape (n_imp, 2), got <{}>'.format(
                    impact.shape))

        energy_kev = float(elec_cfg.get('energy_kev', 200.0))
        width = float(elec_cfg.get('width', 0.5))

        ene_ev = energy_kev * 1000.0
        vel = float(np.sqrt(1.0 - 1.0 / (1.0 + ene_ev / 0.51e6) ** 2))

        cutoff = elec_cfg.get('cutoff', None)
        if cutoff is not None:
            cutoff = float(cutoff)

        return EELSStat(self.p, impact, width = width, vel = vel, cutoff = cutoff)

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
        n_imp = int(exc.impact.shape[0])

        psurf = np.zeros((n_wl, n_imp))
        pbulk = np.zeros((n_wl, n_imp))

        print_info('EELSStatIter: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        warm_s = time.time() - t_warm

        ps0, pb0 = exc.loss(sig)
        psurf[0, :] = self._flatten(ps0, n_imp)
        pbulk[0, :] = self._flatten(pb0, n_imp)

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))
        print_info('  psurf[0] = {}'.format(psurf[0, :].tolist()))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))

            ps_i, pb_i = exc.loss(sig)
            psurf[i, :] = self._flatten(ps_i, n_imp)
            pbulk[i, :] = self._flatten(pb_i, n_imp)

            self.save_sigma_for_wavelength(sig, float(enei[i]))

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                    i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        # Map EELS quantities to (ext, sca, abs) schema for postprocess.
        ext = psurf.copy()
        sca = np.zeros_like(ext)
        abs_ = pbulk.copy()

        peak_idx = int(np.argmax(ext[:, 0]))
        peak_wl = float(enei[peak_idx])
        peak_ext_x = float(ext[peak_idx, 0])

        print_info('peak psurf = {:.3e} at {:.2f} nm'.format(peak_ext_x, peak_wl))
        print_info('total wall = {:.2f} min'.format(wall_s / 60.0))

        return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'psurf': psurf,
            'pbulk': pbulk,
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_imp,
            'solver_type': 'BEMStatIter'}

    @staticmethod
    def _flatten(arr: Any,
            n_imp: int) -> np.ndarray:
        flat = np.asarray(arr).real.flatten()
        if flat.size < n_imp:
            out = np.zeros(n_imp)
            out[:flat.size] = flat
            return out
        return flat[:n_imp]
