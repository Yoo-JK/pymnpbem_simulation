import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from .planewave_ret_iter import (_iter_options, _iter_hmatrix_options,
        _iter_preconditioner_options, _iter_schur_options)
from ..util import print_info


class EELSRetLayerIterRunner(SimulationRunner):
    """Iterative retarded EELS with substrate (BEMRetLayerIter, GMRES + ACA).

    MNPBEM has no dedicated ``EELSRetLayerIter`` or ``EELSRetLayer``
    excitation class.  We combine BEMRetLayerIter with the standard EELSRet
    excitation; the layer structure enters through the BEM solver while the
    electron beam external potential remains the free-space retarded form
    (same convention as the existing EelsRetLayerRunner using dense BEMRetLayer).

    YAML config example::

        structure:
          type: with_substrate
          base: { type: sphere, diameter: 30, mesh_density: 60 }
          substrate: { eps: glass, gap: 0.001 }

        simulation:
          type: ret_layer_iter
          excitation: eels
          electron:
            impact: [[35, 0]]
            energy_kev: 200
            width: 0.5
          enei_min: 450
          enei_max: 750
          n_wavelengths: 21
          tab_n: 5
          iter:
            solver: gmres
            tol: 1.0e-6
            maxit: 200
            precond: hmat
            hmatrix: auto
    """

    def build_layer(self) -> Any:
        layer = getattr(self.p, '_mnpbem_layer', None)

        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)

        if layer is None:
            raise RuntimeError(
                '[error] EELSRetLayerIterRunner: particle has no <_mnpbem_layer>; '
                'use structure.type=with_substrate to enable substrate.')

        return layer

    def build_greentab(self,
            layer: Any,
            enei: np.ndarray) -> Any:
        from mnpbem.greenfun import GreenTabLayer

        tab_n = int(self.cfg['simulation'].get('tab_n', len(enei)))
        tab_n = max(2, min(tab_n, len(enei)))

        tab = layer.tabspace(self.p)
        gt = GreenTabLayer(layer, tab = tab)

        enei_tab = np.linspace(float(enei[0]), float(enei[-1]), tab_n)
        gt.set(enei_tab)

        print_info(
            'GreenTabLayer (eels iter): tabulated at {} enei points ({:.1f}-{:.1f} nm)'.format(
                tab_n, float(enei_tab[0]), float(enei_tab[-1])))

        return gt

    def build_excitation(self) -> Any:
        from mnpbem.simulation import EELSRet

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

        return EELSRet(self.p, impact, width = width, vel = vel, cutoff = cutoff)

    def build_solver(self,
            layer: Any,
            greentab: Any) -> Any:
        from mnpbem.bem import BEMRetLayerIter

        opts = _iter_options(self.cfg)
        opts.update(_iter_hmatrix_options(self, self.p, self.cfg))
        opts.update(_iter_preconditioner_options(self, self.cfg))

        schur_iter_opts = _iter_schur_options(self, self.cfg)
        if schur_iter_opts:
            opts.update(schur_iter_opts)

        return self._construct_bem(BEMRetLayerIter, self.p,
                layer = layer, greentab = greentab, **opts)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        greentab = self.build_greentab(layer, enei)
        bem = self.build_solver(layer, greentab)
        exc = self.build_excitation()

        impact_shape = exc.impact.shape if hasattr(exc, 'impact') else (1, 2)
        n_imp = int(impact_shape[0])
        n_wl = len(enei)

        psurf = np.zeros((n_wl, n_imp))
        pbulk = np.zeros((n_wl, n_imp))
        prad = np.zeros((n_wl, n_imp))

        print_info('EELSRetLayerIter: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        warm_s = time.time() - t_warm

        ps0, pb0 = exc.loss(sig)
        psurf[0, :] = self._flatten_loss(ps0, n_imp)
        pbulk[0, :] = self._flatten_loss(pb0, n_imp)

        try:
            pr0, _ = exc.rad(sig)
            prad[0, :] = self._flatten_loss(pr0, n_imp)
        except Exception as e:
            print_info('  rad() failed at warmup: {}'.format(e))

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))
            ps_i, pb_i = exc.loss(sig)
            psurf[i, :] = self._flatten_loss(ps_i, n_imp)
            pbulk[i, :] = self._flatten_loss(pb_i, n_imp)

            try:
                pr_i, _ = exc.rad(sig)
                prad[i, :] = self._flatten_loss(pr_i, n_imp)
            except Exception:
                pass

            self.save_sigma_for_wavelength(sig, float(enei[i]))

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                    i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        # Map EELS quantities to (ext, sca, abs) schema for postprocess.
        ext = psurf.copy()
        sca = prad.copy()
        abs_ = pbulk.copy()

        peak_idx = int(np.argmax(ext[:, 0]))
        peak_wl = float(enei[peak_idx])
        peak_ext_x = float(ext[peak_idx, 0])

        print_info('peak loss = {:.3e} at {:.2f} nm'.format(peak_ext_x, peak_wl))
        print_info('total wall = {:.2f} min'.format(wall_s / 60.0))

        return {
            'wavelength': enei,
            'ext': ext,
            'sca': sca,
            'abs': abs_,
            'eels_loss': psurf,
            'pbulk': pbulk,
            'prad': prad,
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_imp,
            'solver_type': 'BEMRetLayerIter'}

    def _flatten_loss(self,
            val: Any,
            n_imp: int) -> np.ndarray:
        v = np.atleast_1d(np.asarray(val)).real.flatten()
        if v.size < n_imp:
            v = np.tile(v, n_imp)[:n_imp]
        return v[:n_imp]
