import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from .planewave_ret_iter import (_iter_options, _iter_hmatrix_options,
        _iter_preconditioner_options, _iter_schur_options)
from ..util import print_info


class DipoleRetLayerIterRunner(SimulationRunner):
    """Iterative retarded dipole with substrate (BEMRetLayerIter, GMRES + ACA).

    MNPBEM has no dedicated ``DipoleRetLayerIter`` excitation class.  We
    combine BEMRetLayerIter with the standard DipoleRet excitation; the
    layer structure enters through the BEM solver while the dipole external
    potential remains the free-space retarded form (same convention as the
    existing DipoleRetLayerRunner using dense BEMRetLayer).

    YAML config example::

        structure:
          type: with_substrate
          base: { type: sphere, diameter: 30, mesh_density: 60 }
          substrate: { eps: glass, gap: 0.001 }

        simulation:
          type: ret_layer_iter
          excitation: dipole
          dipole_position: [0, 0, 25]
          dipole_moment: [0, 0, 1]
          enei_min: 500
          enei_max: 800
          n_wavelengths: 11
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
                '[error] DipoleRetLayerIterRunner: particle has no <_mnpbem_layer>; '
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
            'GreenTabLayer (dipole iter): tabulated at {} enei points ({:.1f}-{:.1f} nm)'.format(
                tab_n, float(enei_tab[0]), float(enei_tab[-1])))

        return gt

    def build_excitation(self) -> Any:
        from mnpbem.geometry import ComPoint
        from mnpbem.simulation import DipoleRet

        pos = self.cfg['simulation'].get('dipole_position', [0.0, 0.0, 25.0])
        mom = self.cfg['simulation'].get('dipole_moment', [0.0, 0.0, 1.0])

        pos = np.asarray(pos, dtype = float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)

        mom = np.asarray(mom, dtype = float)
        if mom.ndim == 1:
            mom = mom.reshape(1, -1)

        pt = ComPoint(self.p, pos)
        return DipoleRet(pt, dip = mom)

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
        dip = self.build_excitation()

        mom = self.cfg['simulation'].get('dipole_moment', [0.0, 0.0, 1.0])
        mom = np.asarray(mom, dtype = float)
        if mom.ndim == 1:
            mom = mom.reshape(1, -1)
        n_dip = mom.shape[0]

        n_wl = len(enei)

        decay_total = np.zeros((n_wl, n_dip))
        decay_radiative = np.zeros((n_wl, n_dip))
        decay_free = np.zeros((n_wl, n_dip))

        print_info('DipoleRetLayerIter: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(dip(self.p, float(enei[0])))
        warm_s = time.time() - t_warm

        tot, rad, rad0 = dip.decayrate(sig)
        decay_total[0, :] = self._flatten_decay(tot, n_dip)
        decay_radiative[0, :] = self._flatten_decay(rad, n_dip)
        decay_free[0, :] = self._flatten_decay(rad0, n_dip)

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))
        print_info('  decay_total[0] = {}'.format(decay_total[0, :].tolist()))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(dip(self.p, float(enei[i])))

            tot, rad, rad0 = dip.decayrate(sig)
            decay_total[i, :] = self._flatten_decay(tot, n_dip)
            decay_radiative[i, :] = self._flatten_decay(rad, n_dip)
            decay_free[i, :] = self._flatten_decay(rad0, n_dip)

            self.save_sigma_for_wavelength(sig, float(enei[i]))

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
            'n_pol': n_dip,
            'solver_type': 'BEMRetLayerIter'}

    def _flatten_decay(self,
            val: Any,
            n_dip: int) -> np.ndarray:
        v = np.atleast_1d(np.asarray(val)).real.flatten()
        if v.size < n_dip:
            v = np.tile(v, n_dip)[:n_dip]
        return v[:n_dip]
