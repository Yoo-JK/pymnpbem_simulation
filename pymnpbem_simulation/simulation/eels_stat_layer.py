import time

from typing import Any, Dict

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class EELSStatLayerRunner(SimulationRunner):
    """Quasistatic EELS excitation on a particle + planar substrate.

    MNPBEM has no dedicated ``EELSStatLayer`` class. We combine
    ``BEMStatLayer`` with the standard ``EELSStat`` excitation; the layer
    structure influences the BEM matrix while the electron beam external
    potential remains the free-space quasistatic form (matches the MATLAB
    convention for thin substrates with the particle in the embedding medium).

    YAML config (simulation section)::

        simulation:
          type: stat_layer
          excitation: eels
          electron:
            impact: [[15, 0]]
            energy_kev: 200
            width: 0.5
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
                '[error] EELSStatLayerRunner: particle has no <_mnpbem_layer>; '
                'use structure.type=with_substrate to enable substrate.')

        return layer

    def build_excitation(self) -> Any:
        from mnpbem.simulation import EELSStat, EELSBase

        elec_cfg = self.cfg['simulation'].get('electron', dict())

        impact = np.atleast_2d(np.asarray(
                elec_cfg.get('impact', [15.0, 0.0]),
                dtype = np.float64))

        if impact.shape[1] != 2:
            raise ValueError(
                '[error] electron.impact must have shape (n_imp, 2), got <{}>'.format(
                    impact.shape))

        width = float(elec_cfg.get('width', 0.5))
        beam_keV = float(elec_cfg.get('energy_kev', 200.0))
        vel = EELSBase.ene2vel(beam_keV * 1e3)

        cutoff = elec_cfg.get('cutoff', None)
        if cutoff is not None:
            cutoff = float(cutoff)

        return EELSStat(self.p, impact, width, vel, cutoff = cutoff)

    def build_solver(self,
            layer: Any) -> Any:
        from mnpbem.bem import BEMStatLayer

        return BEMStatLayer(self.p, layer)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        bem = self.build_solver(layer)
        exc = self.build_excitation()

        impact_shape = exc.impact.shape if hasattr(exc, 'impact') else (1, 2)
        n_imp = int(impact_shape[0])
        n_wl = len(enei)

        psurf = np.zeros((n_wl, n_imp))
        pbulk = np.zeros((n_wl, n_imp))

        print_info('EELSStatLayer: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        ps0, pb0 = exc.loss(sig)
        warm_s = time.time() - t_warm

        psurf[0, :] = self._flatten_loss(ps0, n_imp)
        pbulk[0, :] = self._flatten_loss(pb0, n_imp)

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))
            ps_i, pb_i = exc.loss(sig)
            psurf[i, :] = self._flatten_loss(ps_i, n_imp)
            pbulk[i, :] = self._flatten_loss(pb_i, n_imp)

            self.save_sigma_for_wavelength(sig, float(enei[i]))

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                    i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        # Map EELS quantities to the (ext, sca, abs) schema for postprocess.
        # ext = surface loss, sca = 0, abs = bulk loss.
        ext = psurf.copy()
        sca = np.zeros_like(ext)
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
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_imp,
            'solver_type': 'BEMStatLayer'}

    def _flatten_loss(self,
            val: Any,
            n_imp: int) -> np.ndarray:
        v = np.atleast_1d(np.asarray(val)).real.flatten()
        if v.size < n_imp:
            v = np.tile(v, n_imp)[:n_imp]
        return v[:n_imp]
