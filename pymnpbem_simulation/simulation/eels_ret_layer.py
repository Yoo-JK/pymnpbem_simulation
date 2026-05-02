import time

from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class EelsRetLayerRunner(SimulationRunner):
    """Retarded EELS excitation on a particle + planar substrate.

    MNPBEM Python port has no dedicated ``EELSRetLayer``. We combine
    ``BEMRetLayer`` with the standard ``EELSRet`` excitation; the layer
    structure influences the BEM matrix while the electron beam external
    potential remains the free-space form (this matches the MATLAB
    convention for thin substrates with the particle in the embedding
    medium).

    YAML config (simulation section)::

        simulation:
          type: ret_layer
          excitation: eels
          impact: [[35, 0]]            # nm; (x, y) per beam
          beam_energy_kev: 200
          beam_width: 0.5
          enei_min_eV: 1.5             # alternative: pass enei in nm directly
          enei_max_eV: 4.5
          n_wavelengths: 21
    """

    def build_layer(self) -> Any:
        layer = getattr(self.p, '_mnpbem_layer', None)

        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)

        if layer is None:
            raise RuntimeError(
                '[error] EelsRetLayerRunner: particle has no <_mnpbem_layer>; '
                'use structure.type=with_substrate to enable substrate.')

        return layer

    def build_excitation(self) -> Any:
        from mnpbem.simulation import EELSRet, EELSBase

        impact = self.cfg['simulation'].get('impact', None)

        if impact is None:
            raise ValueError(
                '[error] simulation.impact required for excitation=eels')

        impact = np.asarray(impact, dtype = float)
        if impact.ndim == 1:
            impact = impact.reshape(1, -1)

        width = float(self.cfg['simulation'].get('beam_width', 0.5))
        beam_keV = float(self.cfg['simulation'].get('beam_energy_kev', 200.0))
        vel = EELSBase.ene2vel(beam_keV * 1e3)

        return EELSRet(self.p, impact, width, vel)

    def build_solver(self,
            layer: Any) -> Any:
        from mnpbem.bem import BEMRetLayer

        return BEMRetLayer(self.p, layer)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        bem = self.build_solver(layer)
        exc = self.build_excitation()

        impact_shape = exc.impact.shape if hasattr(exc, 'impact') else (1, 2)
        n_imp = int(impact_shape[0])
        n_wl = len(enei)

        psurf = np.zeros((n_wl, n_imp))

        print_info('EelsRetLayer: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, float(enei[0])))
        loss = exc.loss(sig)
        warm_s = time.time() - t_warm

        psurf[0, :] = self._flatten_loss(loss, n_imp)

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, float(enei[i])))
            loss = exc.loss(sig)
            psurf[i, :] = self._flatten_loss(loss, n_imp)

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                eta = elapsed / (i + 1) * (n_wl - i - 1)
                print_info('  wl {}/{}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                    i + 1, n_wl, elapsed / 60.0, eta / 60.0))

        wall_s = time.time() - t_loop

        # Map EELS quantities to the (ext, sca, abs) schema for postprocess.
        # ext = surface loss, sca = 0, abs = ext (no decomposition here).
        ext = psurf.copy()
        sca = np.zeros_like(ext)
        abs_ = ext.copy()

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
            'wall_s': wall_s,
            'warmup_s': warm_s,
            'peak_idx': peak_idx,
            'peak_wl_nm': peak_wl,
            'peak_ext_x': peak_ext_x,
            'n_pol': n_imp}

    def _flatten_loss(self,
            val: Any,
            n_imp: int) -> np.ndarray:
        v = np.atleast_1d(np.asarray(val)).real.flatten()
        if v.size < n_imp:
            v = np.tile(v, n_imp)[:n_imp]
        return v[:n_imp]
