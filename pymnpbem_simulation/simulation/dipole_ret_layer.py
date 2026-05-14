import time

from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class DipoleRetLayerRunner(SimulationRunner):
    """Retarded dipole excitation near a particle + planar substrate.

    Looks up the LayerStructure attached to the particle by
    WithSubstrateBuilder via ``_mnpbem_layer``. Computes the total and
    radiative decay rate for each wavelength.

    YAML config (simulation section)::

        simulation:
          type: ret_layer
          excitation: dipole
          dipole_position: [0, 0, 25]
          dipole_moment: [0, 0, 1]      # z-oriented; or list-of-3-vec for multi
          enei_min: 500
          enei_max: 700
          n_wavelengths: 11
    """

    def build_layer(self) -> Any:
        layer = getattr(self.p, '_mnpbem_layer', None)

        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)

        if layer is None:
            raise RuntimeError(
                '[error] DipoleRetLayerRunner: particle has no <_mnpbem_layer>; '
                'use structure.type=with_substrate to enable substrate.')

        return layer

    def build_excitation(self,
            layer: Any) -> Tuple[Any, Any, np.ndarray]:
        from mnpbem.geometry import ComPoint
        from mnpbem.simulation import DipoleRetLayer

        pos = self.cfg['simulation'].get('dipole_position', [0.0, 0.0, 25.0])
        mom = self.cfg['simulation'].get('dipole_moment', [0.0, 0.0, 1.0])

        pos = np.asarray(pos, dtype = float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)

        mom = np.asarray(mom, dtype = float)
        if mom.ndim == 1:
            mom = mom.reshape(1, -1)

        pt = ComPoint(self.p, pos)
        dip = DipoleRetLayer(pt, layer, dip = mom)

        return dip, pt, mom

    def build_solver(self,
            layer: Any) -> Any:
        from mnpbem.bem import BEMRetLayer

        return BEMRetLayer(self.p, layer)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        bem = self.build_solver(layer)
        dip, pt, mom = self.build_excitation(layer)

        n_wl = len(enei)
        n_dip = mom.shape[0] if mom.ndim == 2 else 1

        tot = np.zeros((n_wl, n_dip))
        rad = np.zeros((n_wl, n_dip))

        print_info('DipoleRetLayer: warming up at enei={:.1f} nm'.format(float(enei[0])))
        t_warm = time.time()
        exc = dip(self.p, float(enei[0]))
        sig, bem = bem.solve(exc)
        t_val, r_val, _ = dip.decayrate(sig)
        warm_s = time.time() - t_warm

        tot[0, :] = self._flatten_decay(t_val, n_dip)
        rad[0, :] = self._flatten_decay(r_val, n_dip)

        self.save_sigma_for_wavelength(sig, float(enei[0]))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            exc = dip(self.p, float(enei[i]))
            sig, bem = bem.solve(exc)
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

        # Reuse extinction/scattering schema for downstream postprocess:
        # ext = total decay rate, sca = radiative, abs = non-radiative.
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
            'n_pol': n_dip}

    def _flatten_decay(self,
            val: Any,
            n_dip: int) -> np.ndarray:
        v = np.atleast_1d(np.asarray(val)).real.flatten()
        if v.size < n_dip:
            v = np.tile(v, n_dip)[:n_dip]
        return v[:n_dip]
