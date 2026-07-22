import time
import warnings

from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.linalg import LinAlgWarning as _ScipyLinAlgWarning
except Exception:
    _ScipyLinAlgWarning = RuntimeWarning

from .base import SimulationRunner
from ..util import print_info


class PlaneWaveRetLayerRunner(SimulationRunner):
    """Retarded plane-wave excitation on a particle + planar substrate.

    Looks up the LayerStructure attached to the particle by
    WithSubstrateBuilder via ``_mnpbem_layer``. Uses GreenTabLayer to
    pre-tabulate the substrate Green function before the wavelength
    loop, then BEMRetLayer + PlaneWaveRetLayer for the spectrum.

    YAML config (simulation section)::

        simulation:
          type: ret_layer
          excitation: planewave
          polarizations: [[1, 0, 0]]
          propagation_dirs: [[0, 0, -1]]   # -z = downward incidence from above
          enei_min: 450
          enei_max: 750
          n_wavelengths: 11
          tab_n: 5                          # GreenTabLayer enei tabulation count
    """

    def build_layer(self) -> Any:
        layer = getattr(self.p, '_mnpbem_layer', None)

        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)

        if layer is None:
            raise RuntimeError(
                '[error] PlaneWaveRetLayerRunner: particle has no <_mnpbem_layer>; '
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
            'GreenTabLayer: tabulated at {} enei points ({:.1f}-{:.1f} nm)'.format(
                tab_n, float(enei_tab[0]), float(enei_tab[-1])))

        return gt

    def build_excitation(self,
            layer: Any) -> Any:
        from mnpbem.simulation import PlaneWaveRetLayer

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0]])
        prop = self.cfg['simulation'].get('propagation_dirs',
                [[0, 0, -1]])

        pol = np.asarray(pol, dtype = float)
        prop = np.asarray(prop, dtype = float)

        return PlaneWaveRetLayer(pol, prop, layer)

    def build_solver(self,
            layer: Any,
            greentab: Any) -> Any:
        from mnpbem.bem import BEMRetLayer

        return BEMRetLayer(self.p, layer, greentab = greentab)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        layer = self.build_layer()
        exc = self.build_excitation(layer)

        n_wl = len(enei)
        n_pol = self._infer_n_pol(exc)

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        # Spectrum-sweep RESUME: wavelengths already in the sigma cache are
        # reloaded and their extinction/scattering recomputed from the cached
        # BEM solution (cheap) instead of re-solving.  The Green tabulation +
        # solver (the expensive ~warmup) build lazily on the first cache miss,
        # so a fully-cached re-run skips them entirely.  A killed run thus
        # resumes near where it stopped rather than restarting from wl 0.
        output_dir = self._sigma_output_dir()
        cache_load_enabled = self._sigma_cache_load_enabled()
        cache_save_enabled = self._sigma_cache_enabled()
        cache_manifest_ok = self._cache_manifest_compatible() if output_dir else False

        print_info(
            'PlaneWaveRetLayer cache: load={} save={} manifest_ok={} out={}'.format(
                int(cache_load_enabled), int(cache_save_enabled),
                int(cache_manifest_ok), output_dir if output_dir else '<none>'))

        n_cached0 = self.count_cached_wavelengths(enei)
        n_to_solve = n_wl - n_cached0
        if n_cached0 > 0:
            print_info('PlaneWaveRetLayer: resume — {}/{} wl cached, {} to solve'.format(
                n_cached0, n_wl, n_to_solve))

        bem = None
        warm_s = 0.0
        n_solved = 0
        t_loop = time.time()

        for i in range(n_wl):
            wl = float(enei[i])
            sig = self.load_sigma_for_wavelength(wl)

            if sig is None:
                if bem is None:
                    print_info(
                        'PlaneWaveRetLayer: warming up at enei={:.1f} nm'.format(wl))
                    t_warm = time.time()
                    greentab = self.build_greentab(layer, enei)
                    print_info('PlaneWaveRetLayer: building BEMRetLayer solver')
                    bem = self.build_solver(layer, greentab)
                    print_info('PlaneWaveRetLayer: starting first BEM solve at enei={:.1f} nm'.format(wl))
                    t_first_solve = time.time()
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('error', category = _ScipyLinAlgWarning)
                            sig, bem = bem.solve(exc(self.p, wl))
                    except _ScipyLinAlgWarning as e:
                        raise RuntimeError(
                            'BEMRetLayer solve singular/ill-conditioned at {:.2f} nm: {}'.format(
                                wl, e))

                    if not self._sigma_compstruct_finite(sig):
                        raise RuntimeError(
                            'BEMRetLayer solve produced non-finite sigma at {:.2f} nm'.format(wl))
                    print_info('PlaneWaveRetLayer: first BEM solve finished in {:.1f}s'.format(
                        time.time() - t_first_solve))
                    warm_s = time.time() - t_warm
                    print_info('warmup done in {:.1f}s'.format(warm_s))
                    t_loop = time.time()
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('error', category = _ScipyLinAlgWarning)
                            sig, bem = bem.solve(exc(self.p, wl))
                    except _ScipyLinAlgWarning as e:
                        raise RuntimeError(
                            'BEMRetLayer solve singular/ill-conditioned at {:.2f} nm: {}'.format(
                                wl, e))

                    if not self._sigma_compstruct_finite(sig):
                        raise RuntimeError(
                            'BEMRetLayer solve produced non-finite sigma at {:.2f} nm'.format(wl))
                self.save_sigma_for_wavelength(sig, wl)
                n_solved += 1

            ev = self._extract_extinction(exc.extinction(sig), n_pol)
            sv = self._extract_scattering(exc.scattering(sig), n_pol)

            ext[i, :] = ev[:n_pol]
            sca[i, :] = sv[:n_pol]
            abs_[i, :] = ext[i, :] - sca[i, :]

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                elapsed = time.time() - t_loop
                if n_solved > 0:
                    eta = elapsed / n_solved * max(n_to_solve - n_solved, 0)
                else:
                    eta = 0.0
                print_info(
                    '  wl {}/{}  cached={} solved={}  elapsed={:.1f}min  ETA={:.1f}min'.format(
                        i + 1, n_wl, (i + 1) - n_solved, n_solved,
                        elapsed / 60.0, eta / 60.0))

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
            'n_pol': n_pol}

    def _infer_n_pol(self, exc: Any) -> int:
        pol = self.cfg['simulation'].get('polarizations', [[1, 0, 0]])
        return len(pol)

    def _extract_extinction(self,
            ev_raw: Any,
            n_pol: int) -> np.ndarray:
        ev = np.atleast_1d(np.asarray(ev_raw)).real.flatten()
        if ev.size < n_pol:
            ev = np.tile(ev, n_pol)[:n_pol]
        return ev

    def _extract_scattering(self,
            sv_raw: Any,
            n_pol: int) -> np.ndarray:
        if isinstance(sv_raw, tuple):
            sv = sv_raw[0]
        else:
            sv = sv_raw
        sv = np.atleast_1d(np.asarray(sv)).real.flatten()
        if sv.size < n_pol:
            sv = np.tile(sv, n_pol)[:n_pol]
        return sv
