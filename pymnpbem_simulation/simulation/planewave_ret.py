import time

from typing import Any, Dict, List, Tuple

import numpy as np

from .base import SimulationRunner
from ..util import print_info


class PlaneWaveRetRunner(SimulationRunner):

    def build_excitation(self) -> Any:
        from mnpbem.simulation import PlaneWaveRet

        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])
        prop = self.cfg['simulation'].get('propagation_dirs',
                [[0, 0, 1], [0, 0, 1]])

        return PlaneWaveRet(pol, prop)

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMRet

        return BEMRet(self.p)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_pol = self._infer_n_pol(exc)

        ext = np.zeros((n_wl, n_pol))
        sca = np.zeros((n_wl, n_pol))
        abs_ = np.zeros((n_wl, n_pol))

        sc_target_wls, sc_tol = self._get_surface_charge_targets(enei)
        sc_records = []

        print_info('PlaneWaveRet: warming up at enei={:.1f} nm'.format(enei[0]))
        t_warm = time.time()
        sig, bem = bem.solve(exc(self.p, enei[0]))
        warm_s = time.time() - t_warm

        ev = np.asarray(exc.extinction(sig)).real.flatten()
        sv = self._extract_scattering(exc.scattering(sig))

        ext[0, :] = ev[:n_pol]
        sca[0, :] = sv[:n_pol]
        abs_[0, :] = ext[0, :] - sca[0, :]

        if self._is_surface_charge_wl(enei[0], sc_target_wls, sc_tol):
            sc_records.append(self._extract_sigma(sig, enei[0], 0, n_pol))

        print_info('warmup done in {:.1f}s'.format(warm_s))

        t_loop = time.time()

        for i in range(1, n_wl):
            sig, bem = bem.solve(exc(self.p, enei[i]))
            ev = np.asarray(exc.extinction(sig)).real.flatten()
            sv = self._extract_scattering(exc.scattering(sig))

            ext[i, :] = ev[:n_pol]
            sca[i, :] = sv[:n_pol]
            abs_[i, :] = ext[i, :] - sca[i, :]

            if self._is_surface_charge_wl(enei[i], sc_target_wls, sc_tol):
                sc_records.append(self._extract_sigma(sig, enei[i], i, n_pol))

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

        result = {
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

        if len(sc_records) > 0:
            sc_dict = self._stack_surface_charge(sc_records, n_pol)
            sc_dict.update(self._extract_mesh_info())
            result['surface_charge'] = sc_dict
            print_info('surface_charge: stored sigma at {} wavelength(s)'.format(
                    len(sc_records)))

        return result

    def _infer_n_pol(self, exc: Any) -> int:
        pol = self.cfg['simulation'].get('polarizations',
                [[1, 0, 0], [0, 1, 0]])

        return len(pol)

    def _extract_scattering(self,
            sv_raw: Any) -> np.ndarray:

        if isinstance(sv_raw, tuple):
            sv = sv_raw[0]
        else:
            sv = sv_raw

        return np.asarray(sv).real.flatten()

    def _get_surface_charge_targets(self,
            enei: np.ndarray) -> Tuple[List[float], float]:

        sim_cfg = self.cfg.get('simulation', dict())

        flag = sim_cfg.get('calculate_surface_charge', None)
        targets = sim_cfg.get('surface_charge_wavelength_idx', None)

        if targets is None:
            targets = sim_cfg.get('field_wavelength_idx', None)

        if flag is None:
            flag = bool(sim_cfg.get('calculate_fields', False)) and targets is not None

        if not flag or targets is None:
            return [], 5.0

        targets = [float(x) for x in targets]
        tol = float(sim_cfg.get('surface_charge_wavelength_tol', 5.0))

        return targets, tol

    def _is_surface_charge_wl(self,
            wl: float,
            targets: List[float],
            tol: float) -> bool:

        if not targets:
            return False

        for t in targets:

            if abs(float(wl) - t) <= tol:
                return True

        return False

    def _extract_sigma(self,
            sig: Any,
            wavelength: float,
            wl_idx: int,
            n_pol: int) -> Dict[str, Any]:

        sig2 = self._to_host_array(sig.sig2 if hasattr(sig, 'sig2') else sig['sig2'])
        sig1 = self._to_host_array(sig.sig1 if hasattr(sig, 'sig1') else sig['sig1'])

        sig2_arr = np.asarray(sig2, dtype = complex)
        sig1_arr = np.asarray(sig1, dtype = complex)

        if sig2_arr.ndim == 1:
            sig2_arr = sig2_arr[:, np.newaxis]

        if sig1_arr.ndim == 1:
            sig1_arr = sig1_arr[:, np.newaxis]

        return {
                'wavelength': float(wavelength),
                'wl_idx': int(wl_idx),
                'sig1': sig1_arr,
                'sig2': sig2_arr}

    def _to_host_array(self, x: Any) -> np.ndarray:

        if hasattr(x, 'get'):

            try:
                return x.get()
            except Exception:
                pass

        try:
            import cupy as cp

            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)

        except ImportError:
            pass

        return np.asarray(x)

    def _stack_surface_charge(self,
            records: List[Dict[str, Any]],
            n_pol: int) -> Dict[str, Any]:

        wls = np.asarray([r['wavelength'] for r in records], dtype = float)
        wl_idxs = np.asarray([r['wl_idx'] for r in records], dtype = int)

        nfaces = records[0]['sig2'].shape[0]

        sig2_all = np.zeros((len(records), nfaces, n_pol), dtype = complex)
        sig1_all = np.zeros((len(records), nfaces, n_pol), dtype = complex)

        for i, r in enumerate(records):
            s2 = r['sig2']
            s1 = r['sig1']
            n_pol_rec = min(n_pol, s2.shape[1])
            sig2_all[i, :, :n_pol_rec] = s2[:, :n_pol_rec]
            sig1_all[i, :, :n_pol_rec] = s1[:, :n_pol_rec]

        return {
                'wavelengths': wls,
                'wl_indices': wl_idxs,
                'sig2': sig2_all,
                'sig1': sig1_all}

    def _extract_mesh_info(self) -> Dict[str, Any]:

        p = self.p

        verts = self._to_host_array(getattr(p, 'verts'))
        faces = self._to_host_array(getattr(p, 'faces'))
        pos = self._to_host_array(getattr(p, 'pos'))
        nvec = self._to_host_array(getattr(p, 'nvec'))
        area = self._to_host_array(getattr(p, 'area'))

        polarizations = np.asarray(self.cfg['simulation'].get(
                'polarizations', [[1, 0, 0], [0, 1, 0]]), dtype = float)

        return {
                'verts': np.asarray(verts, dtype = float),
                'faces': np.asarray(faces, dtype = float),
                'centroids': np.asarray(pos, dtype = float),
                'normals': np.asarray(nvec, dtype = float),
                'areas': np.asarray(area, dtype = float),
                'polarizations': polarizations}
