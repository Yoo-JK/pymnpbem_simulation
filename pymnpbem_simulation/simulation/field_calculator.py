import os
import sys

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from box import Box

from .base import SimulationRunner
from . import grid_builder
from ..util import print_info


class FieldCalculator(SimulationRunner):

    def __init__(self,
            cfg: Dict[str, Any],
            p: Any,
            epstab: Any) -> None:
        super().__init__(cfg, p, epstab)

        sim_cfg = cfg['simulation']
        self.grid_cfg = sim_cfg.get('grid', dict())
        self.mindist = float(sim_cfg.get('mindist', 1.0))
        nmax_raw = sim_cfg.get('nmax', None)
        self.nmax = int(nmax_raw) if nmax_raw is not None else None
        self.inout = int(sim_cfg.get('inout', 2))
        self.fmm = bool(sim_cfg.get('fmm', False))
        self.fmm_eps = float(sim_cfg.get('fmm_eps', 1e-12))

        self.grid_x, self.grid_y, self.grid_z, self.grid_points = self._build_grid()

    def _build_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gtype = self.grid_cfg.get('type', 'rectangular').lower()

        match gtype:

            case 'rectangular':

                x_range = self.grid_cfg.get('x_range', [-50.0, 50.0])
                y_range = self.grid_cfg.get('y_range', [-50.0, 50.0])
                z_range = self.grid_cfg.get('z_range', [0.0, 0.0])
                n_points = self.grid_cfg.get('n_points', [21, 21, 1])
                x, y, z, pts = grid_builder.make_rectangular_grid(
                        x_range, y_range, z_range, n_points)

            case 'spherical':

                r_range = self.grid_cfg.get('r_range', [10.0, 50.0])
                theta_range = self.grid_cfg.get('theta_range', [0.0, np.pi])
                phi_range = self.grid_cfg.get('phi_range', [0.0, 2 * np.pi])
                n_points = self.grid_cfg.get('n_points', [10, 10, 10])
                x, y, z, pts = grid_builder.make_spherical_grid(
                        r_range, theta_range, phi_range, n_points)

            case 'custom_points':

                points = self.grid_cfg.get('points', None)
                assert points is not None, '[error] Missing <grid.points> for custom_points type!'
                x, y, z, pts = grid_builder.make_custom_points(points)

            case _:

                raise ValueError('[error] Invalid <grid.type> = <{}>!'.format(gtype))

        print_info('FieldCalculator: grid type=<{}> n_points={}'.format(gtype, pts.shape[0]))

        return x, y, z, pts

    def build_excitation(self) -> Any:
        from mnpbem.simulation import PlaneWaveRet, PlaneWaveStat

        sim_cfg = self.cfg['simulation']
        exc_type = sim_cfg.get('excitation', 'planewave')
        sim_type = sim_cfg.get('type', 'ret')

        pol = sim_cfg.get('polarizations', [[1, 0, 0]])
        prop = sim_cfg.get('propagation_dirs', [[0, 0, 1]] * len(pol))

        match (sim_type, exc_type):

            case ('ret', 'planewave') | ('ret', 'planewave_ret'):

                return PlaneWaveRet(pol, prop)

            case ('stat', 'planewave') | ('stat', 'planewave_stat'):

                return PlaneWaveStat(pol)

            case _:

                raise ValueError('[error] Unsupported (sim_type, excitation) = (<{}>, <{}>)!'.format(
                        sim_type, exc_type))

    def build_solver(self) -> Any:
        from mnpbem.bem import BEMRet, BEMStat

        sim_type = self.cfg['simulation'].get('type', 'ret')

        match sim_type:

            case 'ret':

                return BEMRet(self.p)

            case 'stat':

                return BEMStat(self.p)

            case _:

                raise ValueError('[error] Invalid <simulation.type> = <{}>!'.format(sim_type))

    def _make_meshfield(self) -> Any:
        from mnpbem.simulation import MeshField

        sim_type = self.cfg['simulation'].get('type', 'ret')

        return MeshField(
                self.p,
                self.grid_x,
                self.grid_y,
                self.grid_z,
                nmax = self.nmax,
                mindist = self.mindist,
                sim = sim_type)

    def evaluate(self,
            sig: Any) -> Box:

        mf = self._make_meshfield()
        e, h = mf(sig, inout = self.inout, fmm = self.fmm, fmm_eps = self.fmm_eps)

        e_flat = self._flatten_field(e)
        h_flat = self._flatten_field(h) if h is not None else None

        return Box({
                'e': e_flat,
                'h': h_flat,
                'pos': self.grid_points,
                'grid_shape': self.grid_x.shape,
                'inout': self.inout})

    def __call__(self, sig: Any) -> Box:
        return self.evaluate(sig)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        bem = self.build_solver()
        exc = self.build_excitation()

        n_wl = len(enei)
        n_pts = self.grid_points.shape[0]

        n_pol = len(self.cfg['simulation'].get('polarizations', [[1, 0, 0]]))

        e_all = np.zeros((n_wl, n_pts, 3, n_pol), dtype = np.complex128)
        h_all = None

        first_h_set = False

        for i in range(n_wl):
            sig, bem = bem.solve(exc(self.p, enei[i]))
            field_res = self.evaluate(sig)

            e_arr = self._broadcast_pol(field_res.e, n_pol)
            e_all[i] = e_arr

            if field_res.h is not None:
                h_arr = self._broadcast_pol(field_res.h, n_pol)
                if not first_h_set:
                    h_all = np.zeros((n_wl, n_pts, 3, n_pol), dtype = np.complex128)
                    first_h_set = True
                h_all[i] = h_arr

            if (i + 1) % 5 == 0 or (i + 1) == n_wl:
                print_info('  wl {}/{}: enei={:.2f} nm'.format(i + 1, n_wl, enei[i]))

        out = {
                'wavelength': enei,
                'pos': self.grid_points,
                'e': e_all,
                'h': h_all,
                'grid_shape': self.grid_x.shape,
                'n_pol': n_pol,
                'inout': self.inout}

        return out

    def _flatten_field(self,
            arr: np.ndarray) -> np.ndarray:

        if arr is None:
            return None

        a = np.asarray(arr)
        n_pts = self.grid_points.shape[0]

        if a.ndim == 2 and a.shape == (n_pts, 3):
            return a

        if a.size == n_pts * 3:
            return a.reshape(n_pts, 3)

        return a.reshape(-1, 3)[:n_pts]

    def _broadcast_pol(self,
            arr: np.ndarray,
            n_pol: int) -> np.ndarray:

        a = np.asarray(arr)
        n_pts = self.grid_points.shape[0]

        if a.ndim == 2 and a.shape == (n_pts, 3):
            out = np.zeros((n_pts, 3, n_pol), dtype = a.dtype)
            for j in range(n_pol):
                out[..., j] = a
            return out

        if a.ndim == 3 and a.shape[:2] == (n_pts, 3):
            if a.shape[2] == n_pol:
                return a
            out = np.zeros((n_pts, 3, n_pol), dtype = a.dtype)
            for j in range(n_pol):
                out[..., j] = a[..., min(j, a.shape[2] - 1)]
            return out

        flat = a.reshape(n_pts, 3, -1)
        out = np.zeros((n_pts, 3, n_pol), dtype = flat.dtype)
        for j in range(n_pol):
            out[..., j] = flat[..., min(j, flat.shape[2] - 1)]
        return out
