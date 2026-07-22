import os
import sys

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from box import Box

from .base import SimulationRunner
from . import grid_builder
from ..util import print_info


def _field_vram_share_active() -> bool:
    """Return True when the FieldCalculator should distribute its grid
    evaluation across multiple CUDA devices.

    Gated by the same ``MNPBEM_VRAM_SHARE_*`` env vars the BEM dispatch
    uses, plus a dedicated ``MNPBEM_FIELD_VRAM_SHARE`` knob so callers
    can disable field-side distribution without touching the BEM-side
    pool (e.g. to compare baselines).
    """
    enabled = os.environ.get('MNPBEM_VRAM_SHARE', '0').strip()
    if enabled not in ('1', 'true', 'True', 'TRUE'):
        return False

    field_knob = os.environ.get('MNPBEM_FIELD_VRAM_SHARE', '1').strip()
    if field_knob in ('0', 'false', 'False', 'FALSE'):
        return False

    try:
        n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '0'))
    except ValueError:
        return False

    return n_gpus >= 2


def _field_vram_share_gpu_count() -> int:
    try:
        return max(1, int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1')))
    except ValueError:
        return 1


def _select_preserve_groups(pt: Any,
        ind: np.ndarray) -> Any:
    """Variant of ``ComPoint.select(index=ind)`` that retains empty groups
    so the Green-function region-index clamp inside
    ``CompGreenRet.field`` (``p1_region = min(inout-1, n_regions-1)``)
    picks the same medium as the parent ComPoint's single-shot
    evaluation.

    ``ind`` is a global index into ``pt._pc.pos`` (the concat order
    walked by ``pt._mask``). The returned ComPoint has the same
    ``inout``, ``_mask`` and ``_ind`` layout as ``pt`` — only the
    per-group ``Point`` objects are down-selected to the slice.
    """
    import copy

    from mnpbem.geometry.compoint import Point

    obj = copy.deepcopy(pt)

    # Translate global indices to (group_idx, local_idx) pairs using the
    # parent's mask ordering.
    ipt: List[int] = []
    local_idx: List[int] = []
    for i in pt._mask:
        for j in range(pt.p[i].n):
            ipt.append(i)
            local_idx.append(j)
    ipt_arr = np.asarray(ipt, dtype = int)
    local_arr = np.asarray(local_idx, dtype = int)

    idx_arr = np.asarray(ind, dtype = int)
    sel_ipt = ipt_arr[idx_arr]
    sel_local = local_arr[idx_arr]

    new_p: List[Any] = []
    new_ind: List[np.ndarray] = []

    for grp_i in range(len(pt.p)):
        mask = sel_ipt == grp_i
        if np.any(mask):
            sel_l = sel_local[mask]
            sub_pt = pt.p[grp_i].select(index = sel_l)
            new_p.append(sub_pt)
            # Map back to absolute grid indices via pt._ind[grp_i]
            if pt._ind is not None and grp_i < len(pt._ind):
                abs_inds = np.asarray(pt._ind[grp_i], dtype = int)[sel_l]
            else:
                abs_inds = sel_l
            new_ind.append(abs_inds)
        else:
            # Empty group — preserve it so the region/group indexing
            # in the Green function stays aligned with the parent.
            new_p.append(Point(np.zeros((0, 3), dtype = np.float64)))
            new_ind.append(np.array([], dtype = int))

    obj.p = new_p
    obj.inout = np.asarray(pt.inout, dtype = int).copy()
    obj._mask = list(pt._mask)
    obj._ind = new_ind
    obj._npos = pt._npos
    obj._update_pc()
    return obj


def _field_vram_share_device_ids(n_gpus: int) -> List[int]:
    """Read the explicit ``MNPBEM_VRAM_SHARE_DEVICE_IDS`` list when present,
    otherwise return ``[0, 1, ..., n_gpus-1]``.

    The returned ids are interpreted as *local* indices inside the
    process's CUDA_VISIBLE_DEVICES space (matches the convention used
    by the multi_gpu dispatch helpers).
    """
    raw = os.environ.get('MNPBEM_VRAM_SHARE_DEVICE_IDS', '').strip()

    if raw:
        try:
            ids = [int(x) for x in raw.split(',') if x.strip()]

            if ids:
                return ids

        except ValueError:
            pass

    return list(range(n_gpus))


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

        # When the cfg requests multi-GPU pooling for the FieldCalculator
        # but the upstream dispatch path did *not* materialise the
        # MNPBEM_VRAM_SHARE env bridge (the field branch in
        # dispatch_single_node._dispatch_field skips the vram_share env
        # setup that the BEM dispatch path applies), populate the env
        # vars here so ``evaluate()`` activates the grid-split path. The
        # env vars are restored on each evaluate() exit so we don't leak
        # state into BEM solves that follow in the same process.
        self._field_vram_share_env_saved: Optional[Dict[str, str]] = None
        self._maybe_seed_vram_share_env_from_cfg()

        # Reuse MeshField across wavelengths. Rebuilding layer-aware Green
        # objects each wavelength is expensive and can trigger allocator
        # thrash in substrate runs.
        self._meshfield_cache: Optional[Any] = None

    def _maybe_seed_vram_share_env_from_cfg(self) -> None:
        """Cache a deferred env snapshot derived from cfg so
        ``evaluate()`` can flip ``MNPBEM_VRAM_SHARE*`` on for the
        grid-split eval without leaking the flag back into a subsequent
        BEM solve that the same FieldCalculator may trigger.

        Honors ``compute.vram_share`` (preferred) and the
        ``compute.n_gpus_per_worker`` shorthand.
        """
        compute = self.cfg.get('compute', dict())
        n_gpus_per_worker = int(compute.get('n_gpus_per_worker', 1))

        vs_cfg = compute.get('vram_share', None)
        if isinstance(vs_cfg, dict):
            enabled = bool(vs_cfg.get('enabled', n_gpus_per_worker > 1))
            n_gpus = int(vs_cfg.get('n_gpus', n_gpus_per_worker))
            device_ids = vs_cfg.get('device_ids', None)
        elif isinstance(vs_cfg, bool):
            enabled = vs_cfg
            n_gpus = n_gpus_per_worker
            device_ids = None
        else:
            enabled = n_gpus_per_worker > 1
            n_gpus = n_gpus_per_worker
            device_ids = None

        self._field_cfg_enabled = bool(enabled and n_gpus >= 2)
        self._field_cfg_n_gpus = int(n_gpus)
        self._field_cfg_device_ids = list(device_ids) if device_ids else None

    def _push_field_vram_env(self) -> Optional[Dict[str, Optional[str]]]:
        """Set the ``MNPBEM_VRAM_SHARE*`` env vars from cfg when the
        env bridge is not already active and the cfg requests it.

        Returns a snapshot of the previous env state so ``_pop_field_vram_env``
        can restore it; returns None when no env-mutation was needed.
        """
        if not getattr(self, '_field_cfg_enabled', False):
            return None

        if os.environ.get('MNPBEM_VRAM_SHARE', '') in ('1', 'true', 'True', 'TRUE'):
            # Caller already managed the env bridge — let them own it.
            return None

        keys = ('MNPBEM_VRAM_SHARE',
                'MNPBEM_VRAM_SHARE_GPUS',
                'MNPBEM_VRAM_SHARE_DEVICE_IDS')
        snapshot: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}

        os.environ['MNPBEM_VRAM_SHARE'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_GPUS'] = str(int(self._field_cfg_n_gpus))
        if self._field_cfg_device_ids:
            os.environ['MNPBEM_VRAM_SHARE_DEVICE_IDS'] = ','.join(
                    str(int(d)) for d in self._field_cfg_device_ids)
        else:
            os.environ.pop('MNPBEM_VRAM_SHARE_DEVICE_IDS', None)

        return snapshot

    def _pop_field_vram_env(self,
            snapshot: Optional[Dict[str, Optional[str]]]) -> None:
        if snapshot is None:
            return
        for k, v in snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

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
        sim_cfg = self.cfg['simulation']
        exc_type = sim_cfg.get('excitation', 'planewave')
        sim_type = sim_cfg.get('type', 'ret')

        pol = sim_cfg.get('polarizations', [[1, 0, 0]])
        prop = sim_cfg.get('propagation_dirs', [[0, 0, 1]] * len(pol))

        # Strip the _iter suffix — the excitation class is the same regardless
        # of whether the BEM solver runs dense LU or GMRES.
        base_sim_type = sim_type.replace('_iter', '')

        if exc_type != 'planewave':
            raise ValueError(
                    '[error] FieldCalculator fallback supports planewave only '
                    '(got <{}>). Provide a complete sigma cache to skip BEM solve.'.format(
                            exc_type))

        if base_sim_type == 'ret':
            from mnpbem.simulation import PlaneWaveRet
            return PlaneWaveRet(pol, prop)

        if base_sim_type == 'stat':
            from mnpbem.simulation import PlaneWaveStat
            return PlaneWaveStat(pol)

        if base_sim_type == 'ret_layer':
            from mnpbem.simulation import PlaneWaveRetLayer
            return PlaneWaveRetLayer(pol, prop, self._build_layer())

        raise ValueError(
                '[error] Unsupported (sim_type, excitation) = (<{}>, <{}>) '
                'for FieldCalculator BEM fallback'.format(sim_type, exc_type))

    def build_solver(self) -> Any:
        """Construct the BEM solver, forwarding iter / hmatrix /
        preconditioner / schur options from cfg.

        Mirrors the planewave_ret_iter / planewave_ret_layer_iter
        helpers so iter solvers respect the yaml's hmatrix=false (etc.)
        instead of falling back to BEMRetIter defaults that auto-enable
        H-matrix for face>5000 — which OOMs on our 15k-face Au@Ag mesh.
        """
        sim_type = self.cfg['simulation'].get('type', 'ret')

        if sim_type == 'ret':
            from mnpbem.bem import BEMRet
            return BEMRet(self.p)

        if sim_type == 'stat':
            from mnpbem.bem import BEMStat
            return BEMStat(self.p)

        if sim_type == 'ret_iter':
            from mnpbem.bem import BEMRetIter
            opts = self._iter_solver_opts()
            return self._construct_bem(BEMRetIter, self.p, **opts)

        if sim_type == 'stat_iter':
            from mnpbem.bem import BEMStatIter
            opts = self._iter_solver_opts()
            return self._construct_bem(BEMStatIter, self.p, **opts)

        if sim_type == 'ret_layer':
            from mnpbem.bem import BEMRetLayer
            return BEMRetLayer(self.p, self._build_layer())

        if sim_type == 'ret_layer_iter':
            from mnpbem.bem import BEMRetLayerIter
            opts = self._iter_solver_opts()
            return self._construct_bem(BEMRetLayerIter, self.p, **opts)

        raise ValueError(
                '[error] Invalid <simulation.type> = <{}>!'.format(sim_type))

    def _build_layer(self) -> Any:
        """Extract the LayerStructure attached to the particle (mirrors
        planewave_ret_layer.build_layer) for substrate field evaluation.
        """
        layer = getattr(self.p, '_mnpbem_layer', None)
        if layer is None and hasattr(self.p, 'pfull'):
            layer = getattr(self.p.pfull, '_mnpbem_layer', None)
        if layer is None:
            raise RuntimeError(
                    '[error] FieldCalculator: particle has no <_mnpbem_layer>; '
                    'use structure.type=with_substrate to enable substrate.')
        return layer

    def _iter_solver_opts(self) -> Dict[str, Any]:
        """Collect iter solver kwargs (gmres knobs + hmatrix + precond)
        mirroring planewave_ret_iter._iter_options et al.
        """
        from .planewave_ret_iter import (
                _iter_options, _iter_hmatrix_options,
                _iter_preconditioner_options, _iter_schur_options)

        opts = _iter_options(self.cfg)
        bem_opts = self._bem_options()
        bem_opts.pop('refun', None)
        opts.update(bem_opts)
        opts.update(_iter_hmatrix_options(self, self.p, self.cfg))
        opts.update(_iter_preconditioner_options(self, self.cfg))
        schur = _iter_schur_options(self, self.cfg)
        if schur:
            opts.update(schur)
        return opts

    def _layer_field_kwargs(self) -> Dict[str, Any]:
        """For substrate (ret_layer) sims, return the ``layer`` + ``greentab``
        kwargs MeshField needs so its Green function is the layer-aware
        CompGreenRetLayer (direct + reflected) rather than free-space
        CompGreenRet. For non-layer sims return an empty dict.

        The reflected-Green tabulation must cover the *grid* points (not
        just the particle faces, which is what the BEM-solve greentab
        spans), so we build a grid-aware tabspace via
        ``layer.tabspace(p, grid_point)``. Cached per instance.
        """
        sim_type = self.cfg['simulation'].get('type', 'ret').replace('_iter', '')
        if sim_type != 'ret_layer':
            return dict()

        if getattr(self, '_layer_field_cache', None) is not None:
            return self._layer_field_cache

        from mnpbem.greenfun import GreenTabLayer
        from mnpbem.geometry.compoint import ComPoint

        layer = self._build_layer()

        # ComPoint over the flat grid so tabspace sees the grid (r, z) span.
        grid_pt = ComPoint(self.p, self.grid_points,
                mindist = self.mindist, layer = layer)
        tab = layer.tabspace(self.p, grid_pt)
        gt = GreenTabLayer(layer, tab = tab)

        enei = np.atleast_1d(np.asarray(
                getattr(self, '_field_enei', None)
                if getattr(self, '_field_enei', None) is not None
                else self.cfg['simulation'].get('field_wavelengths', [500.0]),
                dtype = float))
        if enei.size == 1:
            e0 = float(enei[0])
            enei_tab = np.array([0.999 * e0, 1.001 * e0])
        else:
            enei_tab = np.linspace(float(enei.min()), float(enei.max()),
                    min(5, enei.size))
        gt.set(enei_tab)

        print_info(
                'FieldCalculator: layer greentab tabulated at {} enei '
                '({:.1f}-{:.1f} nm) over grid span'.format(
                        len(enei_tab), float(enei_tab[0]), float(enei_tab[-1])))

        self._layer_field_cache = dict(layer = layer, greentab = gt)
        return self._layer_field_cache

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
                sim = sim_type,
                **self._layer_field_kwargs())

    def _is_layer_field_sim(self) -> bool:
        sim_type = str(self.cfg.get('simulation', dict()).get('type', 'ret'))
        base = sim_type.replace('_iter', '')
        return base in ('ret_layer', 'stat_layer')

    def _get_meshfield(self) -> Any:
        if self._meshfield_cache is None:
            self._meshfield_cache = self._make_meshfield()
        return self._meshfield_cache

    def _make_meshfield_chunk(self,
            x_chunk: np.ndarray,
            y_chunk: np.ndarray,
            z_chunk: np.ndarray) -> Any:
        """Build a MeshField that owns only the supplied chunk of grid
        points. The chunk arrays are 1-D and identical length, so
        MeshField's ``_expand`` broadcast is a no-op and the resulting
        evaluation runs the same code path as a fully flat custom grid.
        """
        from mnpbem.simulation import MeshField

        sim_type = self.cfg['simulation'].get('type', 'ret')

        return MeshField(
                self.p,
                x_chunk,
                y_chunk,
                z_chunk,
                nmax = self.nmax,
                mindist = self.mindist,
                sim = sim_type,
                **self._layer_field_kwargs())

    def evaluate(self,
            sig: Any) -> Box:

        # Scope the cfg-driven env-var bridge to evaluate() so we don't
        # leak MNPBEM_VRAM_SHARE into a BEM solve triggered by callers
        # that reuse the same Python process.
        env_snapshot = self._push_field_vram_env()

        try:
            if _field_vram_share_active():
                return self._evaluate_distributed(sig)

            mf = self._get_meshfield()
            e, h = mf(sig, inout = self.inout,
                    fmm = self.fmm, fmm_eps = self.fmm_eps)

            e_flat = self._flatten_field(e)
            h_flat = self._flatten_field(h) if h is not None else None

            try:
                e_abs = np.abs(np.asarray(e_flat))
                n_tot = int(e_abs.size)
                n_fin = int(np.isfinite(e_abs).sum())
                if n_tot > 0 and n_fin == 0:
                    print_info(
                            'FieldCalculator: warning — all E values are non-finite '
                            '(inout={}, mindist={:.3f})'.format(self.inout, self.mindist))
                elif n_fin > 0:
                    fin_vals = e_abs[np.isfinite(e_abs)]
                    if fin_vals.size > 0 and float(np.nanmax(fin_vals)) < 1e-300:
                        print_info(
                                'FieldCalculator: warning — E field is near-zero everywhere '
                                '(max|E|<1e-300)')
            except Exception:
                pass

            return Box({
                    'e': e_flat,
                    'h': h_flat,
                    'pos': self.grid_points,
                    'grid_shape': self.grid_x.shape,
                    'inout': self.inout})

        finally:
            self._pop_field_vram_env(env_snapshot)

    def _evaluate_distributed(self,
            sig: Any) -> Box:
        """Multi-GPU grid-split field evaluation.

        Activated when ``MNPBEM_VRAM_SHARE=1`` and
        ``MNPBEM_VRAM_SHARE_GPUS>=2``. Builds the full ``MeshField`` once
        so the ``ComPoint`` retains its medium grouping (required for the
        Green-function region-index clamp inside ``CompGreenRet.field``),
        then splits the full point list into ``n_gpus`` slices and
        evaluates each slice via ``pt.select(index=...)`` -> fresh
        Green-function on the slice. Mirrors the existing single-GPU
        ``_field2`` path (``mnpbem.simulation.meshfield._field2``), so
        results are bit-identical to ``MeshField(nmax=chunk_size)`` on a
        single device — modulo the device dispatch.

        Sigma (surface charges/currents) is small relative to the grid
        side of the matmul (N faces × few-Pol vs M points × N faces), so
        it lives on the host and is shipped per-device by mnpbem's
        existing internal h2d transfer.
        """
        n_gpus = _field_vram_share_gpu_count()
        device_ids = _field_vram_share_device_ids(n_gpus)

        try:
            import cupy as cp
        except Exception:
            cp = None

        n_points = self.grid_points.shape[0]

        # Build the full MeshField once so the ComPoint group structure
        # matches the single-GPU path exactly.
        mf_full = self._get_meshfield()

        # Fast path: FMM (sparse free-space O(N) eval) and the
        # ``sig.val['e']`` shortcut both bypass the Green-function
        # mat-mul that we'd split across devices, so we just run the
        # standard single-device path.
        if self.fmm and mf_full._fmm_eligible(sig, self.inout):
            e, h = mf_full(sig, inout = self.inout,
                    fmm = True, fmm_eps = self.fmm_eps)
            e_flat = self._flatten_field(e)
            h_flat = self._flatten_field(h) if h is not None else None
            return Box({
                    'e': e_flat,
                    'h': h_flat,
                    'pos': self.grid_points,
                    'grid_shape': self.grid_x.shape,
                    'inout': self.inout})

        pt = mf_full.pt
        npts = pt.n
        if npts == 0:
            e_total = np.full((n_points, 3), np.nan, dtype = np.complex128)
            return Box({
                    'e': e_total, 'h': None, 'pos': self.grid_points,
                    'grid_shape': self.grid_x.shape, 'inout': self.inout})

        n_chunks = min(n_gpus, npts)
        chunk_size = (npts + n_chunks - 1) // n_chunks

        print_info(
                'FieldCalculator: distributed eval — n_chunks={}, n_active_pts={}, total_pts={}, device_ids={}'.format(
                        n_chunks, npts, n_points, device_ids))

        sim_type = self.cfg['simulation'].get('type', 'ret')

        # Pre-grab one f_sub to discover the trailing tensor shape (n_pol).
        e_total: Optional[np.ndarray] = None
        h_total: Optional[np.ndarray] = None
        has_h: bool = False

        for ci in range(n_chunks):
            c_start = ci * chunk_size
            c_stop = min(npts, (ci + 1) * chunk_size)
            if c_start >= c_stop:
                continue

            ind = np.arange(c_start, c_stop, dtype = int)
            dev_idx = device_ids[ci % len(device_ids)] if device_ids else ci

            def _eval_chunk() -> Tuple[np.ndarray, Optional[np.ndarray]]:
                # Build a sub-ComPoint that down-selects ``ind`` (in
                # ``pt._pc.pos`` order) but *preserves* the parent's
                # group structure — empty groups stay so the Green
                # function's ``con`` table keeps its row indices aligned
                # with the parent (so ``p1_region`` clamp picks the
                # same medium as the single-shot evaluation).
                pt_sub = _select_preserve_groups(pt, ind)
                g_sub = mf_full._make_green(pt_sub, self.p, sim_type)
                f_sub = g_sub.field(sig, self.inout)
                e_sub = self._to_host(f_sub.e)
                h_sub = None
                if hasattr(f_sub, 'val') and 'h' in f_sub.val:
                    h_sub = self._to_host(f_sub.h)
                return e_sub, h_sub

            if cp is not None:
                local_idx = dev_idx if 0 <= dev_idx < n_gpus else (ci % n_gpus)
                with cp.cuda.Device(local_idx):
                    e_sub_host, h_sub_host = _eval_chunk()
            else:
                e_sub_host, h_sub_host = _eval_chunk()

            # e_sub_host has shape (c_stop - c_start, 3, ...) — the sub
            # ComPoint produces results in the pt._pc.pos order, but we
            # still need to scatter them back into the original grid
            # index space via the parent's pt mapping.
            if e_total is None:
                e_local_shape = e_sub_host.shape
                trailing = e_local_shape[1:] if len(e_local_shape) > 1 else (3,)
                # _all is laid out in pt._pc.pos order (npts rows, then 3,
                # then n_pol if present) so we can apply the parent pt()
                # remap at the end to recover absolute grid indices.
                e_local_all = np.zeros((npts,) + trailing, dtype = e_sub_host.dtype)
                if h_sub_host is not None:
                    has_h = True
                    trailing_h = h_sub_host.shape[1:] if h_sub_host.ndim > 1 else (3,)
                    h_local_all = np.zeros((npts,) + trailing_h, dtype = h_sub_host.dtype)

                # Stash references so the next chunk reuses the buffers.
                self.__e_local_all = e_local_all
                self.__h_local_all = h_local_all if h_sub_host is not None else None
                e_total = e_local_all  # alias; final scatter happens below
                h_total = self.__h_local_all

            e_total[c_start:c_stop] = e_sub_host.reshape(
                    (c_stop - c_start,) + e_total.shape[1:])

            if has_h and h_sub_host is not None and h_total is not None:
                h_total[c_start:c_stop] = h_sub_host.reshape(
                        (c_stop - c_start,) + h_total.shape[1:])

        # Scatter from pt._pc order back to the absolute grid order using
        # the parent ComPoint (NaN-fills any mindist-invalid grid cells).
        e_grid = pt(e_total)
        # Reshape to (n_points, 3, ...) — pt(...) returns (npos,) + trailing
        # which is the absolute-grid order matching self.grid_points.
        e_grid = np.asarray(e_grid).reshape((n_points,) + e_grid.shape[1:])

        h_grid = None
        if has_h and h_total is not None:
            h_grid = pt(h_total)
            h_grid = np.asarray(h_grid).reshape((n_points,) + h_grid.shape[1:])

        # Clean up the temporary attrs we stashed.
        try:
            del self.__e_local_all
        except AttributeError:
            pass
        try:
            del self.__h_local_all
        except AttributeError:
            pass

        # Match the non-distributed code path's final flatten: when the
        # single-shot evaluate() collapses higher-rank tensors via
        # ``_flatten_field`` (which mis-reshapes multi-pol back to (n,3)),
        # mirror that here so downstream consumers see an identical tensor
        # shape. ``run()`` later re-broadcasts to (n_pts, 3, n_pol).
        e_grid = self._flatten_field(e_grid)
        h_grid = self._flatten_field(h_grid) if h_grid is not None else None

        return Box({
                'e': e_grid,
                'h': h_grid,
                'pos': self.grid_points,
                'grid_shape': self.grid_x.shape,
                'inout': self.inout})

    @staticmethod
    def _to_host(arr: Any) -> np.ndarray:
        """Pull a cupy / device array back onto the host as a numpy array.

        ``MeshField.field`` returns numpy when cupy is absent or when the
        Green function falls back to CPU, so we tolerate either form.
        """
        if arr is None:
            return None

        if hasattr(arr, 'get'):
            try:
                return arr.get()
            except Exception:
                pass

        try:
            import cupy as _cp

            if isinstance(arr, _cp.ndarray):
                return _cp.asnumpy(arr)

        except Exception:
            pass

        return np.asarray(arr)

    def __call__(self, sig: Any) -> Box:
        return self.evaluate(sig)

    def run(self,
            enei: np.ndarray) -> Dict[str, Any]:

        # Remember the field wavelengths so a layer-aware greentab (built
        # lazily in _make_meshfield) tabulates the reflected Green function
        # at the wavelengths we actually evaluate.
        self._field_enei = np.atleast_1d(np.asarray(enei, dtype = float))

        n_wl = len(enei)
        n_pts = self.grid_points.shape[0]

        n_pol = len(self.cfg['simulation'].get('polarizations', [[1, 0, 0]]))

        e_all = np.zeros((n_wl, n_pts, 3, n_pol), dtype = np.complex128)
        h_all = None
        first_h_set = False

        # Lazy build — only instantiate the BEM solver/excitation when at
        # least one wavelength misses the sigma cache.
        bem = None
        exc = None

        for i in range(n_wl):
            sig = self._sig_from_cache_or_solve(float(enei[i]), bem, exc)
            # If we had to fall back to a BEM solve, remember the solver
            # so subsequent misses can reuse the init state.
            if isinstance(sig, tuple):
                sig, bem, exc = sig

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

        # Drop heavy caches eagerly at end of run.
        self._meshfield_cache = None

        return out

    def _cache_manifest_compatible(self) -> bool:
        """Return True when the on-disk sigma manifest matches the current
        cfg's structure/eps hashes (or when no manifest exists yet).

        Cached result: hash check runs once per FieldCalculator instance.
        """
        if hasattr(self, '_cache_compat_cached'):
            return self._cache_compat_cached

        from .. import sigma_cache as _sc

        output_dir = self._sigma_output_dir()
        if not output_dir:
            self._cache_compat_cached = False
            return False

        manifest = _sc.read_manifest(output_dir)
        if manifest is None:
            self._cache_compat_cached = True
            return True

        struct_h = _sc.compute_structure_hash(self.cfg.get('structure', dict()))
        eps_h = _sc.compute_eps_hash(self.cfg.get('materials', dict()))
        compat = (manifest.get('structure_hash') == struct_h
                and manifest.get('eps_hash') == eps_h)
        self._cache_compat_cached = compat
        return compat

    def _sig_from_cache_or_solve(self,
            wavelength_nm: float,
            bem: Any,
            exc: Any) -> Any:
        """Try to load sigma from <output_dir>/sigma/; fall back to BEM
        solve when a file is missing.

        Returns either:
          * a CompStruct (cache hit; bem/exc unchanged)
          * a (CompStruct, bem, exc) tuple (cache miss — bem/exc may have
            been lazily instantiated and should be propagated by caller).
        """
        from .. import sigma_cache as _sc

        output_dir = self._sigma_output_dir()
        sim = self.cfg.get('simulation', dict())
        pol = sim.get('polarizations', [[1, 0, 0]])
        prop = sim.get('propagation_dirs', [[0, 0, 1]] * len(pol))

        # Read and write cache controls are intentionally separate.
        # save_sigma_cache gates writes (handled in save_sigma_for_wavelength),
        # while load_sigma_cache gates reads. This avoids accidental full
        # recomputation when callers disable writes during field follow-up.
        cache_load_enabled = bool(sim.get('load_sigma_cache', True))
        cached = None
        if cache_load_enabled and output_dir and self._cache_manifest_compatible():
            try:
                cached = _sc.load_sigma(output_dir, wavelength_nm, pol, prop)
            except Exception as e:
                print_info('sigma load failed at {:.2f} nm — will BEM solve. ({})'.format(
                        wavelength_nm, e))
                cached = None

        if cached is not None:
            sim_type = str(sim.get('type', 'ret')).lower()
            expected_solver_type = 'quasistatic' if 'stat' in sim_type else 'retarded'
            cached_solver_type = str(cached.get('solver_type', ''))
            if cached_solver_type != expected_solver_type:
                print_info(
                        'sigma cache at {:.2f} nm has solver_type <{}> but this run expects <{}>. Recomputing.'.format(
                                wavelength_nm, cached_solver_type, expected_solver_type))
                cached = None

        if cached is not None:
            from mnpbem.greenfun import CompStruct
            if cached['solver_type'] == 'retarded':
                required = ('sig1', 'sig2', 'h1', 'h2')
                if not all(k in cached and cached[k] is not None for k in required):
                    print_info(
                            'sigma cache at {:.2f} nm is incomplete for retarded field '
                            '(needs sig1/sig2/h1/h2). Recomputing.'.format(wavelength_nm))
                    cached = None
                else:
                    return CompStruct(self.p, wavelength_nm,
                            sig1 = cached['sig1'], sig2 = cached['sig2'],
                            h1 = cached['h1'], h2 = cached['h2'])
            if cached is not None:
                if 'sig' not in cached or cached['sig'] is None:
                    print_info(
                            'sigma cache at {:.2f} nm is incomplete (missing sig). Recomputing.'.format(
                                    wavelength_nm))
                    cached = None
                else:
                    return CompStruct(self.p, wavelength_nm, sig = cached['sig'])

        # Cache miss: lazy-build solver and run BEM solve. Save sigma
        # afterwards so subsequent field passes can reuse it.
        if bem is None:
            bem = self.build_solver()
        if exc is None:
            exc = self.build_excitation()

        sig, bem = bem.solve(exc(self.p, wavelength_nm))
        try:
            self.save_sigma_for_wavelength(sig, wavelength_nm)
        except Exception as e:
            print_info('sigma save failed at {:.2f} nm (continuing). ({})'.format(
                    wavelength_nm, e))
        return (sig, bem, exc)

    def _flatten_field(self,
            arr: np.ndarray) -> np.ndarray:

        if arr is None:
            return None

        a = np.asarray(arr)
        n_pts = self.grid_points.shape[0]

        if a.ndim == 1 and a.size == n_pts:
            out = np.zeros((n_pts, 3), dtype = a.dtype)
            out[:, :] = a[:, None]
            return out

        if a.ndim == 2 and a.shape == (n_pts, 3):
            return a

        # Common grid-shaped layout from MeshField._reshape_field:
        # (...grid..., 3[, n_pol]). Flatten leading grid dims to n_pts.
        if a.ndim >= 2 and a.shape[-1] == 3:
            return a.reshape(-1, 3)[:n_pts]

        if a.ndim >= 3 and a.shape[-2] == 3:
            tail = a.shape[-1]
            return a.reshape(-1, 3, tail)[:n_pts]

        if a.ndim >= 2 and a.shape[0] == n_pts and a.shape[1] == 3:
            if a.ndim == 2:
                return a
            return a.reshape(n_pts, 3, -1)

        if a.ndim == 3:
            pos_axes = [ax for ax, size in enumerate(a.shape) if size == n_pts]
            comp_axes = [ax for ax, size in enumerate(a.shape) if size == 3]

            pos_axis = pos_axes[0] if pos_axes else 0
            comp_axis = next((ax for ax in comp_axes if ax != pos_axis), None)

            if comp_axis is not None:
                pol_axis = next(
                        (ax for ax in range(a.ndim) if ax not in (pos_axis, comp_axis)),
                        None)
                if pol_axis is not None:
                    normalized = np.moveaxis(a, (pos_axis, comp_axis, pol_axis), (0, 1, 2))
                    if normalized.shape[1] == 3:
                        return normalized
                    if normalized.shape[2] == 3:
                        return np.moveaxis(normalized, 2, 1)

        if a.size == n_pts * 3:
            return a.reshape(n_pts, 3)

        if a.size % (n_pts * 3) == 0:
            return a.reshape(n_pts, 3, -1)

        # Scalar-per-point payload (or scalar-per-point-per-pol) can appear
        # on some fallback paths; promote it to a 3-component field with
        # replicated values on x/y/z so component selection in downstream
        # visualization does not collapse to all-zero artifact channels.
        if a.size % n_pts == 0:
            tail = int(a.size // n_pts)
            scalar = a.reshape(n_pts, tail)
            if tail == 1:
                out = np.zeros((n_pts, 3), dtype = a.dtype)
                out[:, :] = scalar[:, 0][:, None]
                return out
            out = np.zeros((n_pts, 3, tail), dtype = a.dtype)
            out[:, :, :] = scalar[:, None, :]
            return out

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
