import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from mnpbem.geometry import (
    ComParticle, ComParticleMirror, ComPoint, Particle,
)
from mnpbem.bem import (
    BEMStat, BEMRet,
    BEMStatMirror, BEMRetMirror,
    BEMStatLayer, BEMRetLayer,
    BEMStatIter, BEMRetIter,
    BEMRetLayerIter,
)
from mnpbem.simulation import (
    PlaneWaveStat, PlaneWaveRet,
    PlaneWaveStatMirror, PlaneWaveRetMirror,
    DipoleStat, DipoleRet,
    DipoleStatMirror, DipoleRetMirror,
    EELSStat, EELSRet,
    PlaneWaveStatLayer, PlaneWaveRetLayer,
    DipoleStatLayer, DipoleRetLayer,
)


# Excitation classes that return (ext, sca) vs (sca, dsca)
# PlaneWaveStat: absorption() -> scalar, scattering() -> scalar, extinction() = sca + abs
# PlaneWaveRet: absorption() = ext - sca, scattering() -> (sca, dsca), extinction() -> scalar

_PEAK_OPTIONS = {'peak', 'peak_ext', 'peak_sca'}


class BEMSolver(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose

        self._validate_field_options()

    def _validate_field_options(self) -> None:
        calculate_cross_sections = self.config.get('calculate_cross_sections', True)
        calculate_fields = self.config.get('calculate_fields', False)
        field_wl_idx = self.config.get('field_wavelength_idx', 'middle')

        if calculate_fields and not calculate_cross_sections:
            if field_wl_idx in _PEAK_OPTIONS:
                raise ValueError(
                    '[error] Invalid configuration: '
                    'calculate_cross_sections=False with field_wavelength_idx={!r}. '
                    'Peak-based wavelength selection requires spectrum calculation. '
                    'Use "middle", integer index, or wavelength list instead.'.format(
                        field_wl_idx))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self,
            particles: List[Particle],
            materials: Dict[str, Any]) -> Dict[str, Any]:

        t_start = time.time()

        # 1. Build comparticle
        comparticle = self._create_comparticle(particles, materials)
        if self.verbose:
            print('[info] ComParticle created: {} boundary elements'.format(
                comparticle.nfaces))

        # 2. Build BEM solver
        bem = self._create_bem_solver(comparticle)
        if self.verbose:
            print('[info] BEM solver created: {}'.format(type(bem).__name__))

        # 3. Build excitations
        excitations = self._create_excitations(comparticle)
        if self.verbose:
            print('[info] Excitations created: {} object(s)'.format(len(excitations)))

        # 4. Wavelength array
        wl_range = self.config['wavelength_range']
        wavelengths = np.linspace(wl_range[0], wl_range[1], wl_range[2])

        # 5. Cross section calculation
        calculate_cross_sections = self.config.get('calculate_cross_sections', True)
        calculate_fields = self.config.get('calculate_fields', False)
        use_parallel = self.config.get('use_parallel', False)
        num_workers = self.config.get('num_workers', 1)

        if calculate_cross_sections:
            if use_parallel and num_workers != 1:
                spectrum = self._run_parallel(
                    bem, excitations, comparticle, wavelengths, num_workers)
            else:
                spectrum = self._run_wavelength_loop(
                    bem, excitations, comparticle, wavelengths)
        else:
            n_pol = len(excitations)
            spectrum = {
                'extinction': np.zeros((n_pol, len(wavelengths))),
                'scattering': np.zeros((n_pol, len(wavelengths))),
                'absorption': np.zeros((n_pol, len(wavelengths))),
            }

        # 6. Field calculation
        fields_data = []
        surface_charges_data = []

        if calculate_fields:
            field_wl_indices = self._determine_field_wavelengths(
                wavelengths, spectrum['extinction'])

            for wl_idx in field_wl_indices:
                enei = wavelengths[wl_idx]
                if self.verbose:
                    print('[info] Calculating fields at lambda = {:.1f} nm (index {})'.format(
                        enei, wl_idx))

                for j, exc in enumerate(excitations):
                    field_result = self._calculate_fields(
                        bem, comparticle, exc, enei, pol_index = j)
                    fields_data.append({
                        'wavelength': enei,
                        'wavelength_idx': wl_idx,
                        'pol_index': j,
                        'field': field_result,
                    })

        t_elapsed = time.time() - t_start

        # 7. Assemble results
        polarizations = self.config.get('polarizations', [[1, 0, 0]])
        propagation_dirs = self.config.get('propagation_dirs', [[0, 0, 1]])

        results = {
            'wavelength': wavelengths,
            'extinction': spectrum['extinction'],
            'scattering': spectrum['scattering'],
            'absorption': spectrum['absorption'],
            'polarizations': polarizations,
            'propagation_dirs': propagation_dirs,
            'fields': fields_data if fields_data else [],
            'surface_charges': surface_charges_data if surface_charges_data else [],
            'calculation_time': t_elapsed,
        }

        if self.verbose:
            print('[info] Simulation completed in {:.2f} seconds ({:.2f} minutes)'.format(
                t_elapsed, t_elapsed / 60.0))

        return results

    # ------------------------------------------------------------------
    # ComParticle creation
    # ------------------------------------------------------------------

    def _create_comparticle(self,
            particles: List[Particle],
            materials: Dict[str, Any]) -> Union[ComParticle, ComParticleMirror]:

        epstab = materials['epstab']
        inout = materials['inout']
        closed_args = materials.get('closed', [])

        use_mirror = self.config.get('use_mirror_symmetry', False)

        if use_mirror:
            if isinstance(use_mirror, str):
                sym = use_mirror
            else:
                sym = 'xy'

            if closed_args:
                cp = ComParticleMirror(epstab, particles, inout, *closed_args, sym = sym)
            else:
                cp = ComParticleMirror(epstab, particles, inout, sym = sym)

            if self.verbose:
                print('[info] ComParticleMirror created with sym = {!r}'.format(sym))
        else:
            if closed_args:
                cp = ComParticle(epstab, particles, inout, *closed_args)
            else:
                cp = ComParticle(epstab, particles, inout)

        return cp

    # ------------------------------------------------------------------
    # BEM solver creation
    # ------------------------------------------------------------------

    def _create_bem_solver(self,
            comparticle: Union[ComParticle, ComParticleMirror]) -> Any:

        sim_type = self.config['simulation_type']
        use_mirror = self.config.get('use_mirror_symmetry', False)
        use_substrate = self.config.get('use_substrate', False)
        use_iterative = self.config.get('use_iterative_solver', False)

        if use_substrate and sim_type != 'ret':
            raise ValueError(
                '[error] Substrate simulations require "ret" simulation type. '
                'Current: {!r}'.format(sim_type))

        # Dispatch to the correct BEM solver class
        if use_iterative:
            if use_substrate:
                bem = BEMRetLayerIter(comparticle)
            elif sim_type == 'stat':
                bem = BEMStatIter(comparticle)
            else:
                bem = BEMRetIter(comparticle)

        elif use_substrate:
            if sim_type == 'stat':
                bem = BEMStatLayer(comparticle)
            else:
                bem = BEMRetLayer(comparticle)

        elif use_mirror:
            if sim_type == 'stat':
                bem = BEMStatMirror(comparticle)
            else:
                bem = BEMRetMirror(comparticle)

        else:
            if sim_type == 'stat':
                bem = BEMStat(comparticle)
            else:
                bem = BEMRet(comparticle)

        return bem

    # ------------------------------------------------------------------
    # Excitation creation
    # ------------------------------------------------------------------

    def _create_excitations(self,
            comparticle: Union[ComParticle, ComParticleMirror]) -> List[Any]:

        excitation_type = self.config['excitation_type']
        use_mirror = self.config.get('use_mirror_symmetry', False)
        use_substrate = self.config.get('use_substrate', False)
        sim_type = self.config['simulation_type']

        if excitation_type == 'eels' and use_mirror:
            raise ValueError(
                '[error] EELS excitation is NOT compatible with mirror symmetry.')

        if excitation_type == 'planewave':
            return self._create_planewave_excitations(
                comparticle, sim_type, use_mirror, use_substrate)

        elif excitation_type == 'dipole':
            return self._create_dipole_excitations(
                comparticle, sim_type, use_mirror, use_substrate)

        elif excitation_type == 'eels':
            return self._create_eels_excitations(comparticle, sim_type)

        else:
            raise ValueError(
                '[error] Unknown excitation type: {!r}'.format(excitation_type))

    def _create_planewave_excitations(self,
            comparticle: Any,
            sim_type: str,
            use_mirror: Any,
            use_substrate: bool) -> List[Any]:

        polarizations = self.config.get('polarizations', [[1, 0, 0]])
        propagation_dirs = self.config.get('propagation_dirs', [[0, 0, 1]])

        pol_array = np.array(polarizations, dtype = float)
        dir_array = np.array(propagation_dirs, dtype = float)

        # Create single excitation with all polarizations
        # MNPBEM handles multiple polarizations internally
        if use_substrate:
            if sim_type == 'stat':
                exc = PlaneWaveStatLayer(pol_array)
            else:
                exc = PlaneWaveRetLayer(pol_array, dir_array)
        elif use_mirror:
            if sim_type == 'stat':
                exc = PlaneWaveStatMirror(pol_array)
            else:
                exc = PlaneWaveRetMirror(pol_array, dir_array)
        else:
            if sim_type == 'stat':
                exc = PlaneWaveStat(pol_array)
            else:
                exc = PlaneWaveRet(pol_array, dir_array)

        # Return one excitation object (it handles all polarizations)
        return [exc]

    def _create_dipole_excitations(self,
            comparticle: Any,
            sim_type: str,
            use_mirror: Any,
            use_substrate: bool) -> List[Any]:

        position = np.array(self.config.get('dipole_position', [0, 0, 15]), dtype = float)
        moment = np.array(self.config.get('dipole_moment', [0, 0, 1]), dtype = float)

        pt = ComPoint(comparticle, position.reshape(1, -1))

        if use_substrate:
            if sim_type == 'stat':
                exc = DipoleStatLayer(pt, moment)
            else:
                exc = DipoleRetLayer(pt, moment)
        elif use_mirror:
            if sim_type == 'stat':
                exc = DipoleStatMirror(pt, moment)
            else:
                exc = DipoleRetMirror(pt, moment)
        else:
            if sim_type == 'stat':
                exc = DipoleStat(pt, moment)
            else:
                exc = DipoleRet(pt, moment)

        return [exc]

    def _create_eels_excitations(self,
            comparticle: Any,
            sim_type: str) -> List[Any]:

        impact = self.config.get('impact_parameter', [10, 0])
        beam_energy = self.config.get('beam_energy', 200e3)
        width = self.config.get('beam_width', 0.2)

        impact_array = np.array(impact, dtype = float).reshape(1, -1)

        if sim_type == 'stat':
            exc = EELSStat(comparticle, impact_array, width = width)
        else:
            exc = EELSRet(comparticle, impact_array, width = width)

        return [exc]

    # ------------------------------------------------------------------
    # Wavelength loop (serial)
    # ------------------------------------------------------------------

    def _run_wavelength_loop(self,
            bem: Any,
            excitations: List[Any],
            comparticle: Any,
            wavelengths: np.ndarray) -> Dict[str, np.ndarray]:

        n_wl = len(wavelengths)
        excitation_type = self.config['excitation_type']
        sim_type = self.config['simulation_type']

        # Determine number of polarizations
        n_pol = self._get_n_pol()

        ext = np.zeros((n_pol, n_wl))
        sca = np.zeros((n_pol, n_wl))
        absorb = np.zeros((n_pol, n_wl))

        exc = excitations[0]  # single excitation object with all pols

        for i, enei in enumerate(wavelengths):
            if self.verbose and i % max(1, n_wl // 20) == 0:
                print('[info] Progress: {}/{} (lambda = {:.1f} nm, {:.1f}%)'.format(
                    i + 1, n_wl, enei, 100.0 * (i + 1) / n_wl))

            try:
                # Initialize BEM at this wavelength
                bem(enei)

                # Compute excitation potential
                pot = exc(comparticle, enei)

                # Solve BEM equations
                sig, _ = bem.solve(pot)

                # Extract cross sections
                ext_val, sca_val, abs_val = self._extract_cross_sections(
                    exc, sig, sim_type, excitation_type)

                ext[:, i] = np.atleast_1d(ext_val)
                sca[:, i] = np.atleast_1d(sca_val)
                absorb[:, i] = np.atleast_1d(abs_val)

            except Exception as e:
                print('[error] Error at wavelength {} ({:.1f} nm): {}'.format(
                    i, enei, e))

        return {
            'extinction': ext,
            'scattering': sca,
            'absorption': absorb,
        }

    def _extract_cross_sections(self,
            exc: Any,
            sig: Any,
            sim_type: str,
            excitation_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if excitation_type == 'eels':
            # EELS returns loss probability, not cross sections
            loss = exc.loss(sig)
            return loss, np.zeros_like(loss), loss

        if sim_type == 'stat':
            # PlaneWaveStat / DipoleStat
            abs_val = exc.absorption(sig)
            sca_val = exc.scattering(sig)
            ext_val = abs_val + sca_val  # or exc.extinction(sig)
        else:
            # PlaneWaveRet / DipoleRet
            ext_val = exc.extinction(sig)
            sca_result = exc.scattering(sig)
            # PlaneWaveRet.scattering returns (sca, dsca) tuple
            if isinstance(sca_result, tuple):
                sca_val = sca_result[0]
            else:
                sca_val = sca_result
            abs_val = ext_val - sca_val

        return ext_val, sca_val, abs_val

    def _get_n_pol(self) -> int:
        excitation_type = self.config['excitation_type']

        if excitation_type == 'planewave':
            polarizations = self.config.get('polarizations', [[1, 0, 0]])
            return len(polarizations)
        elif excitation_type == 'dipole':
            moment = self.config.get('dipole_moment', [0, 0, 1])
            moment_arr = np.array(moment)
            if moment_arr.ndim == 1:
                return 1
            return moment_arr.shape[0]
        elif excitation_type == 'eels':
            return 1
        else:
            return 1

    # ------------------------------------------------------------------
    # Wavelength loop (parallel with chunking)
    # ------------------------------------------------------------------

    def _run_parallel(self,
            bem: Any,
            excitations: List[Any],
            comparticle: Any,
            wavelengths: np.ndarray,
            num_workers: Union[int, str]) -> Dict[str, np.ndarray]:

        if num_workers == 'auto':
            actual_workers = os.cpu_count() or 1
        elif num_workers == 'env':
            env_val = os.environ.get('MNPBEM_NUM_WORKERS',
                      os.environ.get('SLURM_CPUS_PER_TASK', '1'))
            actual_workers = int(env_val)
        elif isinstance(num_workers, int):
            actual_workers = num_workers
        else:
            actual_workers = 1

        actual_workers = max(1, actual_workers)

        if self.verbose:
            print('[info] Parallel execution with {} workers'.format(actual_workers))

        # Chunk wavelengths
        chunk_size = self.config.get('wavelength_chunk_size', None)
        if chunk_size is None:
            chunk_size = max(1, len(wavelengths) // actual_workers)

        n_wl = len(wavelengths)
        n_pol = self._get_n_pol()

        ext = np.zeros((n_pol, n_wl))
        sca = np.zeros((n_pol, n_wl))
        absorb = np.zeros((n_pol, n_wl))

        # For BEM objects that are not picklable, fall back to serial chunked
        # execution. Python MNPBEM Green functions contain large matrix state
        # that is expensive to serialize. Instead, we use a serial chunk loop
        # that periodically clears memory.
        exc = excitations[0]
        excitation_type = self.config['excitation_type']
        sim_type = self.config['simulation_type']

        n_chunks = (n_wl + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            i_start = chunk_idx * chunk_size
            i_end = min(i_start + chunk_size, n_wl)
            wl_chunk = wavelengths[i_start:i_end]

            if self.verbose:
                print('[info] Chunk {}/{}: wavelengths {:.1f} - {:.1f} nm ({} points)'.format(
                    chunk_idx + 1, n_chunks,
                    wl_chunk[0], wl_chunk[-1], len(wl_chunk)))

            for local_i, enei in enumerate(wl_chunk):
                global_i = i_start + local_i
                try:
                    bem(enei)
                    pot = exc(comparticle, enei)
                    sig, _ = bem.solve(pot)

                    ext_val, sca_val, abs_val = self._extract_cross_sections(
                        exc, sig, sim_type, excitation_type)

                    ext[:, global_i] = np.atleast_1d(ext_val)
                    sca[:, global_i] = np.atleast_1d(sca_val)
                    absorb[:, global_i] = np.atleast_1d(abs_val)

                except Exception as e:
                    print('[error] Error at wavelength {} ({:.1f} nm): {}'.format(
                        global_i, enei, e))

        return {
            'extinction': ext,
            'scattering': sca,
            'absorption': absorb,
        }

    # ------------------------------------------------------------------
    # Field calculation
    # ------------------------------------------------------------------

    def _calculate_fields(self,
            bem: Any,
            comparticle: Any,
            exc: Any,
            enei: float,
            pol_index: int = 0) -> Dict[str, Any]:

        field_region = self.config.get('field_region', {})
        x_range = field_region.get('x_range', [-50, 50, 101])
        y_range = field_region.get('y_range', [0, 0, 1])
        z_range = field_region.get('z_range', [-50, 50, 101])
        mindist = self.config.get('field_mindist', 0.5)

        # Build mesh grid
        x = np.linspace(x_range[0], x_range[1], int(x_range[2]))
        y = np.linspace(y_range[0], y_range[1], int(y_range[2]))
        z = np.linspace(z_range[0], z_range[1], int(z_range[2]))

        # Determine which two axes form the 2D grid
        if int(y_range[2]) == 1:
            # xz-plane at y = y_range[0]
            xx, zz = np.meshgrid(x, z)
            yy = np.full_like(xx, y_range[0])
            grid_shape = xx.shape
        elif int(z_range[2]) == 1:
            # xy-plane at z = z_range[0]
            xx, yy = np.meshgrid(x, y)
            zz = np.full_like(xx, z_range[0])
            grid_shape = xx.shape
        elif int(x_range[2]) == 1:
            # yz-plane at x = x_range[0]
            yy, zz = np.meshgrid(y, z)
            xx = np.full_like(yy, x_range[0])
            grid_shape = yy.shape
        else:
            # Full 3D (rare)
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
            grid_shape = xx.shape

        positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Filter points too close to particle surface
        if mindist > 0:
            positions = self._filter_interior_points(
                positions, comparticle, mindist)

        if len(positions) == 0:
            if self.verbose:
                print('[info] No valid field points after filtering')
            return {
                'grid_shape': grid_shape,
                'x': xx,
                'y': yy,
                'z': zz,
                'e_field': np.zeros((*grid_shape, 3), dtype = complex),
                'enhancement': np.zeros(grid_shape),
            }

        # Create ComPoint for field evaluation
        pt = ComPoint(comparticle, positions)

        # Initialize BEM and solve
        bem(enei)
        pot = exc(comparticle, enei)
        sig, _ = bem.solve(pot)

        # Get field at external points
        try:
            field_result = bem.field(sig, inout = 2)

            if hasattr(field_result, 'e'):
                e_field = field_result.e
            elif isinstance(field_result, dict) and 'e' in field_result:
                e_field = field_result['e']
            else:
                e_field = np.zeros((len(positions), 3), dtype = complex)
        except Exception as e:
            print('[error] Field calculation failed: {}'.format(e))
            e_field = np.zeros((len(positions), 3), dtype = complex)

        # For multi-polarization, select the specific polarization
        if e_field.ndim == 3:
            # (n_points, 3, n_pol) -> select pol_index
            if pol_index < e_field.shape[2]:
                e_field = e_field[:, :, pol_index]
            else:
                e_field = e_field[:, :, 0]

        # Compute field enhancement |E|^2 / |E0|^2
        e_norm_sq = np.sum(np.abs(e_field) ** 2, axis = 1)  # (n_points,)
        enhancement = e_norm_sq  # |E|^2 (normalize by |E0|^2 = 1 for plane wave)

        return {
            'grid_shape': grid_shape,
            'x': xx,
            'y': yy,
            'z': zz,
            'positions': positions,
            'e_field': e_field,
            'enhancement': enhancement,
            'wavelength': enei,
            'pol_index': pol_index,
        }

    def _filter_interior_points(self,
            positions: np.ndarray,
            comparticle: Any,
            mindist: float) -> np.ndarray:

        # Simple distance-based filtering from particle surface vertices
        try:
            if hasattr(comparticle, 'pos'):
                particle_pos = comparticle.pos  # (nfaces, 3)
            elif hasattr(comparticle, 'pc') and hasattr(comparticle.pc, 'pos'):
                particle_pos = comparticle.pc.pos
            else:
                return positions

            # Compute minimum distance from each field point to any particle vertex
            # Use chunked computation for memory efficiency
            mask = np.ones(len(positions), dtype = bool)
            chunk = 1000

            for i_start in range(0, len(positions), chunk):
                i_end = min(i_start + chunk, len(positions))
                pos_chunk = positions[i_start:i_end]  # (chunk, 3)

                # (chunk, 1, 3) - (1, n_vert, 3) -> (chunk, n_vert, 3)
                diff = pos_chunk[:, np.newaxis, :] - particle_pos[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis = 2))  # (chunk, n_vert)
                min_dists = np.min(dists, axis = 1)  # (chunk,)
                mask[i_start:i_end] = min_dists >= mindist

            filtered = positions[mask]
            if self.verbose:
                n_removed = len(positions) - len(filtered)
                if n_removed > 0:
                    print('[info] Filtered {} interior points (mindist = {} nm)'.format(
                        n_removed, mindist))

            return filtered

        except Exception:
            # If filtering fails, return all points
            return positions

    # ------------------------------------------------------------------
    # Field wavelength selection
    # ------------------------------------------------------------------

    def _determine_field_wavelengths(self,
            wavelengths: np.ndarray,
            extinction: np.ndarray) -> List[int]:

        field_wl_idx = self.config.get('field_wavelength_idx', 'middle')
        n_wl = len(wavelengths)
        n_pol = extinction.shape[0]

        if field_wl_idx == 'middle':
            return [n_wl // 2]

        elif field_wl_idx == 'peak' or field_wl_idx == 'peak_ext':
            # Find peak extinction for each polarization -> unique indices
            indices = set()
            for j in range(n_pol):
                idx = int(np.argmax(extinction[j, :]))
                indices.add(idx)
                if self.verbose:
                    print('[info] Peak extinction for pol {}: lambda = {:.1f} nm (index {})'.format(
                        j, wavelengths[idx], idx))
            return sorted(indices)

        elif field_wl_idx == 'peak_sca':
            # Need scattering data - find from absorption = ext - sca
            indices = set()
            for j in range(n_pol):
                idx = int(np.argmax(extinction[j, :]))
                indices.add(idx)
            return sorted(indices)

        elif isinstance(field_wl_idx, int):
            idx = min(field_wl_idx, n_wl - 1)
            return [idx]

        elif isinstance(field_wl_idx, list):
            # List of wavelengths in nm -> map to nearest indices
            indices = set()
            for target_wl in field_wl_idx:
                idx = int(np.argmin(np.abs(wavelengths - target_wl)))
                indices.add(idx)
                if self.verbose:
                    print('[info] Target {:.1f} nm -> index {} (actual {:.1f} nm)'.format(
                        target_wl, idx, wavelengths[idx]))
            return sorted(indices)

        else:
            return [n_wl // 2]

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self,
            results: Dict[str, Any],
            output_path: str) -> str:

        save_format = self.config.get('save_format', 'npz')

        if save_format == 'npz':
            return self._save_npz(results, output_path)
        elif save_format == 'mat':
            return self._save_mat(results, output_path)
        elif save_format == 'hdf5':
            return self._save_hdf5(results, output_path)
        else:
            return self._save_npz(results, output_path)

    def _save_npz(self,
            results: Dict[str, Any],
            output_path: str) -> str:

        filepath = os.path.join(output_path, 'simulation_results.npz')

        save_dict = {
            'wavelength': results['wavelength'],
            'extinction': results['extinction'],
            'scattering': results['scattering'],
            'absorption': results['absorption'],
            'polarizations': np.array(results['polarizations']),
            'propagation_dirs': np.array(results['propagation_dirs']),
            'calculation_time': np.array(results['calculation_time']),
        }

        np.savez(filepath, **save_dict)

        if self.verbose:
            print('[info] Results saved to: {}'.format(filepath))

        # Save cross sections to text file
        self._save_txt(results, output_path)

        # Save field data separately if present
        if results.get('fields'):
            self._save_fields_npz(results['fields'], output_path)

        return filepath

    def _save_mat(self,
            results: Dict[str, Any],
            output_path: str) -> str:

        try:
            from scipy.io import savemat

            filepath = os.path.join(output_path, 'simulation_results.mat')
            save_dict = {
                'wavelength': results['wavelength'],
                'extinction': results['extinction'],
                'scattering': results['scattering'],
                'absorption': results['absorption'],
                'polarizations': np.array(results['polarizations']),
                'propagation_dirs': np.array(results['propagation_dirs']),
                'calculation_time': results['calculation_time'],
            }

            savemat(filepath, {'results': save_dict})

            if self.verbose:
                print('[info] Results saved to: {}'.format(filepath))

            return filepath

        except ImportError:
            print('[error] scipy.io not available, falling back to npz')
            return self._save_npz(results, output_path)

    def _save_hdf5(self,
            results: Dict[str, Any],
            output_path: str) -> str:

        try:
            import h5py

            filepath = os.path.join(output_path, 'simulation_results.hdf5')

            with h5py.File(filepath, 'w') as f:
                f.create_dataset('wavelength', data = results['wavelength'])
                f.create_dataset('extinction', data = results['extinction'])
                f.create_dataset('scattering', data = results['scattering'])
                f.create_dataset('absorption', data = results['absorption'])
                f.create_dataset('polarizations', data = np.array(results['polarizations']))
                f.create_dataset('propagation_dirs', data = np.array(results['propagation_dirs']))
                f.attrs['calculation_time'] = results['calculation_time']

            if self.verbose:
                print('[info] Results saved to: {}'.format(filepath))

            return filepath

        except ImportError:
            print('[error] h5py not available, falling back to npz')
            return self._save_npz(results, output_path)

    def _save_txt(self,
            results: Dict[str, Any],
            output_path: str) -> None:

        filepath = os.path.join(output_path, 'simulation_results.txt')

        wavelength = results['wavelength']
        ext = results['extinction']
        sca = results['scattering']
        absorb = results['absorption']
        n_pol = ext.shape[0]

        with open(filepath, 'w') as f:
            # Header
            header_parts = ['Wavelength(nm)']
            for j in range(n_pol):
                header_parts.append('Ext_pol{}'.format(j + 1))
            for j in range(n_pol):
                header_parts.append('Sca_pol{}'.format(j + 1))
            for j in range(n_pol):
                header_parts.append('Abs_pol{}'.format(j + 1))
            f.write('\t'.join(header_parts) + '\n')

            # Data
            for i in range(len(wavelength)):
                parts = ['{:.2f}'.format(wavelength[i])]
                for j in range(n_pol):
                    parts.append('{:.6e}'.format(ext[j, i]))
                for j in range(n_pol):
                    parts.append('{:.6e}'.format(sca[j, i]))
                for j in range(n_pol):
                    parts.append('{:.6e}'.format(absorb[j, i]))
                f.write('\t'.join(parts) + '\n')

        if self.verbose:
            print('[info] Cross sections saved to: {}'.format(filepath))

    def _save_fields_npz(self,
            fields_data: List[Dict[str, Any]],
            output_path: str) -> None:

        filepath = os.path.join(output_path, 'field_data.npz')

        save_dict = {}
        for idx, fd in enumerate(fields_data):
            prefix = 'field_{}'.format(idx)
            save_dict['{}_wavelength'.format(prefix)] = np.array(fd.get('wavelength', 0.0))
            save_dict['{}_pol_index'.format(prefix)] = np.array(fd.get('pol_index', 0))

            field_info = fd.get('field', {})
            if isinstance(field_info, dict):
                for key in ['x', 'y', 'z', 'e_field', 'enhancement', 'positions']:
                    if key in field_info:
                        val = field_info[key]
                        if isinstance(val, np.ndarray):
                            save_dict['{}_{}'.format(prefix, key)] = val
                if 'grid_shape' in field_info:
                    save_dict['{}_grid_shape'.format(prefix)] = np.array(field_info['grid_shape'])

        save_dict['n_fields'] = np.array(len(fields_data))

        np.savez(filepath, **save_dict)

        if self.verbose:
            print('[info] Field data saved to: {}'.format(filepath))
