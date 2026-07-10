"""
Config-driven PostprocessManager.

Orchestrates the full postprocessing pipeline using NEW-style analysis-config
dicts and NEW postprocess module functions, matching the structure of the OLD
PostprocessManager.run() workflow.

Ported from OLD postprocess/postprocess.py (PostprocessManager), adapted so
every step delegates to the NEW functional modules (spectrum.py, eigenmode.py,
fano_analysis.py, export.py, etc.) rather than the OLD class hierarchy.

Recognised config keys
----------------------
Required:
    output_dir       : str   root directory for outputs
    simulation_name  : str   sub-directory name

Optional (all default False/None if absent):
    calculate_cross_sections   : bool   run spectral cross-section analysis
    calculate_fields           : bool   run field analysis + integration
    field_region               : dict   passed through to field functions
    run_eigenmode_analysis     : bool   run plasmon eigenmode analysis
    eigenmode_top_k            : int    modes to retain (default 5)
    calculate_fano             : bool   run Fano-fit analysis
    calculate_multipole        : bool   run multipole decomposition
    export_npz                 : bool   save result as .npz
    export_h5                  : bool   save result as .h5
    export_csv                 : bool   save spectrum as .csv
    near_field_depths          : list   interior depths for integration [nm]
    near_field_center_only     : bool   cluster: integrate centre sphere only
    structure                  : str    particle structure type (for geometry)
    verbose                    : bool
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .spectrum import (
    analyze_spectrum,
    find_spectrum_peaks,
    check_unpolarized_conditions,
    calculate_unpolarized_spectrum,
    analyze_spectrum_unpolarized)
from .field_analyzer import (
    hotspot_summary,
    field_statistics,
    calculate_near_field_integration,
    save_near_field_results)
from .geometry_cross_section import GeometryCrossSection
from .export import export_npz, export_h5, export_csv, export_json
from ..util import ensure_dir, print_info


class PostprocessManager:
    """Coordinate all postprocessing tasks driven by an analysis-config dict.

    Parameters
    ----------
    config : dict
        Analysis configuration (see module docstring for recognised keys).
    verbose : bool
        Echo progress to stdout.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        self.config = config
        self.verbose = verbose

        output_dir = config.get('output_dir')
        simulation_name = config.get('simulation_name')
        if output_dir is None:
            raise ValueError("PostprocessManager: config missing 'output_dir'")
        if simulation_name is None:
            raise ValueError("PostprocessManager: config missing 'simulation_name'")

        self.output_dir = os.path.join(str(output_dir), str(simulation_name))
        ensure_dir(self.output_dir)

        self.geometry = GeometryCrossSection(config, verbose=verbose)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete postprocessing workflow.

        Parameters
        ----------
        data : dict
            Simulation result containing at least ``'wavelength'``, ``'ext'``,
            ``'sca'``, ``'abs'`` arrays, and optionally ``'fields'``,
            ``'sigma'``, ``'polarizations'``, ``'particle'``.

        Returns
        -------
        dict
            Aggregated postprocessing outputs keyed by step name.
        """
        cfg = self.config
        verbose = self.verbose
        out_dir = self.output_dir
        outputs: Dict[str, Any] = {}

        # ---- Step 1: spectra ------------------------------------------------
        if cfg.get('calculate_cross_sections', True):
            if verbose:
                print_info('[1] Spectral analysis ...')
            try:
                spectrum_result = analyze_spectrum(data)
                outputs['spectrum'] = spectrum_result

                unpol_cond = check_unpolarized_conditions(
                    data.get('polarizations'), data.get('wavelength'))
                if unpol_cond.get('can_compute', False):
                    unpol_spectrum = calculate_unpolarized_spectrum(data, unpol_cond)
                    outputs['unpolarized_spectrum'] = unpol_spectrum

                if csv := cfg.get('export_csv', False):
                    csv_path = os.path.join(out_dir, 'spectrum.csv')
                    export_csv(data, csv_path)

            except Exception as exc:
                print_info('[!] Spectral analysis failed: {}'.format(exc))
                if verbose:
                    import traceback; traceback.print_exc()

        # ---- Step 2: Fano fit -----------------------------------------------
        if cfg.get('calculate_fano', False):
            if verbose:
                print_info('[2] Fano analysis ...')
            try:
                from .fano_analysis import analyze_fano
                fano_out = analyze_fano(data, cfg)
                outputs['fano'] = fano_out
            except Exception as exc:
                print_info('[!] Fano analysis failed: {}'.format(exc))
                if verbose:
                    import traceback; traceback.print_exc()

        # ---- Step 3: Multipole decomposition ---------------------------------
        if cfg.get('calculate_multipole', False):
            if verbose:
                print_info('[3] Multipole decomposition ...')
            try:
                from .multipole import multipole_decomposition
                mp_out = multipole_decomposition(data.get('sigma'), data.get('particle'))
                outputs['multipole'] = mp_out
            except Exception as exc:
                print_info('[!] Multipole decomposition failed: {}'.format(exc))
                if verbose:
                    import traceback; traceback.print_exc()

        # ---- Step 4: Field analysis + near-field integration -----------------
        if cfg.get('calculate_fields', False):
            fields = data.get('fields', [])
            if fields:
                if verbose:
                    print_info('[4] Field analysis ...')
                try:
                    field_stats_list = []
                    for fd in fields:
                        field_stats_list.append({
                            'wavelength': fd.get('wavelength'),
                            'polarization_idx': fd.get('polarization_idx'),
                            'statistics': field_statistics(fd)
                                if 'e' in fd else {},
                            'hotspot_summary': hotspot_summary(fd)
                                if 'e' in fd else {},
                        })
                    outputs['field_stats'] = field_stats_list
                except Exception as exc:
                    print_info('[!] Field statistics failed: {}'.format(exc))
                    if verbose:
                        import traceback; traceback.print_exc()

                # Near-field integration (all spheres)
                if verbose:
                    print_info('[4.5] Near-field integration ...')
                try:
                    depths = cfg.get('near_field_depths', None)
                    nf_results = calculate_near_field_integration(
                        fields, cfg, geometry=self.geometry,
                        center_only=False,
                        depths=depths,
                        verbose=verbose)
                    if nf_results:
                        outputs['near_field_integration'] = nf_results
                        nf_path = os.path.join(out_dir, 'near_field_integration.txt')
                        save_near_field_results(nf_results, cfg, nf_path,
                                                center_only=False, depths=depths)
                except Exception as exc:
                    print_info('[!] Near-field integration failed: {}'.format(exc))
                    if verbose:
                        import traceback; traceback.print_exc()

                # Near-field integration (centre sphere only — cluster structures)
                structure = cfg.get('structure', '')
                if structure in ('sphere_cluster', 'sphere_cluster_aggregate'):
                    if cfg.get('near_field_center_only', True):
                        if verbose:
                            print_info('[4.6] Near-field integration (centre sphere only) ...')
                        try:
                            center_results = calculate_near_field_integration(
                                fields, cfg, geometry=self.geometry,
                                center_only=True,
                                depths=depths,
                                verbose=verbose)
                            if center_results:
                                outputs['near_field_integration_center'] = center_results
                                cp = os.path.join(out_dir, 'near_field_integration_center.txt')
                                save_near_field_results(center_results, cfg, cp,
                                                        center_only=True, depths=depths)
                        except Exception as exc:
                            print_info('[!] Centre-sphere integration failed: {}'.format(exc))
                            if verbose:
                                import traceback; traceback.print_exc()

        # ---- Step 5: Eigenmode analysis --------------------------------------
        if cfg.get('run_eigenmode_analysis', False):
            if verbose:
                print_info('[5] Eigenmode analysis ...')
            try:
                eigenmode_out = self._run_eigenmode_analysis(data)
                outputs['eigenmode'] = eigenmode_out
            except Exception as exc:
                print_info('[!] Eigenmode analysis failed: {}'.format(exc))
                if verbose:
                    import traceback; traceback.print_exc()

        # ---- Step 6: Export --------------------------------------------------
        if cfg.get('export_npz', False):
            try:
                npz_path = os.path.join(out_dir, 'result.npz')
                export_npz(data, npz_path)
            except Exception as exc:
                print_info('[!] NPZ export failed: {}'.format(exc))

        if cfg.get('export_h5', False):
            try:
                h5_path = os.path.join(out_dir, 'result.h5')
                export_h5(data, h5_path)
            except Exception as exc:
                print_info('[!] HDF5 export failed: {}'.format(exc))

        if cfg.get('export_json', False):
            try:
                json_path = os.path.join(out_dir, 'outputs.json')
                _json_safe = {k: v for k, v in outputs.items()
                              if isinstance(v, (dict, list, str, int, float, bool, type(None)))}
                export_json(_json_safe, json_path)
            except Exception as exc:
                print_info('[!] JSON export failed: {}'.format(exc))

        if verbose:
            print_info('Postprocessing complete. Results in: {}'.format(out_dir))

        return outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_eigenmode_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run plasmon eigenmode analysis via NEW eigenmode / mode_compare modules."""
        from .eigenmode import qs_eigenmodes, svd_decomposition, retarded_eigen_full, project
        from .mode_compare import (mode_similarity_matrix, match_modes,
                                   assign_bright_dark_multipole)
        from .multipole import multipole_decomposition, dipole_quadrupole_ratio

        cfg = self.config
        top_k = int(cfg.get('eigenmode_top_k', 5))
        eigenmode_out: Dict[str, Any] = {}

        p = data.get('particle')
        sigma = data.get('sigma')
        wavelength = data.get('wavelength')

        if p is None:
            return {'error': 'particle object not present in data'}

        # QS modes
        try:
            qs_vals, qs_vecs = qs_eigenmodes(p)
            eigenmode_out['qs'] = {'eigenvalues': qs_vals, 'eigenvectors': qs_vecs}
        except Exception as exc:
            eigenmode_out['qs_error'] = str(exc)

        # SVD
        if sigma is not None:
            try:
                sig_mat = np.asarray(sigma)
                u_mat, s_vec, vt_mat = svd_decomposition(sig_mat)
                eigenmode_out['svd'] = {'U': u_mat, 's': s_vec, 'Vt': vt_mat}
            except Exception as exc:
                eigenmode_out['svd_error'] = str(exc)

        # Retarded modes
        if wavelength is not None:
            try:
                wl_arr = np.asarray(wavelength)
                wl_center = float(wl_arr[len(wl_arr) // 2])
                ret_out = retarded_eigen_full(p, wl_center)
                eigenmode_out['retarded'] = ret_out
            except Exception as exc:
                eigenmode_out['retarded_error'] = str(exc)

        return eigenmode_out
