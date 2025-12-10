"""
Postprocess Manager for pyMNPBEM simulations.

Orchestrates the complete postprocessing pipeline:
1. Load simulation results
2. Analyze spectra
3. Analyze fields
4. Create visualizations
5. Export data
"""

import os
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from .post_utils import (
    DataLoader,
    SpectrumAnalyzer,
    FieldAnalyzer,
    Visualizer,
    SurfaceChargeVisualizer,
    GeometryCrossSection,
    DataExporter
)


class PostprocessManager:
    """
    Manages the complete postprocessing workflow for pyMNPBEM simulations.

    Provides a unified interface to:
    - Load and validate simulation results
    - Perform spectral analysis (peaks, FWHM, etc.)
    - Analyze field distributions (hotspots, statistics)
    - Generate publication-quality visualizations
    - Export data to various formats
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the postprocess manager.

        Args:
            config: Merged configuration dictionary (structure + simulation settings)
            verbose: Whether to print progress information
        """
        self.config = config
        self.verbose = verbose

        # Determine run folder from config
        output_dir = config.get('output_dir', './results')
        run_name = config.get('run_name')
        if not run_name:
            structure = config.get('structure', 'particle')
            sim_type = config.get('simulation_type', 'stat')
            run_name = f"{structure}_{sim_type}"

        self.run_folder = os.path.join(output_dir, run_name)
        self.plots_dir = os.path.join(self.run_folder, 'plots')
        self.data_dir = os.path.join(self.run_folder, 'data')

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize data loader
        self.loader = DataLoader(self.run_folder)

        # Use provided config, structure sim_config for compatibility
        self.sim_config = config
        self.structure_config = config

        # Initialize visualization tools
        self.visualizer = Visualizer(self.sim_config)
        self.surface_charge_viz = SurfaceChargeVisualizer(self.sim_config)
        self.geometry_cross = GeometryCrossSection(self.structure_config)

        # Initialize data exporter
        self.exporter = DataExporter(self.data_dir, self.sim_config)

        # Storage for results
        self.data = {}
        self.analysis = {}

    def run(self) -> Tuple[Dict, Dict, List]:
        """
        Run the complete postprocessing pipeline.

        Returns:
            Tuple of (data, analysis, field_analysis):
                - data: Dictionary with loaded simulation data
                - analysis: Dictionary with analysis results
                - field_analysis: List of field analysis results (or empty list)
        """
        verbose = self.verbose
        if verbose:
            print("=" * 60)
            print("Starting Postprocessing")
            print("=" * 60)

        # Step 1: Load data
        if verbose:
            print("\n[1/5] Loading simulation results...")

        self.data = self.loader.load_all()

        if verbose:
            self._print_data_summary()

        # Step 2: Analyze spectra
        if verbose:
            print("\n[2/5] Analyzing spectra...")

        if 'spectrum' in self.data:
            self.analysis['spectrum'] = self._analyze_spectrum()
            self.analysis['spectrum_unpolarized'] = self._compute_unpolarized_spectrum()

            if verbose:
                self._print_spectrum_analysis()

        # Step 3: Analyze fields
        if verbose:
            print("\n[3/5] Analyzing electric fields...")

        if 'field' in self.data:
            self.analysis['field'] = self._analyze_fields()

            if verbose:
                self._print_field_analysis()

        # Step 4: Generate visualizations
        if self.sim_config.get('save_plots', True):
            if verbose:
                print("\n[4/5] Generating visualizations...")

            self._generate_all_plots()

        # Step 5: Export data
        if verbose:
            print("\n[5/5] Exporting data...")

        self._export_all_data()

        if verbose:
            print("\n" + "=" * 60)
            print(f"Postprocessing complete. Results in: {self.run_folder}")
            print("=" * 60)

        # Build field_analysis list for compatibility with original interface
        field_analysis = []
        if 'field' in self.analysis:
            for pol_idx in sorted(self.analysis['field'].keys()):
                field_analysis.append(self.analysis['field'][pol_idx])

        return self.data, self.analysis, field_analysis

    def _analyze_spectrum(self) -> Dict[str, Any]:
        """Analyze optical spectra."""
        analyzer = SpectrumAnalyzer(self.data['spectrum'])

        analysis = {
            'resonance_summary': analyzer.get_resonance_summary(),
            'peaks': {},
            'fwhm': {},
            'statistics': {},
        }

        for pol_idx in range(analyzer.n_polarizations):
            # Find peaks
            peaks = analyzer.find_peaks('extinction', pol_idx)
            analysis['peaks'][pol_idx] = peaks

            # Calculate FWHM for main peak
            if peaks:
                fwhm = analyzer.calculate_fwhm('extinction', pol_idx, 0)
                analysis['fwhm'][pol_idx] = fwhm

            # Statistics
            stats = analyzer.get_statistics('extinction', pol_idx)
            analysis['statistics'][pol_idx] = stats

        return analysis

    def _compute_unpolarized_spectrum(self) -> Dict[str, np.ndarray]:
        """Compute unpolarized spectrum."""
        analyzer = SpectrumAnalyzer(self.data['spectrum'])
        return analyzer.calculate_unpolarized()

    def _analyze_fields(self) -> Dict[int, Dict[str, Any]]:
        """Analyze electric field distributions."""
        field_analysis = {}

        for pol_idx, field_data in self.data['field'].items():
            analyzer = FieldAnalyzer(field_data)

            field_analysis[pol_idx] = {
                'statistics': analyzer.get_statistics(),
                'hotspots': analyzer.find_hotspots(),
                'enhanced_volume': analyzer.compute_enhanced_volume(),
            }

        self.analysis['field_stats'] = {k: v['statistics'] for k, v in field_analysis.items()}
        self.analysis['hotspots'] = {k: v['hotspots'] for k, v in field_analysis.items()}

        return field_analysis

    def _generate_all_plots(self):
        """Generate all visualization plots."""
        # Spectrum plots
        if 'spectrum' in self.data:
            self._generate_spectrum_plots()

        # Field plots
        if 'field' in self.data:
            self._generate_field_plots()

        # Surface charge plots
        if 'surface_charges' in self.data:
            self._generate_surface_charge_plots()

    def _generate_spectrum_plots(self):
        """Generate spectrum plots."""
        spectrum = self.data['spectrum']
        n_pol = spectrum['extinction'].shape[1]

        # Individual polarization plots
        for pol_idx in range(n_pol):
            self.visualizer.plot_spectrum(
                spectrum, pol_idx,
                title=f'Optical Spectrum - Polarization {pol_idx + 1}',
                save_path=os.path.join(self.plots_dir, f'spectrum_pol{pol_idx + 1}')
            )

        # Comparison plot
        if n_pol > 1:
            self.visualizer.plot_spectrum_comparison(
                spectrum,
                title='Extinction Spectrum Comparison',
                save_path=os.path.join(self.plots_dir, 'spectrum_comparison')
            )

            # Unpolarized plot
            if 'spectrum_unpolarized' in self.analysis:
                self.visualizer.plot_spectrum_unpolarized(
                    spectrum, self.analysis['spectrum_unpolarized'],
                    title='Unpolarized vs Polarized Spectra',
                    save_path=os.path.join(self.plots_dir, 'spectrum_unpolarized')
                )

        self.visualizer.close_all()

    def _generate_field_plots(self):
        """Generate field enhancement plots."""
        field_data = self.data['field']

        # Get geometry overlay
        geometry_overlay = self.geometry_cross.calculate('xz', 0.0)

        for pol_idx, data in field_data.items():
            # Basic field plot
            self.visualizer.plot_field_enhancement(
                data,
                geometry_overlay=geometry_overlay,
                title=f'Field Enhancement - Polarization {pol_idx + 1}',
                save_path=os.path.join(self.plots_dir, f'field_pol{pol_idx + 1}')
            )

            # Linear scale version
            self.visualizer.plot_field_enhancement(
                data,
                log_scale=False,
                geometry_overlay=geometry_overlay,
                title=f'Field Enhancement (Linear) - Pol {pol_idx + 1}',
                save_path=os.path.join(self.plots_dir, f'field_pol{pol_idx + 1}_linear')
            )

        # Comparison plot if multiple polarizations
        if len(field_data) > 1:
            self.visualizer.plot_field_comparison(
                field_data,
                title='Field Enhancement Comparison',
                save_path=os.path.join(self.plots_dir, 'field_comparison')
            )

        self.visualizer.close_all()

    def _generate_surface_charge_plots(self):
        """Generate surface charge plots."""
        charge_data = self.data['surface_charges']
        mode_info = self.data.get('summary', {}).get('mode_analysis', {})

        for pol_idx, data in charge_data.items():
            # 3D surface charge plot
            self.surface_charge_viz.plot_surface_charges_3d(
                data,
                component='real',
                title=f'Surface Charge - Polarization {pol_idx + 1}',
                save_path=os.path.join(self.plots_dir, f'surface_charge_pol{pol_idx + 1}')
            )

            # Multi-view plot
            self.surface_charge_viz.plot_charge_multiview(
                data,
                title=f'Surface Charge Multi-View - Pol {pol_idx + 1}',
                save_path=os.path.join(self.plots_dir, f'surface_charge_multiview_pol{pol_idx + 1}')
            )

            # Mode analysis plot
            if str(pol_idx) in mode_info:
                self.surface_charge_viz.plot_mode_analysis(
                    data, mode_info[str(pol_idx)],
                    save_path=os.path.join(self.plots_dir, f'mode_analysis_pol{pol_idx + 1}')
                )

    def _export_all_data(self):
        """Export all data to files."""
        self.exporter.export_all(self.data, self.analysis)

    def _print_data_summary(self):
        """Print summary of loaded data."""
        print(f"      Run folder: {self.run_folder}")

        if 'spectrum' in self.data:
            n_wl = len(self.data['spectrum']['wavelengths'])
            n_pol = self.data['spectrum']['extinction'].shape[1]
            wl_range = self.loader.get_wavelength_range()
            print(f"      Spectrum: {n_wl} wavelengths, {n_pol} polarization(s)")
            print(f"      Wavelength range: {wl_range[0]:.0f} - {wl_range[1]:.0f} nm")

        if 'field' in self.data:
            print(f"      Field data: {len(self.data['field'])} polarization(s)")

        if 'surface_charges' in self.data:
            print(f"      Surface charge data: {len(self.data['surface_charges'])} polarization(s)")

    def _print_spectrum_analysis(self):
        """Print spectrum analysis results."""
        if 'spectrum' not in self.analysis:
            return

        summary = self.analysis['spectrum'].get('resonance_summary', {})

        for pol_key, pol_data in summary.get('resonances', {}).items():
            print(f"      {pol_key}: {pol_data['n_peaks']} peak(s) found")
            for i, peak in enumerate(pol_data['peaks'][:2]):  # Top 2 peaks
                wl = peak['wavelength']
                ext = peak['extinction']
                fwhm = peak.get('fwhm')
                fwhm_str = f", FWHM={fwhm:.1f}nm" if fwhm else ""
                print(f"        Peak {i+1}: λ={wl:.1f}nm, σ_ext={ext:.1f}nm²{fwhm_str}")

    def _print_field_analysis(self):
        """Print field analysis results."""
        if 'field_stats' not in self.analysis:
            return

        for pol_idx, stats in self.analysis['field_stats'].items():
            max_enh = stats.get('max', 0)
            mean_enh = stats.get('mean', 0)
            print(f"      Pol {pol_idx + 1}: Max enhancement={max_enh:.1f}, Mean={mean_enh:.2f}")

            hotspots = self.analysis.get('hotspots', {}).get(pol_idx, [])
            if hotspots:
                top_hotspot = hotspots[0]
                pos = top_hotspot['position']
                print(f"        Top hotspot at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) nm")

    def get_results(self) -> tuple:
        """Get analysis results."""
        return self.data, self.analysis
