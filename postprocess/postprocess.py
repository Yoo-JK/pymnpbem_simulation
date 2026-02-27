import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from postprocess.post_utils.data_loader import DataLoader
from postprocess.post_utils.spectrum_analyzer import SpectrumAnalyzer
from postprocess.post_utils.visualizer import Visualizer
from postprocess.post_utils.field_analyzer import FieldAnalyzer
from postprocess.post_utils.field_exporter import FieldExporter
from postprocess.post_utils.data_exporter import DataExporter
from postprocess.post_utils.geometry_cross_section import GeometryCrossSection


class PostprocessManager(object):

    def __init__(self,
            config: Dict[str, Any],
            verbose: bool = False) -> None:

        self.config = config
        self.verbose = verbose

        output_dir = config.get('output_dir')
        simulation_name = config.get('simulation_name')

        if output_dir is None:
            raise ValueError('[error] Config missing required key: <output_dir>')
        if simulation_name is None:
            raise ValueError('[error] Config missing required key: <simulation_name>')

        self.output_dir = os.path.join(output_dir, simulation_name)

        self.data_loader = DataLoader(config, verbose)
        self.analyzer = SpectrumAnalyzer(config, verbose)
        self.visualizer = Visualizer(config, verbose)
        self.field_analyzer = FieldAnalyzer(verbose)
        self.field_exporter = FieldExporter(self.output_dir, verbose)
        self.data_exporter = DataExporter(config, verbose)
        self.geometry = GeometryCrossSection(config, verbose)

    def run(self) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:

        if self.verbose:
            print('\n' + '=' * 60)
            print('Starting Postprocessing')
            print('=' * 60)

        # Step 1: Load data
        if self.verbose:
            print('\n[1/6] Loading simulation results...')

        try:
            data = self.data_loader.load_simulation_results()
        except FileNotFoundError:
            print('  [info] Data file not found, trying text file...')
            data = self.data_loader.load_text_results()

        # Step 2: Analyze spectra
        if self.verbose:
            print('\n[2/6] Analyzing spectra...')

        analysis = self.analyzer.analyze(data)

        if self.verbose:
            self._print_analysis_summary(analysis)

        # Step 2.5: Analyze fields
        field_analysis = []
        if 'fields' in data and data['fields']:
            if self.verbose:
                print('\n[2.5/6] Analyzing electromagnetic fields...')

            for field_data in data['fields']:
                field_result = self.field_analyzer.analyze_field(field_data)
                if field_result is not None:
                    field_analysis.append(field_result)

        # Step 3: Create visualizations (with unpolarized plots if applicable)
        if self.verbose:
            print('\n[3/6] Creating visualizations...')

        plots = self.visualizer.create_all_plots(data, analysis)

        if self.verbose and plots:
            print('  Created {} plot(s)'.format(len(plots)))

        # Step 3.5: Export data to TXT files
        if self.verbose:
            print('\n[3.5/6] Exporting data to TXT files...')

        txt_files = self.data_exporter.export_all(data, analysis)

        if self.verbose and txt_files:
            print('  Exported {} TXT file(s)'.format(len(txt_files)))

        # Step 4: Export field data
        if 'fields' in data and data['fields'] and field_analysis:
            if self.verbose:
                print('\n[4/6] Exporting field data...')

            self.field_exporter.export_to_json(data['fields'], field_analysis)

            if self.config.get('export_field_arrays', False):
                self.field_exporter.export_field_data_arrays(data['fields'])

        # Step 4.5: Near-field integration
        near_field_results = None
        if 'fields' in data and data['fields']:
            if self.verbose:
                print('\n[4.5/6] Calculating near-field integration...')

            try:
                near_field_results = self.field_analyzer.calculate_near_field_integration(
                    data['fields'], self.config, self.geometry
                )

                if near_field_results:
                    output_file = os.path.join(self.output_dir, 'near_field_integration.txt')
                    self.field_analyzer.save_near_field_results(
                        near_field_results, self.config, output_file
                    )

                    if self.verbose:
                        print('  [info] Near-field integration completed')
                else:
                    if self.verbose:
                        print('  [info] Structure not supported for near-field integration')
            except Exception as e:
                print('  [error] Near-field integration failed: {}'.format(e))
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        # Step 4.6: Near-field integration (center sphere only)
        if 'fields' in data and data['fields']:
            structure_type = self.config.get('structure', 'unknown')
            if structure_type in ['sphere_cluster_aggregate', 'sphere_cluster']:
                if self.verbose:
                    print('\n[4.6/6] Calculating near-field integration (center sphere only)...')

                try:
                    center_results = self.field_analyzer.calculate_near_field_integration(
                        data['fields'], self.config, self.geometry, center_only = True
                    )

                    if center_results:
                        output_file = os.path.join(self.output_dir, 'near_field_integration_center.txt')
                        self.field_analyzer.save_near_field_results(
                            center_results, self.config, output_file, center_only = True
                        )

                        if self.verbose:
                            print('  [info] Near-field integration (center sphere only) completed')
                except Exception as e:
                    print('  [error] Near-field integration (center sphere) failed: {}'.format(e))
                    if self.verbose:
                        import traceback
                        traceback.print_exc()

        # Step 5: Save processed data
        if self.verbose:
            print('\n[5/6] Saving processed data...')

        self._save_processed_data(data, analysis, field_analysis)

        if self.verbose:
            print('\n' + '=' * 60)
            print('Postprocessing Completed Successfully')
            print('=' * 60)
            print('\nResults saved in: {}/'.format(self.output_dir))

        return data, analysis, field_analysis

    def _print_analysis_summary(self,
            analysis: Dict[str, Any]) -> None:

        print('\n  Spectral Analysis Summary:')
        print('  ' + '-' * 50)

        n_pol = len(analysis['peak_wavelengths'])

        for ipol in range(n_pol):
            print('\n  Polarization {}:'.format(ipol + 1))
            print('    Peak Wavelength: {:.2f} nm'.format(analysis['peak_wavelengths'][ipol]))
            print('    Peak Value: {:.2e} nm^2'.format(analysis['peak_values'][ipol]))
            print('    FWHM: {:.2f} nm'.format(analysis['fwhm'][ipol]))

        unpol_info = analysis.get('unpolarized', {})
        if unpol_info.get('can_calculate', False):
            unpol_spec = analysis.get('unpolarized_spectrum', {})
            print('\n  Unpolarized (FDTD-style incoherent average):')
            print('    Method: {}'.format(unpol_info.get('method', 'N/A')))
            print('    Peak Wavelength: {:.2f} nm'.format(unpol_spec.get('peak_wavelength', 0)))
            print('    Peak Absorption: {:.2e} nm^2'.format(unpol_spec.get('peak_absorption', 0)))
            print('    Peak Extinction: {:.2e} nm^2'.format(unpol_spec.get('peak_extinction', 0)))
        else:
            print('\n  Unpolarized: Not calculated')
            print('    Reason: {}'.format(unpol_info.get('reason', 'Unknown')))

    def _save_processed_data(self,
            data: Dict[str, Any],
            analysis: Dict[str, Any],
            field_analysis: List[Dict[str, Any]]) -> None:

        output_formats = self.config.get('output_formats', ['txt', 'csv', 'json'])

        if 'txt' in output_formats:
            self._save_txt(data, analysis, field_analysis)

        if 'csv' in output_formats:
            self._save_csv(data, analysis)

        if 'json' in output_formats:
            self._save_json(data, analysis, field_analysis)

    def _save_txt(self,
            data: Dict[str, Any],
            analysis: Dict[str, Any],
            field_analysis: List[Dict[str, Any]]) -> None:

        filepath = os.path.join(self.output_dir, 'simulation_processed.txt')

        with open(filepath, 'w') as f:
            f.write('MNPBEM Simulation Results - Processed\n')
            f.write('=' * 60 + '\n\n')

            f.write('SPECTRAL ANALYSIS\n')
            f.write('-' * 60 + '\n\n')

            n_pol = len(analysis['peak_wavelengths'])

            for ipol in range(n_pol):
                f.write('Polarization {}:\n'.format(ipol + 1))
                f.write('  Peak Wavelength: {:.2f} nm\n'.format(analysis['peak_wavelengths'][ipol]))
                f.write('  Peak Value: {:.2e} nm^2\n'.format(analysis['peak_values'][ipol]))
                f.write('  FWHM: {:.2f} nm\n'.format(analysis['fwhm'][ipol]))
                f.write('\n')

            if field_analysis:
                f.write('\nFIELD ANALYSIS\n')
                f.write('-' * 60 + '\n\n')

                for pol_idx, field_result in enumerate(field_analysis):
                    if field_result is None:
                        continue

                    wl = field_result.get('wavelength', 0)
                    f.write('Polarization {} (lambda = {:.1f} nm):\n'.format(pol_idx + 1, wl))

                    stats = field_result.get('enhancement_stats', {})
                    if stats:
                        f.write('  Enhancement Statistics:\n')
                        f.write('    Max:       {:.2f}\n'.format(stats.get('max', 0)))
                        f.write('    Mean:      {:.2f}\n'.format(stats.get('mean', 0)))
                        f.write('    Median:    {:.2f}\n'.format(stats.get('median', 0)))
                        f.write('    95th %ile: {:.2f}\n'.format(stats.get('percentile_95', 0)))

                    hotspots = field_result.get('hotspots', [])
                    if hotspots:
                        f.write('\n  Top Hotspots:\n')
                        for hotspot in hotspots[:5]:
                            pos = hotspot['position']
                            f.write('    #{}: ({:.1f}, {:.1f}, {:.1f}) nm | E/E0 = {:.2f}\n'.format(
                                hotspot['rank'], pos[0], pos[1], pos[2], hotspot['enhancement']))

                    f.write('\n')

            extinction = data.get('extinction')
            has_spectrum = (
                extinction is not None and
                isinstance(extinction, np.ndarray) and
                extinction.size > 0
            )

            if has_spectrum:
                f.write('\nFULL SPECTRUM DATA\n')
                f.write('-' * 60 + '\n\n')
                f.write('Wavelength(nm)\t')

                n_pol = data['extinction'].shape[1] if data['extinction'].ndim == 2 else 1
                for i in range(n_pol):
                    f.write('Ext_pol{}\t'.format(i + 1))
                for i in range(n_pol):
                    f.write('Sca_pol{}\t'.format(i + 1))
                for i in range(n_pol):
                    f.write('Abs_pol{}'.format(i + 1))
                    if i < n_pol - 1:
                        f.write('\t')
                f.write('\n')

                for i, wl in enumerate(data['wavelength']):
                    f.write('{:.2f}\t'.format(wl))
                    for pol in range(n_pol):
                        f.write('{:.6e}\t'.format(data['extinction'][i, pol]))
                    for pol in range(n_pol):
                        f.write('{:.6e}\t'.format(data['scattering'][i, pol]))
                    for pol in range(n_pol):
                        f.write('{:.6e}'.format(data['absorption'][i, pol]))
                        if pol < n_pol - 1:
                            f.write('\t')
                    f.write('\n')
            else:
                f.write('\n[No spectrum data - field-only mode]\n')

        if self.verbose:
            print('  Saved: {}'.format(filepath))

    def _save_csv(self,
            data: Dict[str, Any],
            analysis: Dict[str, Any]) -> None:

        extinction = data.get('extinction')
        has_spectrum = (
            extinction is not None and
            isinstance(extinction, np.ndarray) and
            extinction.size > 0
        )

        if not has_spectrum:
            if self.verbose:
                print('  [info] Skipped CSV - no spectrum data')
            return

        filepath = os.path.join(self.output_dir, 'simulation_processed.csv')

        with open(filepath, 'w', newline = '') as f:
            writer = csv.writer(f)

            n_pol = data['extinction'].shape[1] if data['extinction'].ndim == 2 else 1
            header = ['Wavelength(nm)']
            for i in range(n_pol):
                header.append('Extinction_pol{}'.format(i + 1))
            for i in range(n_pol):
                header.append('Scattering_pol{}'.format(i + 1))
            for i in range(n_pol):
                header.append('Absorption_pol{}'.format(i + 1))

            writer.writerow(header)

            for i, wl in enumerate(data['wavelength']):
                row = [wl]
                for pol in range(n_pol):
                    row.append(data['extinction'][i, pol])
                for pol in range(n_pol):
                    row.append(data['scattering'][i, pol])
                for pol in range(n_pol):
                    row.append(data['absorption'][i, pol])
                writer.writerow(row)

        if self.verbose:
            print('  Saved: {}'.format(filepath))

    def _save_json(self,
            data: Dict[str, Any],
            analysis: Dict[str, Any],
            field_analysis: List[Dict[str, Any]]) -> None:

        filepath = os.path.join(self.output_dir, 'simulation_processed.json')

        def safe_tolist(arr: Any) -> Any:
            if arr is None:
                return []
            if isinstance(arr, np.ndarray):
                if arr.size == 0:
                    return []
                return arr.tolist()
            return arr

        json_data = {
            'wavelength': safe_tolist(data.get('wavelength')),
            'extinction': safe_tolist(data.get('extinction')),
            'scattering': safe_tolist(data.get('scattering')),
            'absorption': safe_tolist(data.get('absorption')),
            'analysis': {}
        }

        for key, value in analysis.items():
            if hasattr(value, 'tolist'):
                json_data['analysis'][key] = value.tolist()
            elif isinstance(value, dict):
                json_data['analysis'][key] = {
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in value.items()
                }
            else:
                json_data['analysis'][key] = value

        if 'fields' in data and data['fields']:
            json_data['field_data_available'] = True
            json_data['field_wavelengths'] = [float(f['wavelength']) for f in data['fields']]

            if field_analysis:
                json_data['field_analysis_summary'] = []
                for field_result in field_analysis:
                    if field_result is None:
                        continue

                    stats = field_result.get('enhancement_stats', {})
                    hotspots = field_result.get('hotspots', [])
                    summary = {
                        'wavelength': field_result.get('wavelength', 0),
                        'max_enhancement': stats.get('max', 0),
                        'mean_enhancement': stats.get('mean', 0),
                        'num_hotspots': len(hotspots),
                        'top_hotspot_position': hotspots[0]['position'] if hotspots else None,
                        'top_hotspot_enhancement': hotspots[0]['enhancement'] if hotspots else None
                    }
                    json_data['field_analysis_summary'].append(summary)
        else:
            json_data['field_data_available'] = False

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent = 2)

        if self.verbose:
            print('  Saved: {}'.format(filepath))
