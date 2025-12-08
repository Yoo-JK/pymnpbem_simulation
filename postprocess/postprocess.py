"""
Postprocessing Manager

Coordinates all postprocessing tasks.
Includes unpolarized light calculation (FDTD-style incoherent averaging).
"""

import os
import sys
import json
import csv
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from postprocess.post_utils.data_loader import DataLoader
from postprocess.post_utils.spectrum_analyzer import SpectrumAnalyzer
from postprocess.post_utils.visualizer import Visualizer
from postprocess.post_utils.field_analyzer import FieldAnalyzer
from postprocess.post_utils.field_exporter import FieldExporter
from postprocess.post_utils.data_exporter import DataExporter


class PostprocessManager:
    """Manages the entire postprocessing workflow."""

    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose

        # Validate required config keys
        output_dir = config.get('output_dir')
        simulation_name = config.get('simulation_name')

        if output_dir is None:
            raise ValueError("Config missing required key: 'output_dir'")
        if simulation_name is None:
            raise ValueError("Config missing required key: 'simulation_name'")

        self.output_dir = os.path.join(output_dir, simulation_name)

        # Initialize components
        self.data_loader = DataLoader(config, verbose)
        self.analyzer = SpectrumAnalyzer(config, verbose)
        self.visualizer = Visualizer(config, verbose)
        self.field_analyzer = FieldAnalyzer(verbose)
        self.field_exporter = FieldExporter(self.output_dir, verbose)
        self.data_exporter = DataExporter(config, verbose)
    
    def run(self):
        """Execute complete postprocessing workflow."""
        if self.verbose:
            print("\n" + "="*60)
            print("Starting Postprocessing")
            print("="*60)
        
        # Step 1: Load data
        if self.verbose:
            print("\n[1/6] Loading simulation results...")
        
        try:
            data = self.data_loader.load_simulation_results()
        except FileNotFoundError:
            print("  MAT file not found, trying text file...")
            data = self.data_loader.load_text_results()
        
        # Step 2: Analyze spectra
        if self.verbose:
            print("\n[2/6] Analyzing spectra...")

        analysis = self.analyzer.analyze(data)

        if self.verbose:
            self._print_analysis_summary(analysis)

        # Step 2.5: Analyze fields
        field_analysis = []
        if 'fields' in data and data['fields']:
            if self.verbose:
                print("\n[2.5/6] Analyzing electromagnetic fields...")

            for field_data in data['fields']:
                field_result = self.field_analyzer.analyze_field(field_data)
                field_analysis.append(field_result)

        # Step 3: Create visualizations (with unpolarized plots if applicable)
        if self.verbose:
            print("\n[3/6] Creating visualizations...")

        plots = self.visualizer.create_all_plots(data, analysis)

        if self.verbose and plots:
            print(f"  Created {len(plots)} plot(s)")

        # Step 3.5: Export data to TXT files
        if self.verbose:
            print("\n[3.5/6] Exporting data to TXT files...")

        txt_files = self.data_exporter.export_all(data, analysis)

        if self.verbose and txt_files:
            print(f"  Exported {len(txt_files)} TXT file(s)")
        
        # Step 4: Export field data
        if 'fields' in data and data['fields'] and field_analysis:
            if self.verbose:
                print("\n[4/6] Exporting field data...")

            # Export field analysis to JSON
            self.field_exporter.export_to_json(data['fields'], field_analysis)

            # Optionally export downsampled field arrays
            if self.config.get('export_field_arrays', False):
                self.field_exporter.export_field_data_arrays(data['fields'])

        # Step 5: Save processed data
        if self.verbose:
            print(f"\n[5/6] Saving processed data...")
        
        self._save_processed_data(data, analysis, field_analysis)
        
        if self.verbose:
            print("\n" + "="*60)
            print("Postprocessing Completed Successfully")
            print("="*60)
            print(f"\nResults saved in: {self.output_dir}/")
        
        return data, analysis, field_analysis
    
    def _print_analysis_summary(self, analysis):
        """Print summary of spectral analysis."""
        print("\n  Spectral Analysis Summary:")
        print("  " + "-"*50)

        # Extract per-polarization info from arrays
        n_pol = len(analysis['peak_wavelengths'])

        for ipol in range(n_pol):
            print(f"\n  Polarization {ipol + 1}:")
            print(f"    Peak Wavelength: {analysis['peak_wavelengths'][ipol]:.2f} nm")
            print(f"    Peak Value: {analysis['peak_values'][ipol]:.2e} nm²")
            print(f"    FWHM: {analysis['fwhm'][ipol]:.2f} nm")

        # Print unpolarized analysis if available
        unpol_info = analysis.get('unpolarized', {})
        if unpol_info.get('can_calculate', False):
            unpol_spec = analysis.get('unpolarized_spectrum', {})
            print(f"\n  Unpolarized (FDTD-style incoherent average):")
            print(f"    Method: {unpol_info.get('method', 'N/A')}")
            print(f"    Peak Wavelength: {unpol_spec.get('peak_wavelength', 0):.2f} nm")
            print(f"    Peak Absorption: {unpol_spec.get('peak_absorption', 0):.2e} nm²")
            print(f"    Peak Extinction: {unpol_spec.get('peak_extinction', 0):.2e} nm²")
        else:
            print(f"\n  Unpolarized: Not calculated")
            print(f"    Reason: {unpol_info.get('reason', 'Unknown')}")
    
    def _save_processed_data(self, data, analysis, field_analysis):
        """Save processed data in various formats."""
        output_formats = self.config.get('output_formats', ['txt', 'csv', 'json'])
        
        # Save analysis results
        if 'txt' in output_formats:
            self._save_txt(data, analysis, field_analysis)
        
        if 'csv' in output_formats:
            self._save_csv(data, analysis)
        
        if 'json' in output_formats:
            self._save_json(data, analysis, field_analysis)
    
    def _save_txt(self, data, analysis, field_analysis):
        """Save processed data as text file."""
        filepath = os.path.join(self.output_dir, 'simulation_processed.txt')
        
        with open(filepath, 'w') as f:
            f.write("MNPBEM Simulation Results - Processed\n")
            f.write("="*60 + "\n\n")
            
            # Write analysis summary
            f.write("SPECTRAL ANALYSIS\n")
            f.write("-"*60 + "\n\n")
            
            # ✅ FIX: analysis is a single dict with arrays
            n_pol = len(analysis['peak_wavelengths'])
            
            for ipol in range(n_pol):
                f.write(f"Polarization {ipol + 1}:\n")
                f.write(f"  Peak Wavelength: {analysis['peak_wavelengths'][ipol]:.2f} nm\n")
                f.write(f"  Peak Value: {analysis['peak_values'][ipol]:.2e} nm²\n")
                f.write(f"  FWHM: {analysis['fwhm'][ipol]:.2f} nm\n")
                f.write("\n")
            
            # Write field analysis
            if field_analysis:
                f.write("\nFIELD ANALYSIS\n")
                f.write("-"*60 + "\n\n")
                
                for pol_idx, field_result in enumerate(field_analysis):
                    f.write(f"Polarization {pol_idx + 1} (λ = {field_result['wavelength']:.1f} nm):\n")
                    
                    stats = field_result['enhancement_stats']
                    f.write(f"  Enhancement Statistics:\n")
                    f.write(f"    Max:       {stats['max']:.2f}\n")
                    f.write(f"    Mean:      {stats['mean']:.2f}\n")
                    f.write(f"    Median:    {stats['median']:.2f}\n")
                    f.write(f"    95th %ile: {stats['percentile_95']:.2f}\n")
                    
                    if field_result['hotspots']:
                        f.write(f"\n  Top Hotspots:\n")
                        for hotspot in field_result['hotspots'][:5]:
                            pos = hotspot['position']
                            f.write(f"    #{hotspot['rank']}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) nm "
                                  f"| E/E₀ = {hotspot['enhancement']:.2f}\n")
                    
                    f.write("\n")
            
            # Write full spectrum data
            f.write("\nFULL SPECTRUM DATA\n")
            f.write("-"*60 + "\n\n")
            f.write("Wavelength(nm)\t")
            
            n_pol = data['extinction'].shape[1]
            for i in range(n_pol):
                f.write(f"Ext_pol{i+1}\t")
            for i in range(n_pol):
                f.write(f"Sca_pol{i+1}\t")
            for i in range(n_pol):
                f.write(f"Abs_pol{i+1}")
                if i < n_pol - 1:
                    f.write("\t")
            f.write("\n")
            
            for i, wl in enumerate(data['wavelength']):
                f.write(f"{wl:.2f}\t")
                for pol in range(n_pol):
                    f.write(f"{data['extinction'][i, pol]:.6e}\t")
                for pol in range(n_pol):
                    f.write(f"{data['scattering'][i, pol]:.6e}\t")
                for pol in range(n_pol):
                    f.write(f"{data['absorption'][i, pol]:.6e}")
                    if pol < n_pol - 1:
                        f.write("\t")
                f.write("\n")
        
        if self.verbose:
            print(f"  Saved: {filepath}")
    
    def _save_csv(self, data, analysis):
        """Save processed data as CSV file."""
        filepath = os.path.join(self.output_dir, 'simulation_processed.csv')
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            n_pol = data['extinction'].shape[1]
            header = ['Wavelength(nm)']
            for i in range(n_pol):
                header.append(f'Extinction_pol{i+1}')
            for i in range(n_pol):
                header.append(f'Scattering_pol{i+1}')
            for i in range(n_pol):
                header.append(f'Absorption_pol{i+1}')
            
            writer.writerow(header)
            
            # Data
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
            print(f"  Saved: {filepath}")
    
    def _save_json(self, data, analysis, field_analysis):
        """Save processed data as JSON file."""
        filepath = os.path.join(self.output_dir, 'simulation_processed.json')
        
        # Prepare JSON-serializable data
        json_data = {
            'wavelength': data['wavelength'].tolist(),
            'extinction': data['extinction'].tolist(),
            'scattering': data['scattering'].tolist(),
            'absorption': data['absorption'].tolist(),
            'analysis': {}
        }
        
        # ✅ FIX: Convert analysis dict properly
        for key, value in analysis.items():
            if hasattr(value, 'tolist'):
                json_data['analysis'][key] = value.tolist()
            elif isinstance(value, dict):
                json_data['analysis'][key] = {k: (v.tolist() if hasattr(v, 'tolist') else v) 
                                               for k, v in value.items()}
            else:
                json_data['analysis'][key] = value
        
        # Add field data summary if available
        if 'fields' in data and data['fields']:
            json_data['field_data_available'] = True
            json_data['field_wavelengths'] = [float(f['wavelength']) for f in data['fields']]
            
            # Add field analysis summary
            if field_analysis:
                json_data['field_analysis_summary'] = []
                for field_result in field_analysis:
                    summary = {
                        'wavelength': field_result['wavelength'],
                        'max_enhancement': field_result['enhancement_stats']['max'],
                        'mean_enhancement': field_result['enhancement_stats']['mean'],
                        'num_hotspots': len(field_result['hotspots']),
                        'top_hotspot_position': field_result['hotspots'][0]['position'] if field_result['hotspots'] else None,
                        'top_hotspot_enhancement': field_result['hotspots'][0]['enhancement'] if field_result['hotspots'] else None
                    }
                    json_data['field_analysis_summary'].append(summary)
        else:
            json_data['field_data_available'] = False
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if self.verbose:
            print(f"  Saved: {filepath}")
