#!/usr/bin/env python3
"""
pyMNPBEM Postprocessing Runner

Main entry point for analyzing simulation results.
Loads data from a simulation run folder and generates analysis and plots.

Usage:
    python run_postprocess.py /path/to/run_folder
    python run_postprocess.py --latest

"""

import sys
import os
import argparse
from pathlib import Path
import glob


def find_latest_run(base_dir: str) -> str:
    """Find the most recent simulation run folder."""
    pattern = os.path.join(base_dir, '**/config.json')
    config_files = glob.glob(pattern, recursive=True)

    if not config_files:
        raise FileNotFoundError(f"No simulation runs found in {base_dir}")

    # Sort by modification time
    config_files.sort(key=os.path.getmtime, reverse=True)

    # Return the directory containing the most recent config.json
    return os.path.dirname(config_files[0])


def main():
    """Main entry point for postprocessing."""
    parser = argparse.ArgumentParser(
        description='Run pyMNPBEM postprocessing analysis'
    )
    parser.add_argument(
        'run_folder',
        nargs='?',
        help='Path to simulation run folder'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Process the most recent simulation run'
    )
    parser.add_argument(
        '--search-dir',
        default=os.path.expanduser('~/research/pymnpbem/results'),
        help='Directory to search for latest run'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("pyMNPBEM Postprocessing Runner")
    print("=" * 60)

    # Determine run folder
    if args.run_folder:
        run_folder = args.run_folder
    elif args.latest:
        print(f"\nSearching for latest run in: {args.search_dir}")
        try:
            run_folder = find_latest_run(args.search_dir)
            print(f"Found latest run: {run_folder}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    else:
        print("Error: Please specify a run folder or use --latest")
        parser.print_help()
        return 1

    # Validate run folder
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        return 1

    config_path = os.path.join(run_folder, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: Not a valid run folder (missing config.json): {run_folder}")
        return 1

    print(f"\nProcessing run: {run_folder}")

    # Import and run postprocessing
    try:
        from postprocess import PostprocessManager

        # Override config if needed
        config_override = {}
        if args.no_plots:
            config_override['save_plots'] = False

        # Create manager and run
        manager = PostprocessManager(
            run_folder,
            config={'simulation': config_override} if config_override else None
        )

        data, analysis = manager.run(verbose=not args.quiet)

        # Print summary of results
        print("\n" + "=" * 60)
        print("Postprocessing Summary")
        print("=" * 60)

        if 'spectrum' in analysis:
            resonance_summary = analysis['spectrum'].get('resonance_summary', {})
            for pol_key, pol_data in resonance_summary.get('resonances', {}).items():
                print(f"\n{pol_key}:")
                for peak in pol_data.get('peaks', [])[:3]:
                    wl = peak.get('wavelength', 0)
                    ext = peak.get('extinction', 0)
                    fwhm = peak.get('fwhm')
                    fwhm_str = f", FWHM={fwhm:.1f}nm" if fwhm else ""
                    print(f"  Peak: λ={wl:.1f}nm, σ_ext={ext:.1f}nm²{fwhm_str}")

        if 'hotspots' in analysis:
            print("\nField Hotspots:")
            for pol_idx, hotspots in analysis['hotspots'].items():
                if hotspots:
                    top = hotspots[0]
                    pos = top['position']
                    enh = top['enhancement']
                    print(f"  Pol {pol_idx+1}: Max enhancement {enh:.1f}x at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) nm")

        print("\n" + "=" * 60)
        print(f"Results saved to: {run_folder}")
        print("=" * 60)

        return 0

    except ImportError as e:
        print(f"\nError importing postprocess modules: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\nError during postprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
