"""
pyMNPBEM Postprocessing Runner

This script loads simulation results and performs analysis and visualization.
"""

import argparse
import sys
import os
from pathlib import Path

# Add postprocess module to path
sys.path.insert(0, str(Path(__file__).parent))

from postprocess.postprocess import PostprocessManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Postprocess pyMNPBEM simulation results'
    )

    parser.add_argument(
        '--str-conf',
        type=str,
        required=True,
        help='Path to structure configuration file'
    )
    parser.add_argument(
        '--sim-conf',
        type=str,
        required=True,
        help='Path to simulation configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from Python file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config file as module
    config_dict = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config_dict)

    if 'args' not in config_dict:
        raise ValueError(f"Config file must contain 'args' dictionary: {config_path}")

    return config_dict['args']


def merge_configs(structure_path, simulation_path, verbose=False):
    """Merge structure and simulation configs."""
    if verbose:
        print(f"Loading structure config: {structure_path}")
    structure_args = load_config(structure_path)

    if verbose:
        print(f"Loading simulation config: {simulation_path}")
    simulation_args = load_config(simulation_path)

    # Merge: simulation settings override structure if there's overlap
    merged = {**structure_args, **simulation_args}

    if verbose:
        print(f"✓ Configurations loaded and merged successfully")

    return merged


def main():
    """Main postprocessing function."""
    args = parse_arguments()

    try:
        # Load and merge configurations
        config = merge_configs(args.str_conf, args.sim_conf, args.verbose)

        # Initialize postprocessing manager
        postprocess = PostprocessManager(config, verbose=args.verbose)

        # Run postprocessing
        result = postprocess.run()

        # Unpack based on return type
        if len(result) == 3:
            data, analysis, field_analysis = result
        elif len(result) == 2:
            data, analysis = result
            field_analysis = []
        else:
            raise ValueError(f"Unexpected return value from postprocess.run(): {result}")

        # Print summary
        if args.verbose:
            print("\n✓ Postprocessing completed successfully")

            # Print spectrum info
            if 'spectrum' in data:
                print(f"\n  Spectrum data processed:")
                wavelengths = data['spectrum'].get('wavelengths', [])
                if len(wavelengths) > 0:
                    print(f"    Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
                    print(f"    Number of points: {len(wavelengths)}")
                else:
                    print(f"    No wavelength data available")

            # Print field data info if available
            if 'field' in data and data['field']:
                print(f"\n  Field data processed:")
                for pol_idx, field in data['field'].items():
                    print(f"    Polarization {pol_idx+1}:")
                    enhancement = field.get('enhancement')
                    if enhancement is not None and hasattr(enhancement, 'shape') and enhancement.size > 0:
                        print(f"      Grid size: {enhancement.shape}")
                        print(f"      Max enhancement: {enhancement.max():.1f}")

            # Print surface charge info if available
            if 'surface_charges' in data and data['surface_charges']:
                print(f"\n  Surface charge data processed:")
                for pol_idx, charge in data['surface_charges'].items():
                    print(f"    Polarization {pol_idx+1}:")
                    if 'charges' in charge and hasattr(charge['charges'], 'shape'):
                        print(f"      Number of surface elements: {charge['charges'].shape[0]}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during postprocessing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
