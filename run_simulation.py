"""
pyMNPBEM Simulation Runner

This script runs BEM simulations based on structure and simulation configuration files.
Replaces MATLAB MNPBEM with Python pyMNPBEM.
"""

import argparse
import sys
import os
from pathlib import Path

# Add simulation module to path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.calculate import SimulationManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run pyMNPBEM simulation'
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


def validate_config(args):
    """Validate configuration parameters."""
    required_keys = [
        'structure',
        'simulation_type',
        'excitation_type',
        'wavelength_range',
        'output_dir'
    ]

    missing_keys = []
    for key in required_keys:
        if key not in args:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Required configuration keys missing: {', '.join(missing_keys)}")

    # Validate simulation type
    if args['simulation_type'] not in ['stat', 'ret']:
        raise ValueError(f"Invalid simulation_type: {args['simulation_type']}")

    # Validate excitation type
    if args['excitation_type'] not in ['planewave', 'dipole', 'eels']:
        raise ValueError(f"Invalid excitation_type: {args['excitation_type']}")

    return True


def main():
    """Main simulation function."""
    args = parse_arguments()

    try:
        # Load and merge configurations
        config = merge_configs(args.str_conf, args.sim_conf, args.verbose)

        # Validate configuration
        validate_config(config)

        # Initialize simulation manager
        sim_manager = SimulationManager(config, verbose=args.verbose)

        # Create run folder
        run_folder = sim_manager.create_run_folder()

        # Save configuration snapshot
        sim_manager.save_config_snapshot()

        # Run the simulation (replaces MATLAB code generation + execution)
        sim_manager.run_simulation()

        # Save results
        sim_manager.save_results()

        # Print run folder for master.sh to capture
        print(f"RUN_FOLDER={run_folder}")

        if args.verbose:
            print("\n✓ Simulation completed successfully")
            summary = sim_manager.get_summary()
            print(f"  Structure: {summary['structure']}")
            print(f"  Simulation type: {summary['simulation_type']}")
            print(f"  Wavelength range: {summary['wavelength_range']}")
            print(f"  Run folder: {summary['run_folder']}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
