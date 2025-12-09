#!/usr/bin/env python3
"""
pyMNPBEM Simulation Runner

Main entry point for running plasmonic BEM simulations.
Loads configuration from config files and executes the simulation pipeline.

Usage:
    python run_simulation.py

Configuration:
    - config/structure/config_structure.py : Structure parameters
    - config/simulation/config_simulation.py : Simulation parameters
"""

import sys
import os
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from a Python file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.args


def validate_config(structure_config: dict, sim_config: dict) -> bool:
    """Validate configuration parameters."""
    errors = []

    # Check required structure parameters
    if 'structure' not in structure_config:
        errors.append("Missing 'structure' in structure config")

    if 'materials' not in structure_config:
        errors.append("Missing 'materials' in structure config")

    # Check required simulation parameters
    if 'wavelength_range' not in sim_config:
        errors.append("Missing 'wavelength_range' in simulation config")

    # Print errors
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def main():
    """Main entry point for simulation."""
    parser = argparse.ArgumentParser(
        description='Run pyMNPBEM plasmonic simulation'
    )
    parser.add_argument(
        '--structure-config',
        default='config/structure/config_structure.py',
        help='Path to structure configuration file'
    )
    parser.add_argument(
        '--sim-config',
        default='config/simulation/config_simulation.py',
        help='Path to simulation configuration file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Override output directory'
    )

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.resolve()

    # Load configurations
    structure_config_path = script_dir / args.structure_config
    sim_config_path = script_dir / args.sim_config

    print("=" * 60)
    print("pyMNPBEM Simulation Runner")
    print("=" * 60)

    print(f"\nLoading structure config: {structure_config_path}")
    structure_config = load_config(str(structure_config_path))

    print(f"Loading simulation config: {sim_config_path}")
    sim_config = load_config(str(sim_config_path))

    # Override output directory if specified
    if args.output_dir:
        sim_config['output_dir'] = args.output_dir

    # Validate configurations
    print("\nValidating configurations...")
    if not validate_config(structure_config, sim_config):
        print("\nConfiguration validation failed. Exiting.")
        sys.exit(1)

    print("Configuration valid.")

    # Print summary
    print("\n" + "-" * 60)
    print("Simulation Summary:")
    print("-" * 60)
    print(f"  Structure: {structure_config.get('structure', 'unknown')}")
    print(f"  Structure name: {structure_config.get('structure_name', 'unnamed')}")
    print(f"  Materials: {structure_config.get('materials', [])}")
    print(f"  Medium: {structure_config.get('medium', 'air')}")
    print(f"  Simulation type: {sim_config.get('simulation_type', 'stat')}")
    print(f"  Wavelength range: {sim_config.get('wavelength_range', [])}")
    print(f"  Calculate fields: {sim_config.get('calculate_fields', False)}")
    print(f"  Calculate surface charges: {sim_config.get('calculate_surface_charges', False)}")
    print(f"  Output directory: {sim_config.get('output_dir', './results')}")
    print("-" * 60)

    # Import and run simulation
    try:
        from simulation import SimulationRunner

        # Get pyMNPBEM path
        pymnpbem_path = sim_config.get('pymnpbem_path')
        if pymnpbem_path:
            sys.path.insert(0, str(pymnpbem_path))

        # Create and run simulation
        runner = SimulationRunner(
            structure_config,
            sim_config,
            pymnpbem_path=pymnpbem_path
        )

        results = runner.run(show_progress=not args.quiet)

        print("\n" + "=" * 60)
        print("Simulation completed successfully!")
        print(f"Results saved to: {runner.get_run_folder()}")
        print("=" * 60)

        return 0

    except ImportError as e:
        print(f"\nError importing simulation modules: {e}")
        print("\nMake sure pyMNPBEM is installed and the path is correct.")
        print(f"Current pymnpbem_path: {sim_config.get('pymnpbem_path', 'Not set')}")
        return 1

    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
