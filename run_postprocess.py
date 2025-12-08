"""
MNPBEM Postprocessing Runner

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
        description='Postprocess MNPBEM simulation results'
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


def merge_configs(structure_path, simulation_path):
    """Merge structure and simulation configs."""
    if args.verbose:
        print(f"Loading structure config: {structure_path}")
    structure_args = load_config(structure_path)
    
    if args.verbose:
        print(f"Loading simulation config: {simulation_path}")
    simulation_args = load_config(simulation_path)
    
    # Merge: simulation settings override structure if there's overlap
    merged = {**structure_args, **simulation_args}
    
    if args.verbose:
        print(f"✓ Configurations loaded and merged successfully")
    
    return merged


def main():
    """Main postprocessing function."""
    global args
    args = parse_arguments()
    
    try:
        # Load and merge configurations
        config = merge_configs(args.str_conf, args.sim_conf)
        
        # Initialize postprocessing manager
        postprocess = PostprocessManager(config, verbose=args.verbose)
        
        # Run postprocessing
        # ✅ FIX: Handle three return values
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
            
            # Print field data info if available
            if 'fields' in data and data['fields']:
                print(f"\n  Field data processed:")
                for i, field in enumerate(data['fields']):
                    print(f"    Polarization {i+1}: λ = {field['wavelength']:.1f} nm")

                    enhancement = field['enhancement']
                    if hasattr(enhancement, 'shape'):
                        print(f"      Grid size: {enhancement.shape}")
                    else:
                        print(f"      Grid size: (1,) - single point")
        
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
