import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from postprocess.postprocess import PostprocessManager


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Postprocess pyMNPBEM simulation results'
    )

    parser.add_argument(
        '--str-conf',
        type = str,
        required = True,
        help = 'Path to structure configuration file'
    )
    parser.add_argument(
        '--sim-conf',
        type = str,
        required = True,
        help = 'Path to simulation configuration file'
    )
    parser.add_argument(
        '--verbose',
        action = 'store_true',
        help = 'Enable verbose output'
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            '[error] Config file not found: <{}>'.format(config_path))

    config_dict = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config_dict)

    if 'args' not in config_dict:
        raise ValueError(
            '[error] Config file must contain <args> dictionary: <{}>'.format(config_path))

    return config_dict['args']


def merge_configs(
        structure_path: str,
        simulation_path: str,
        verbose: bool = False) -> Dict[str, Any]:

    if verbose:
        print('[info] Loading structure config: {}'.format(structure_path))
    structure_args = load_config(structure_path)

    if verbose:
        print('[info] Loading simulation config: {}'.format(simulation_path))
    simulation_args = load_config(simulation_path)

    # Merge: simulation settings override structure if there's overlap
    merged = {**structure_args, **simulation_args}

    if verbose:
        print('[info] Configurations loaded and merged successfully')

    return merged


def main() -> int:
    args = parse_arguments()

    try:
        # Load and merge configurations
        config = merge_configs(args.str_conf, args.sim_conf, verbose = args.verbose)

        # Initialize postprocessing manager
        postprocess = PostprocessManager(config, verbose = args.verbose)

        # Run postprocessing
        result = postprocess.run()

        # Unpack based on return type
        if len(result) == 3:
            data, analysis, field_analysis = result
        elif len(result) == 2:
            data, analysis = result
            field_analysis = []
        else:
            raise ValueError(
                '[error] Unexpected return value from postprocess.run(): <{}>'.format(result))

        # Print summary
        if args.verbose:
            print('\n[info] Postprocessing completed successfully')

            # Print field data info if available
            if 'fields' in data and data['fields']:
                print('\n  Field data processed:')
                for i, field in enumerate(data['fields']):
                    print('    Polarization {}: lambda = {:.1f} nm'.format(
                        i + 1, field['wavelength']))

                    enhancement = field['enhancement']
                    if hasattr(enhancement, 'shape'):
                        print('      Grid size: {}'.format(enhancement.shape))
                    else:
                        print('      Grid size: (1,) - single point')

        return 0

    except FileNotFoundError as e:
        print('[error] {}'.format(e), file = sys.stderr)
        return 1
    except Exception as e:
        print('[error] Error during postprocessing: {}'.format(e), file = sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
