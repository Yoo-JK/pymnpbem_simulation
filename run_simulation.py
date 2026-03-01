import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Run pyMNPBEM simulation'
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
        '--mnpbem-path',
        type = str,
        default = None,
        help = 'Path to mnpbem source directory (e.g. ~/workspace/MNPBEM)'
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


def merge_configs(structure_path: str, simulation_path: str) -> Dict[str, Any]:
    print('[info] Loading structure config: {}'.format(structure_path))
    structure_args = load_config(structure_path)

    print('[info] Loading simulation config: {}'.format(simulation_path))
    simulation_args = load_config(simulation_path)

    # Merge: simulation settings override structure if there's overlap
    merged = {**structure_args, **simulation_args}

    print('[info] Configurations loaded and merged successfully')

    return merged


def validate_config(args: Dict[str, Any]) -> None:
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
        raise ValueError(
            '[error] Required configuration keys missing: <{}>'.format(', '.join(missing_keys)))

    assert args['simulation_type'] in ['stat', 'ret'], \
        '[error] Invalid <simulation_type>: {}'.format(args['simulation_type'])

    assert args['excitation_type'] in ['planewave', 'dipole', 'eels'], \
        '[error] Invalid <excitation_type>: {}'.format(args['excitation_type'])

    # Compatibility checks
    use_mirror = args.get('use_mirror_symmetry', False)
    use_substrate = args.get('use_substrate', False)
    use_iterative = args.get('use_iterative_solver', False)
    excitation_type = args['excitation_type']
    structure = args.get('structure', '')

    if use_mirror and use_substrate:
        raise ValueError(
            '[error] Mirror symmetry + substrate is not supported. '
            'BEMStatMirrorLayer / BEMRetMirrorLayer classes do not exist in mnpbem.')

    if use_mirror and use_iterative:
        raise ValueError(
            '[error] Mirror symmetry + iterative solver is not supported. '
            'BEMStatMirrorIter / BEMRetMirrorIter classes do not exist in mnpbem.')

    if excitation_type == 'eels' and use_substrate:
        raise ValueError(
            '[error] EELS excitation + substrate is not supported. '
            'EELSStatLayer / EELSRetLayer classes do not exist in mnpbem.')

    if excitation_type == 'eels' and use_mirror:
        raise ValueError(
            '[error] EELS excitation is NOT compatible with mirror symmetry.')

    if use_mirror and structure not in ('sphere', 'dimer_sphere', 'dimer', ''):
        warnings.warn(
            '[info] Mirror symmetry with structure <{}> may not be valid. '
            'Ensure your mesh has the required symmetry.'.format(structure))

    print('[info] Configuration validated successfully')


def main() -> None:
    args_cli = parse_arguments()

    print('=' * 60)
    print('pyMNPBEM Simulation Runner')
    print('=' * 60)
    print()

    # Load and merge configurations (before importing mnpbem)
    try:
        config = merge_configs(args_cli.str_conf, args_cli.sim_conf)
    except Exception as e:
        print('[error] Error loading configuration: {}'.format(e))
        sys.exit(1)

    # Resolve mnpbem path: CLI arg > config > already installed
    mnpbem_path = args_cli.mnpbem_path or config.get('mnpbem_path', None)
    if mnpbem_path is not None:
        mnpbem_path = str(Path(mnpbem_path).expanduser().resolve())
        sys.path.insert(0, mnpbem_path)
        print('[info] mnpbem path: {}'.format(mnpbem_path))

    from simulation.calculate import SimulationManager

    # Validate configuration
    print('\nValidating configuration...')
    try:
        validate_config(config)
    except Exception as e:
        print('[error] Configuration validation failed: {}'.format(e))
        sys.exit(1)

    # Create simulation manager
    print('\nInitializing simulation manager...')
    try:
        sim_manager = SimulationManager(config, verbose = args_cli.verbose)
        print('[info] Simulation manager initialized')
    except Exception as e:
        print('[error] Error initializing simulation manager: {}'.format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create run folder
    print('\nCreating run folder...')
    try:
        run_folder = sim_manager.create_run_folder()
        print('[info] Run folder created: {}'.format(run_folder))
    except Exception as e:
        print('[error] Error creating run folder: {}'.format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save config snapshot
    print('\nSaving config snapshot...')
    try:
        sim_manager.save_config_snapshot()
    except Exception as e:
        print('[error] Error saving config snapshot: {}'.format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run simulation directly (no MATLAB code generation)
    print('\nRunning BEM simulation...')
    try:
        sim_manager.run()
    except Exception as e:
        print('[error] Error running simulation: {}'.format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print('[info] Simulation completed successfully')

    # Print summary
    print('\n' + '=' * 60)
    print('Simulation Complete')
    print('=' * 60)
    print('Structure:        {}'.format(
        config.get('structure_name', config.get('structure', 'N/A'))))
    print('Structure type:   {}'.format(config.get('structure', 'N/A')))
    print('Simulation:       {}'.format(config.get('simulation_name', 'N/A')))
    print('Simulation type:  {}'.format(config['simulation_type']))
    print('Excitation:       {}'.format(config['excitation_type']))
    print('Wavelength range: {}-{} nm ({} points)'.format(
        config['wavelength_range'][0],
        config['wavelength_range'][1],
        config['wavelength_range'][2]))
    print('Run folder:       {}'.format(run_folder))
    print()
    print('=' * 60)

    # Export run folder path for master.sh to use
    print('\nRUN_FOLDER={}'.format(run_folder))


if __name__ == '__main__':
    main()
