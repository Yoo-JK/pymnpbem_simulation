args = {
    'simulation_type': 'stat',
    'excitation_type': 'planewave',
    'wavelength_range': [400, 700, 4],
    'polarizations': [
        [1, 0, 0]],
    'propagation_dirs': [
        [0, 0, 1]],
    'compute': {
        'n_workers': 1,
        'n_threads': 1,
        'n_gpus_per_worker': 0,
        'multi_node': False,
        'hmode': 'dense'},
    'output': {
        'dir': './_sweep_results',
        'name': 'sphere_smoke',
        'formats': ['npz', 'json'],
        'save_plots': False}}
