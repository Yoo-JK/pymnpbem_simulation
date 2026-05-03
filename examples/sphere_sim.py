args = {
    'simulation_type': 'stat',
    'excitation_type': 'planewave',
    'wavelength_range': [400, 800, 50],
    'polarizations': [
        [1, 0, 0]],
    'propagation_dirs': [
        [0, 0, 1]],
    'interp': 'curv',
    'relcutoff': 3,
    'calculate_cross_sections': True,
    'calculate_fields': False,
    'compute': {
        'n_workers': 1,
        'n_threads': 4,
        'n_gpus_per_worker': 0,
        'multi_node': False,
        'hmode': 'dense'},
    'output': {
        'dir': './results',
        'name': 'sphere_au_50nm',
        'formats': ['json', 'npz', 'png'],
        'save_plots': True}}
