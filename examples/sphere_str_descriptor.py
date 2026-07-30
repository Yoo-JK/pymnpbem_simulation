from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent


args = {
    'structure': 'sphere',
    'diameter': 50,
    'mesh_density': 5,
    'materials': ['gold_drude_user'],
    'medium': 'water',
    'use_substrate': False,
    # Use descriptor form so worker/multi-node runs resolve the callable
    # inside each process instead of trying to pickle a live Python function.
    'refractive_index_paths': {
        'gold_drude_user': {
            'type': 'python_module',
            'module_path': str(_THIS_DIR / 'user_material_gold_drude.py'),
            'factory': 'generate_eps_func'}}}
