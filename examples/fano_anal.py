# Analysis config for run_postprocess.py --anal-conf
# (analyzer hyperparameters; same style as --str-conf / --sim-conf).
#
# Precedence: explicit CLI flag > this file > built-in default.
# Keys = argparse dest names (underscored). Comma-string options also accept
# a Python list; polarizations accepts a nested list.
#
#   python run_postprocess.py --anal-conf examples/fano_anal.py \
#       --result /path/to/case/spectrum.npz
#
import os

_CASE = os.path.join(os.path.expanduser('~'),
        'research', 'pymnpbem', 'au_dimer', 'nosub', 'au_r0.2_g0.6')

args = {
    'analyzers': ['spectrum', 'fano-analysis'],   # or 'spectrum,fano-analysis'
    'xaxis': 'energy',                            # wavelength | energy
    'fano_features': [1.43, 1.79, 1.91],          # Fano dip energies in eV
    'fano_pol': 0,                                # polarization index for sigma
    'n_modes': 10,                                # qs eigenmodes
    'max_l': 4,                                   # multipole order
    'export_formats': 'json,csv',                 # npz,h5,csv,json,txt
    # --- optional paths (else pass on CLI) -------------------------------
    # 'result': os.path.join(_CASE, 'spectrum.npz'),
    # 'case_dir': _CASE,
    # 'config': os.path.join(_CASE, 'config.yaml'),
    # 'eig_cache': os.path.join(_CASE, 'eig_cache.npz'),
    # 'polarizations': [[1, 0, 0], [0, 1, 0]],
    # 'output': os.path.join(_CASE, 'postprocess'),
}
