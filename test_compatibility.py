import os
import sys
import itertools
import types
from typing import Dict, Any, List, Tuple


def _setup_mnpbem_mock() -> None:
    # mnpbem 라이브러리가 설치되지 않은 환경에서도 BEMSolver 초기화 테스트 가능
    if 'mnpbem' in sys.modules and hasattr(sys.modules['mnpbem'], '__file__'):
        return

    mock_modules = [
        'mnpbem',
        'mnpbem.geometry',
        'mnpbem.geometry.particle',
        'mnpbem.geometry.mesh_generators',
        'mnpbem.geometry.polygon',
        'mnpbem.geometry.edgeprofile',
        'mnpbem.bem',
        'mnpbem.simulation',
        'mnpbem.materials',
        'mnpbem.utils',
        'mnpbem.utils.constants',
        'mnpbem.greenfun',
        'mnpbem.greenfun.compgreen_stat',
        'mnpbem.greenfun.compgreen_ret',
    ]

    for mod_name in mock_modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # geometry mock classes
    geom = sys.modules['mnpbem.geometry']
    geom_particle = sys.modules['mnpbem.geometry.particle']
    for cls_name in ['ComParticle', 'ComParticleMirror', 'ComPoint', 'Particle', 'LayerStructure']:
        cls = type(cls_name, (), {})
        setattr(geom, cls_name, cls)
    setattr(geom_particle, 'Particle', type('Particle', (), {}))

    # mesh generators mock functions
    mesh_gen = sys.modules['mnpbem.geometry.mesh_generators']
    for func_name in ['trisphere', 'tricube', 'trirod', 'tripolygon', 'trispheresegment']:
        setattr(mesh_gen, func_name, lambda *a, **kw: None)

    # polygon / edgeprofile mock
    setattr(sys.modules['mnpbem.geometry.polygon'], 'Polygon', type('Polygon', (), {}))
    setattr(sys.modules['mnpbem.geometry.edgeprofile'], 'EdgeProfile', type('EdgeProfile', (), {}))

    # bem mock classes
    bem = sys.modules['mnpbem.bem']
    for cls_name in ['BEMStat', 'BEMRet', 'BEMStatMirror', 'BEMRetMirror',
                     'BEMStatLayer', 'BEMRetLayer', 'BEMStatIter', 'BEMRetIter',
                     'BEMRetLayerIter']:
        setattr(bem, cls_name, type(cls_name, (), {}))

    # simulation mock classes
    sim = sys.modules['mnpbem.simulation']
    for cls_name in ['PlaneWaveStat', 'PlaneWaveRet', 'PlaneWaveStatMirror', 'PlaneWaveRetMirror',
                     'DipoleStat', 'DipoleRet', 'DipoleStatMirror', 'DipoleRetMirror',
                     'EELSStat', 'EELSRet', 'PlaneWaveStatLayer', 'PlaneWaveRetLayer',
                     'DipoleStatLayer', 'DipoleRetLayer']:
        setattr(sim, cls_name, type(cls_name, (), {}))

    # materials mock classes
    mat = sys.modules['mnpbem.materials']
    for cls_name in ['EpsConst', 'EpsTable', 'EpsDrude']:
        setattr(mat, cls_name, type(cls_name, (), {}))

    # utils mock
    setattr(sys.modules['mnpbem.utils.constants'], 'EV2NM', 1239.8)

    # greenfun mock
    setattr(sys.modules['mnpbem.greenfun.compgreen_stat'], 'CompGreenStat', type('CompGreenStat', (), {}))
    setattr(sys.modules['mnpbem.greenfun.compgreen_ret'], 'CompGreenRet', type('CompGreenRet', (), {}))


_setup_mnpbem_mock()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.sim_utils.solver import BEMSolver


def build_test_config(
        sim_type: str,
        excitation_type: str,
        use_mirror: bool,
        use_substrate: bool,
        use_iterative: bool) -> Dict[str, Any]:

    config = {
        'simulation_type': sim_type,
        'excitation_type': excitation_type,
        'use_mirror_symmetry': 'xy' if use_mirror else False,
        'use_substrate': use_substrate,
        'use_iterative_solver': use_iterative,
        'wavelength_range': [400, 800, 10],
        'structure': 'sphere',
        'polarizations': [[1, 0, 0]],
        'propagation_dirs': [[0, 0, 1]],
        'calculate_cross_sections': True,
        'calculate_fields': False,
        'output_dir': '/tmp/test_compat',
    }

    if excitation_type == 'dipole':
        config['dipole_position'] = [0, 0, 15]
        config['dipole_moment'] = [0, 0, 1]

    if excitation_type == 'eels':
        config['impact_parameter'] = [10, 0]
        config['beam_energy'] = 200e3
        config['beam_width'] = 0.2

    if use_substrate:
        config['substrate'] = {
            'medium_idx': 0,
            'substrate_idx': 1,
            'z_interface': 0.0,
        }

    return config


def label(val: bool) -> str:
    return 'ON' if val else 'OFF'


def main() -> None:

    sim_types = ['stat', 'ret']
    excitation_types = ['planewave', 'dipole', 'eels']
    mirror_vals = [False, True]
    substrate_vals = [False, True]
    iterative_vals = [False, True]

    combos = list(itertools.product(
        sim_types, excitation_types, mirror_vals, substrate_vals, iterative_vals))

    print('=' * 80)
    print('pymnpbem_simulation: Compatibility Test')
    print('Total combinations: {}'.format(len(combos)))
    print('=' * 80)
    print()

    accepted = []
    rejected = []
    unexpected = []

    for sim_type, exc_type, mirror, substrate, iterative in combos:
        desc = 'sim={}, exc={}, mirror={}, substrate={}, iterative={}'.format(
            sim_type, exc_type, label(mirror), label(substrate), label(iterative))

        config = build_test_config(sim_type, exc_type, mirror, substrate, iterative)

        try:
            solver = BEMSolver(config, verbose = False)
            accepted.append(desc)

        except ValueError as e:
            rejected.append((desc, str(e)))

        except Exception as e:
            unexpected.append((desc, type(e).__name__, str(e)))

    # Report
    print('--- Accepted ({}) ---'.format(len(accepted)))
    for desc in accepted:
        print('  [OK] {}'.format(desc))

    print()
    print('--- Rejected ({}) ---'.format(len(rejected)))
    for desc, reason in rejected:
        print('  [REJECTED] {}'.format(desc))
        print('             {}'.format(reason))

    print()
    print('--- Unexpected Errors ({}) ---'.format(len(unexpected)))
    for desc, exc_name, msg in unexpected:
        print('  [ERROR] {}'.format(desc))
        print('          {}: {}'.format(exc_name, msg))

    print()
    print('=' * 80)
    print('Summary: {} accepted, {} rejected, {} unexpected'.format(
        len(accepted), len(rejected), len(unexpected)))
    print('=' * 80)

    if unexpected:
        sys.exit(1)


if __name__ == '__main__':
    main()
