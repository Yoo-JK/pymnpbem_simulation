#!/usr/bin/env python
"""master.py — simulation -> postprocess 를 한 번에 (config-driven end-to-end).
Run the full pipeline (simulate then analyze) from configs in a single command.

single case (단일 케이스):
  python master.py --str-conf S.py --sim-conf M.py --anal-conf A.py \
      [--n-workers N --n-threads N --n-gpus-per-worker N --auto]

sweep (다중 케이스):
  python master.py --sweep-conf sweep.yaml --anal-conf A.py [옵션]

내부적으로 run_simulation.py 를 돌려 spectrum.npz + sigma 캐시를 만들고, 각 케이스의
출력(output_dir/simulation_name)을 찾아 run_postprocess.py --anal-conf 로 분석까지 이어 실행한다.
(Wraps run_simulation.py then run_postprocess.py; locates each case output from its sim
config and feeds --result/--case-dir to postprocess. --anal-conf hyperparameters are
config-driven, mirroring --str-conf/--sim-conf.)
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUN_SIM = os.path.join(HERE, 'run_simulation.py')
RUN_POST = os.path.join(HERE, 'run_postprocess.py')


def _resolve_case_dir(sim_conf_path):
    """Return <output_dir>/<simulation_name> for a sim-conf, or None.

    Accepts the flat sim-conf keys (output_dir / simulation_name) or the nested
    output.{dir,name} form. (sim-conf 에서 출력 케이스 폴더를 계산.)
    """
    sys.path.insert(0, HERE)
    from pymnpbem_simulation.config import load_py_config
    a = load_py_config(sim_conf_path)
    out = a.get('output', {}) if isinstance(a.get('output'), dict) else {}
    out_dir = a.get('output_dir') or out.get('dir')
    name = a.get('simulation_name') or out.get('name')
    if not out_dir or not name:
        return None
    return os.path.join(str(out_dir), str(name))


def _run(cmd, label):
    print('\n[master] === {} ==='.format(label), flush = True)
    print('[master] $ {}'.format(' '.join(cmd)), flush = True)
    rc = subprocess.call(cmd)
    print('[master] {} -> rc={}'.format(label, rc), flush = True)
    return rc


def _sim_cmd(args, str_conf = None, sim_conf = None, sweep_conf = None):
    cmd = [PY, RUN_SIM]
    if sweep_conf:
        cmd += ['--sweep-conf', sweep_conf]
    else:
        cmd += ['--str-conf', str_conf, '--sim-conf', sim_conf]
    for flag, val in (('--n-workers', args.n_workers),
                      ('--n-threads', args.n_threads),
                      ('--n-gpus-per-worker', args.n_gpus_per_worker)):
        if val is not None:
            cmd += [flag, str(val)]
    if args.auto:
        cmd += ['--auto']
    if args.sim_extra:
        cmd += args.sim_extra.split()
    return cmd


def _anal_cmd(args, case_dir):
    cmd = [PY, RUN_POST,
           '--anal-conf', args.anal_conf,
           '--result', os.path.join(case_dir, 'spectrum.npz'),
           '--case-dir', case_dir]
    if args.anal_extra:
        cmd += args.anal_extra.split()
    return cmd


def _enumerate_sim_confs(args):
    """List of sim-conf paths to postprocess (one per case)."""
    if not args.sweep_conf:
        return [args.sim_conf]
    sys.path.insert(0, HERE)
    from pymnpbem_simulation.dispatch.sweep import _load_sweep_yaml, _expand_cases
    cases = _expand_cases(_load_sweep_yaml(args.sweep_conf), args.sweep_conf)
    return [c['sim_conf'] for c in cases]


def main(argv = None):
    p = argparse.ArgumentParser(
            prog = 'pymnpbem_master',
            description = 'Run simulation then postprocess in one command (config-driven).')
    p.add_argument('--str-conf', default = None)
    p.add_argument('--sim-conf', default = None)
    p.add_argument('--sweep-conf', default = None,
            help = 'Sweep YAML (multi-case). Mutually exclusive with --str-conf/--sim-conf.')
    p.add_argument('--anal-conf', default = None,
            help = 'Analysis config .py (run_postprocess --anal-conf). Omit to skip analysis.')
    # simulation compute passthrough
    p.add_argument('--n-workers', type = int, default = None)
    p.add_argument('--n-threads', type = int, default = None)
    p.add_argument('--n-gpus-per-worker', type = int, default = None)
    p.add_argument('--auto', action = 'store_true')
    # phase toggles
    p.add_argument('--skip-sim', action = 'store_true',
            help = '시뮬 건너뛰고 기존 출력만 분석 (analyze existing output).')
    p.add_argument('--skip-analysis', action = 'store_true',
            help = '시뮬만 하고 분석 생략 (simulate only).')
    # raw passthrough
    p.add_argument('--sim-extra', default = None,
            help = 'run_simulation.py 로 그대로 넘길 추가 플래그 (한 문자열).')
    p.add_argument('--anal-extra', default = None,
            help = 'run_postprocess.py 로 그대로 넘길 추가 플래그 (한 문자열).')
    args = p.parse_args(argv)

    if not args.sweep_conf and not (args.sim_conf and (args.str_conf or args.skip_sim)):
        p.error('either --sweep-conf, or --str-conf + --sim-conf '
                '(--str-conf optional with --skip-sim) is required')
    if args.sweep_conf and (args.str_conf or args.sim_conf):
        p.error('--sweep-conf is mutually exclusive with --str-conf/--sim-conf')
    if not args.skip_analysis and not args.anal_conf:
        p.error('--anal-conf is required unless --skip-analysis')

    # ---- 1) simulation ------------------------------------------------
    if not args.skip_sim:
        if args.sweep_conf:
            rc = _run(_sim_cmd(args, sweep_conf = args.sweep_conf), 'SIMULATION (sweep)')
        else:
            rc = _run(_sim_cmd(args, str_conf = args.str_conf, sim_conf = args.sim_conf),
                      'SIMULATION')
        if rc != 0:
            print('[master] simulation failed — aborting.', flush = True)
            return rc
    else:
        print('[master] --skip-sim: 기존 출력 사용 (skip simulation).', flush = True)

    # ---- 2) postprocess per case -------------------------------------
    if args.skip_analysis:
        print('[master] --skip-analysis: 분석 생략, done.', flush = True)
        return 0

    sim_confs = _enumerate_sim_confs(args)
    n_fail = 0
    for sc in sim_confs:
        case_dir = _resolve_case_dir(sc)
        if not case_dir:
            print('[master] WARN: <{}> 에서 output 경로 못 찾음 — 분석 skip.'.format(sc), flush = True)
            n_fail += 1
            continue
        spec = os.path.join(case_dir, 'spectrum.npz')
        if not os.path.exists(spec):
            print('[master] WARN: <{}> 없음 — 분석 skip.'.format(spec), flush = True)
            n_fail += 1
            continue
        rc = _run(_anal_cmd(args, case_dir),
                  'ANALYSIS [{}]'.format(os.path.basename(case_dir)))
        if rc != 0:
            n_fail += 1

    print('\n[master] ALL DONE — {} case(s), {} analysis failure(s).'.format(
            len(sim_confs), n_fail), flush = True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
