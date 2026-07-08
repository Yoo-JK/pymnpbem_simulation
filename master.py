#!/usr/bin/env python
"""master.py — one command; action is inferred from which configs you pass.

  # simulate only  (str + sim are a set)
  python master.py --str-conf S.py --sim-conf M.py [--verbose]

  # simulate then analyze + save  (add --anal-conf)
  python master.py --str-conf S.py --sim-conf M.py --anal-conf A.py

  # analyze only, no compute  (sim-conf just locates the existing output)
  python master.py --sim-conf M.py --anal-conf A.py

  # sweep (multi-case); analyze each if --anal-conf given
  python master.py --sweep-conf sweep.yaml [--anal-conf A.py]

Rules (규칙):
  - --str-conf + --sim-conf  -> run the simulation (they are a set; str needs sim).
  - --anal-conf              -> run the postprocess/analysis (optional; most runs skip it).
  - --sim-conf + --anal-conf without --str-conf -> postprocess ONLY the existing output
    (the sim-conf supplies output_dir/simulation_name so we can find spectrum.npz + sigma/).
Wraps run_simulation.py and run_postprocess.py; no --skip-* flags — presence of the
configs decides what runs. (분석은 --anal-conf 가 있을 때만; 대부분은 str+sim 만.)
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

    Accepts flat sim-conf keys (output_dir / simulation_name) or nested
    output.{dir,name}. (sim-conf 에서 출력 케이스 폴더를 계산 — 분석-only 에서 결과 위치 파악용.)
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
    if args.verbose:
        cmd += ['--verbose']
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
    """sim-conf paths to postprocess (one per case)."""
    if not args.sweep_conf:
        return [args.sim_conf]
    sys.path.insert(0, HERE)
    from pymnpbem_simulation.dispatch.sweep import _load_sweep_yaml, _expand_cases
    cases = _expand_cases(_load_sweep_yaml(args.sweep_conf), args.sweep_conf)
    return [c['sim_conf'] for c in cases]


def main(argv = None):
    p = argparse.ArgumentParser(
            prog = 'pymnpbem_master',
            description = 'Simulate and/or analyze; the action is inferred from the configs given.')
    p.add_argument('--str-conf', default = None,
            help = 'Structure config .py. With --sim-conf -> run the simulation.')
    p.add_argument('--sim-conf', default = None,
            help = 'Simulation config .py. Required for both simulate and analyze-only '
                   '(it supplies the output location).')
    p.add_argument('--anal-conf', default = None,
            help = 'Analysis config .py. Given -> run postprocess. Omit -> no analysis '
                   '(대부분의 실행은 생략).')
    p.add_argument('--sweep-conf', default = None,
            help = 'Sweep YAML (multi-case). Mutually exclusive with --str-conf/--sim-conf.')
    # simulation compute passthrough
    p.add_argument('--n-workers', type = int, default = None)
    p.add_argument('--n-threads', type = int, default = None)
    p.add_argument('--n-gpus-per-worker', type = int, default = None)
    p.add_argument('--auto', action = 'store_true')
    p.add_argument('--verbose', action = 'store_true',
            help = 'Forwarded to run_simulation.py.')
    # raw passthrough
    p.add_argument('--sim-extra', default = None,
            help = 'Extra flags forwarded verbatim to run_simulation.py (one string).')
    p.add_argument('--anal-extra', default = None,
            help = 'Extra flags forwarded verbatim to run_postprocess.py (one string).')
    args = p.parse_args(argv)

    # ---- validate + infer action from the configs present -------------
    if args.sweep_conf:
        if args.str_conf or args.sim_conf:
            p.error('--sweep-conf is mutually exclusive with --str-conf/--sim-conf')
    else:
        if args.str_conf and not args.sim_conf:
            p.error('--str-conf requires --sim-conf (they are a set)')
        if not args.sim_conf:
            p.error('need --sim-conf (with --str-conf to simulate, and/or '
                    '--anal-conf to analyze)')
        if not args.str_conf and not args.anal_conf:
            p.error('nothing to do — pass --str-conf to simulate and/or --anal-conf to analyze')

    do_sim = bool(args.sweep_conf) or (bool(args.str_conf) and bool(args.sim_conf))
    do_anal = bool(args.anal_conf)

    # ---- 1) simulation ------------------------------------------------
    if do_sim:
        if args.sweep_conf:
            rc = _run(_sim_cmd(args, sweep_conf = args.sweep_conf), 'SIMULATION (sweep)')
        else:
            rc = _run(_sim_cmd(args, str_conf = args.str_conf, sim_conf = args.sim_conf),
                      'SIMULATION')
        if rc != 0:
            print('[master] simulation failed — aborting.', flush = True)
            return rc
    else:
        print('[master] no --str-conf/--sweep-conf: analyze-only (using existing output).',
              flush = True)

    # ---- 2) postprocess per case (only when --anal-conf given) --------
    if not do_anal:
        print('[master] no --anal-conf: simulation only, done.', flush = True)
        return 0

    sim_confs = _enumerate_sim_confs(args)
    n_fail = 0
    for sc in sim_confs:
        case_dir = _resolve_case_dir(sc)
        if not case_dir:
            print('[master] WARN: cannot resolve output dir from <{}> — skip analysis.'.format(sc),
                  flush = True)
            n_fail += 1
            continue
        spec = os.path.join(case_dir, 'spectrum.npz')
        if not os.path.exists(spec):
            print('[master] WARN: <{}> not found — skip analysis.'.format(spec), flush = True)
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
