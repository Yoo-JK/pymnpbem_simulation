"""Run the full regression suite and report grade distribution.

Usage:
  python tests/regression/runners/run_full_regression.py
  python tests/regression/runners/run_full_regression.py --markers fast
  python tests/regression/runners/run_full_regression.py --markers "fast or slow"
  python tests/regression/runners/run_full_regression.py --json artifacts/result.json

Exit code:
  0 = all PASS, no BAD grade
  1 = one or more tests failed or BAD grades
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

from pathlib import Path
from typing import Any, Dict, List


REGRESSION_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = REGRESSION_DIR.parent.parent

GRADE_RE = re.compile(r'\bgrade=(machine|OK|good|warn|BAD)\b')


def _parse_grades_from_output(text: str) -> List[str]:
    return GRADE_RE.findall(text)


def run_pytest(markers: str, extra_args: List[str]) -> Dict[str, Any]:
    cmd = [
            sys.executable, '-m', 'pytest',
            str(REGRESSION_DIR),
            '--tb=short', '-v', '-s']

    if markers:
        cmd.extend(['-m', markers])
    if extra_args:
        cmd.extend(extra_args)

    print('[runner] cmd: {}'.format(' '.join(cmd)))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd = REPO_ROOT, capture_output = True, text = True)
    elapsed = time.time() - t0

    out = proc.stdout + '\n' + proc.stderr
    grades = _parse_grades_from_output(out)

    return {
            'returncode': proc.returncode,
            'elapsed_s': elapsed,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'grades': grades}


def grade_distribution(grades: List[str]) -> Dict[str, int]:
    dist = {'machine': 0, 'OK': 0, 'good': 0, 'warn': 0, 'BAD': 0}
    for g in grades:
        if g in dist:
            dist[g] += 1
    return dist


def main():
    parser = argparse.ArgumentParser(
            description = 'Run regression suite and grade results')
    parser.add_argument('--markers', type = str, default = 'fast or slow',
            help = 'pytest -m expression (default: "fast or slow")')
    parser.add_argument('--json', type = str, default = None,
            help = 'Optional output JSON for grade summary')
    parser.add_argument('--extra', type = str, default = '',
            help = 'Extra pytest args (space-separated)')
    args = parser.parse_args()

    extra = args.extra.split() if args.extra else []

    print('=' * 70)
    print('[runner] pymnpbem_simulation regression — markers={}'.format(args.markers))
    print('=' * 70)

    res = run_pytest(args.markers, extra)
    dist = grade_distribution(res['grades'])
    total_grades = sum(dist.values())

    print('=' * 70)
    print('[runner] returncode = {}'.format(res['returncode']))
    print('[runner] wall-time = {:.1f}s'.format(res['elapsed_s']))
    print('[runner] grades collected: {}'.format(total_grades))
    print('[runner] distribution:')
    for g in ['machine', 'OK', 'good', 'warn', 'BAD']:
        cnt = dist[g]
        pct = (100.0 * cnt / total_grades) if total_grades else 0.0
        print('         {:>8s}: {:3d} ({:5.1f}%)'.format(g, cnt, pct))
    print('=' * 70)

    summary = {
            'returncode': res['returncode'],
            'elapsed_s': res['elapsed_s'],
            'markers': args.markers,
            'grade_total': total_grades,
            'grade_distribution': dist,
            'machine_precision_pct': (100.0 * dist['machine'] / total_grades
                    if total_grades else 0.0),
            'bad_count': dist['BAD']}

    if args.json:
        Path(args.json).parent.mkdir(parents = True, exist_ok = True)
        with open(args.json, 'w') as f:
            json.dump(summary, f, indent = 2)
        print('[runner] wrote summary <{}>'.format(args.json))

    if res['returncode'] != 0:
        print('[runner] FAIL: pytest returned non-zero')
        sys.exit(1)

    if dist['BAD'] > 0:
        print('[runner] FAIL: {} BAD grades'.format(dist['BAD']))
        sys.exit(1)

    print('[runner] PASS')
    sys.exit(0)


if __name__ == '__main__':
    main()
