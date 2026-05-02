"""Grade simulation result vs reference value.

Grade definitions (relative error):
  machine: rel < 1e-12
  OK:      rel < 1e-9
  good:    rel < 1e-6
  warn:    rel < 1e-3
  BAD:     rel >= 1e-3

Used by:
  - tests/regression/test_*.py (assertion of grade)
  - tests/regression/runners/run_full_regression.py (summary)
"""
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Dict, Any, List, Tuple


GRADE_ORDER = ['machine', 'OK', 'good', 'warn', 'BAD']
GRADE_THRESHOLDS = {
        'machine': 1e-12,
        'OK': 1e-9,
        'good': 1e-6,
        'warn': 1e-3}


def compute_grade(measured: float, reference: float) -> str:
    if reference == 0.0:
        return 'machine' if measured == 0.0 else 'BAD'

    rel = abs(measured - reference) / abs(reference)

    if rel < GRADE_THRESHOLDS['machine']:
        return 'machine'
    if rel < GRADE_THRESHOLDS['OK']:
        return 'OK'
    if rel < GRADE_THRESHOLDS['good']:
        return 'good'
    if rel < GRADE_THRESHOLDS['warn']:
        return 'warn'
    return 'BAD'


def relative_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if measured == 0.0 else float('inf')
    return abs(measured - reference) / abs(reference)


def grade_distribution(grades: List[str]) -> Dict[str, int]:
    dist = {g: 0 for g in GRADE_ORDER}
    for g in grades:
        dist[g] = dist.get(g, 0) + 1
    return dist


def grade_summary(grades: List[str]) -> str:
    dist = grade_distribution(grades)
    total = sum(dist.values())
    if total == 0:
        return '(no grades)'

    parts = []
    for g in GRADE_ORDER:
        cnt = dist[g]
        pct = 100.0 * cnt / total
        parts.append('{:s}={:d} ({:.1f}%)'.format(g, cnt, pct))
    return ' | '.join(parts)


def grade_records(records: List[Tuple[str, float, float]]) -> Dict[str, Any]:
    """Grade a list of (name, measured, reference) tuples."""
    out = {'records': [], 'grades': [], 'distribution': None}
    for name, measured, reference in records:
        rel = relative_error(measured, reference)
        grade = compute_grade(measured, reference)
        out['records'].append({
                'name': name,
                'measured': measured,
                'reference': reference,
                'rel_err': rel,
                'grade': grade})
        out['grades'].append(grade)

    out['distribution'] = grade_distribution(out['grades'])
    out['summary'] = grade_summary(out['grades'])
    return out


def main():
    parser = argparse.ArgumentParser(
            description = 'Grade measured vs reference values')
    parser.add_argument('--json', type = str, required = True,
            help = 'JSON file with [{name, measured, reference}, ...]')
    parser.add_argument('--out', type = str, default = None,
            help = 'Optional output JSON for graded records')
    parser.add_argument('--max-bad', type = int, default = 0,
            help = 'Max BAD grades allowed (default 0)')
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        items = json.load(f)

    records = [(it['name'], float(it['measured']), float(it['reference']))
            for it in items]
    result = grade_records(records)

    print('[grade] summary: {}'.format(result['summary']))

    for rec in result['records']:
        print('  [{:8s}] {:30s} measured={:.6g} ref={:.6g} rel={:.3e}'.format(
                rec['grade'], rec['name'][:30], rec['measured'],
                rec['reference'], rec['rel_err']))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent = 2)
        print('[grade] wrote <{}>'.format(args.out))

    n_bad = result['distribution'].get('BAD', 0)
    if n_bad > args.max_bad:
        print('[grade] FAIL: {} BAD grades > max {}'.format(n_bad, args.max_bad))
        sys.exit(1)

    print('[grade] PASS')
    sys.exit(0)


if __name__ == '__main__':
    main()
