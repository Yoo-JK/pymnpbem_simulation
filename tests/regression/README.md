# pymnpbem_simulation Regression Suite (M10)

위치: `tests/regression/`
연관 문서: [`docs/REGRESSION_ACCEPTANCE.md`](../../docs/REGRESSION_ACCEPTANCE.md)
Wave 3 M10 산출물.

## 구조

```
tests/regression/
├── __init__.py
├── README.md                       # 이 파일
├── conftest.py                     # markers + fixtures + grade helper
├── test_baseline_dimer.py          # Wave 1 baseline (dimer 6336 face)
├── test_structures.py              # 14 구조 빌드 + smoke (M4)
├── test_excitation.py              # 6 excitation type (M5)
├── test_field.py                   # field calculation (M3)
├── test_substrate.py               # substrate (M6)
├── test_postprocess.py             # eigenmode + Fano + multipole (M8)
├── test_dispatch.py                # CPU/GPU/multi-node (M2/M9)
├── data/
│   └── reference_results.json      # baseline reference
└── runners/
    ├── __init__.py
    ├── compute_grade.py            # 등급 산정 (machine/OK/good/warn/BAD)
    └── run_full_regression.py      # 전체 회귀 자동 실행 + 등급 리포트
```

## pytest markers

| marker | 의미 | 예상 wall | 실행 빈도 |
|---|---|---|---|
| `fast` | < 1 분 — 빌드/import/analyzer | ~30 s | 매 commit |
| `slow` | 5 ~ 30 분 — CLI smoke 회귀 | ~20 분 | daily nightly |
| `long` | > 30 분 — full spectrum | ~1 시간+ | weekly |
| `gpu` | CUDA + cupy 필요 | - | self-hosted runner |
| `multinode` | SLURM/MPI 필요 | - | cluster runner |

## 로컬 실행

```bash
PYBIN=/home/yoojk20/miniconda3/envs/mnpbem/bin/python

# fast subset (commit 마다)
$PYBIN -m pytest tests/regression/ -m fast --tb=short -q

# slow subset (daily)
$PYBIN -m pytest tests/regression/ -m slow --tb=short -v

# fast + slow 통합
$PYBIN tests/regression/runners/run_full_regression.py --markers "fast or slow"

# JSON summary 출력
$PYBIN tests/regression/runners/run_full_regression.py \
    --markers "fast or slow" --json artifacts/regression_summary.json
```

## 등급 정의

| 등급 | 표기 | 정의 (rel err) | 의미 |
|---|---|---|---|
| machine precision | machine | `< 1e-12` | bit-수준 일치 |
| OK | OK | `< 1e-9` | 수치적으로 동등 |
| good | good | `< 1e-6` | 시각적으로 동등 |
| warn | warn | `< 1e-3` | 추적 권장 |
| BAD | BAD | `>= 1e-3` | 회귀 실패 (블로커) |

## reference 데이터

`data/reference_results.json` 의 출처:
- Wave 2 merge smoke 7종 결과 (`/tmp/pymnpbem_wave2_merge_report.md` 참고)
- `/home/yoojk20/scratch/pymnpbem_sanity_test/lane_results/baseline_cpu.json`
- Wave 1 통합 (main HEAD `9cac8ba`)

reference 갱신은 동일 환경에서 `run_full_regression.py` 재실행 후 수동 update.

## 환경

- conda env: `mnpbem` (Python 3.11)
- shell PYBIN: `/home/yoojk20/miniconda3/envs/mnpbem/bin/python`

## CI 연동

`.github/workflows/regression.yml` (선택)
- fast: PR 마다
- slow: nightly cron
- long: weekly
