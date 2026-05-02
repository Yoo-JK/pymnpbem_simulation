# pymnpbem_simulation Regression Acceptance Criteria

생성일: 2026-05-02 (Wave 3 M10)
대상: pymnpbem_simulation v1.0
근거: Wave 1+2 smoke 7종 + lane_results/baseline_cpu.json

이 문서는 회귀 스위트 (`tests/regression/`) 가 자동으로 판정하는 기준을 정의한다.

---

## 1. 등급 정의

| 등급 | 표기 | 정의 (max relative error) | 의미 |
|---|---|---|---|
| machine precision | `machine` | `< 1e-12` | bit-수준 일치 (FP 누적 한계) |
| OK | `OK` | `< 1e-9` | 수치적으로 동등 |
| good | `good` | `< 1e-6` | 시각적으로 동등 |
| warn | `warn` | `< 1e-3` | 추적 권장 |
| BAD | `BAD` | `>= 1e-3` | 회귀 실패 (블로커) |

판정 근거는 `tests/regression/runners/compute_grade.py` 의 `compute_grade()`.

---

## 2. 정확도 기준

### 2.1 회귀 등급 분포

| 메트릭 | 요구 기준 |
|---|---|
| BAD 개수 | **= 0** (필수) |
| machine precision 비율 | ≥ 80% (smoke 항목 한정 — analyzer/builder/import) |
| warn 합계 | ≤ 1 / N (CLI smoke 한정) |

CLI smoke 항목은 BEM solver 의 ULP 차이로 보통 `warn` 또는 `good` 등급 (rel err ~ 1e-7 ~ 1e-4).
analyzer/builder/import 항목은 deterministic 한 코드 경로이므로 `machine` 등급.

### 2.2 dimer baseline 정확도 (6336 face × 2 wl, Wave 1 회귀)

| 메트릭 | 현재 실측값 | 요구 기준 |
|---|---|---|
| peak ext_x @ 500 nm | 8744.331 | rel err ≤ 1e-3 vs reference |
| n_faces | 6336 | exact match |

근거: `data/reference_results.json` 의 `dimer_baseline_2wl`.

### 2.3 structure / excitation / substrate / postprocess / field

각 모듈은 fast 항목 (analyzer/builder) 와 slow 항목 (CLI smoke) 으로 구성:

- fast 항목: deterministic, machine grade 요구
- slow 항목: CLI 결과의 peak 값 reference 와 ≤ 1e-3 rel diff (BAD 회피)

---

## 3. 속도 기준

`run_full_regression.py --markers fast` 기준:

| 메트릭 | 요구 기준 |
|---|---|
| fast subset 전체 wall | < 60 s |
| slow subset 전체 wall | < 30 min (CPU, n_threads = 4) |

세부:

| 테스트 | 예상 wall (CPU) | 비고 |
|---|---|---|
| test_structures (14 빌드) | < 5 s | fast |
| test_postprocess (5 항목) | < 10 s | fast |
| test_field (analyzer + grid) | < 5 s | fast |
| test_dispatch (smoke) | < 1 s | fast |
| test_dimer_baseline_2wl | ~5 min | slow |
| test_dimer_baseline_cpu_pool | ~5 min | slow |
| test_cube_cli_smoke | ~1 min | slow |
| test_*_excitation (5종) | ~2 min × 5 | slow |
| test_sphere_substrate_smoke | ~3 min | slow |
| test_field_calculator_dimer | ~3 min | slow |

GPU/multinode 항목은 self-hosted runner 한정.

---

## 4. 환경

- Python 3.11
- conda env `mnpbem`
- 주요 의존성: numpy, scipy, h5py, pytest
- 선택: cupy (GPU marker), srun (multinode marker)

---

## 5. 실행 방법

```bash
PYBIN=/home/yoojk20/miniconda3/envs/mnpbem/bin/python

# 매 commit (PR)
$PYBIN -m pytest tests/regression/ -m fast --tb=short -q

# nightly
$PYBIN -m pytest tests/regression/ -m slow --tb=short -v

# weekly (long, full spectrum)
$PYBIN -m pytest tests/regression/ -m long --tb=short -v

# 통합 runner (등급 분포 리포트 포함)
$PYBIN tests/regression/runners/run_full_regression.py \
    --markers "fast or slow" \
    --json artifacts/regression_summary.json
```

`run_full_regression.py` 의 exit code:
- 0: PASS (BAD = 0, pytest rc = 0)
- 1: FAIL (BAD > 0 또는 pytest 실패)

---

## 6. CI 권장 사항

`.github/workflows/regression.yml`:
- `on: pull_request` → fast marker only
- `on: schedule (nightly)` → fast + slow
- `on: schedule (weekly)` → full incl. long

Self-hosted runner (GPU + SLURM) 가 있다면 `gpu`/`multinode` marker 도 함께.

---

## 7. 알려진 한계

- Wave 2 smoke 5 (`dimer_field_1wl`) 는 PartialFallback (CLI grid 분기 미연결).
  Wave 3 의 다른 agent (Agent A — M3 hotfix) 가 처리. 회귀 suite 에서는 단독 import + analyzer 만 검증.
- multinode marker 는 SLURM 환경 부재 시 자동 skip.
- GPU marker 는 cupy 부재 시 자동 skip.
