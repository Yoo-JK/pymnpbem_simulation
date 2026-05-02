# CLI Reference

`run_simulation.py` (or `python -m pymnpbem_simulation.cli`) 의 모든 옵션을 정리.

## Usage

```bash
python run_simulation.py --config CONFIG_YAML [OPTIONS]
```

## Options

### 필수

- `--config PATH`
  YAML config 파일 경로.

### 병렬 (3-축 모델)

- `--n-workers INT`
  동시 worker 프로세스 수. wavelength 분배 단위. 기본값: YAML 의 `compute.n_workers` 또는 1.

- `--n-threads INT`
  각 worker 안의 BLAS/OMP/Numba thread 수. CPU intensive 부분 가속.

- `--n-gpus-per-worker INT`
  - `0`: CPU only
  - `1`: 단일 GPU per worker (Lane D 패턴)
  - `2+`: VRAM pool (cuSolverMg/Magma — Lane E2 후속)

- `--multi-node`
  mpi4py 기반 multi-node MPI dispatch. 기본 OFF. (Wave 3 구현 예정)

- `--auto`
  SLURM/PBS 환경에서 GPU/CPU 자동 감지 후 plan 자동 산출.
  - `SLURM_GPUS_ON_NODE`, `SLURM_JOB_GPUS`, `PBS_GPUFILE`, `CUDA_VISIBLE_DEVICES` 우선순위로 감지

### Output

- `--output-dir PATH`
  결과 root 디렉토리 (YAML `output.dir` override).

- `--simulation-name STR`
  시뮬레이션 이름 (output 폴더명, YAML `output.name` override).

### Debug / Workflow

- `--n-wavelengths INT`
  wavelength sub-sample 수 (작은 회귀 테스트 용).

- `--reanalyze`
  시뮬 skip, 기존 `spectrum.npz` 만 다시 분석/플로팅.

- `--verbose`
  상세 로그.

## Exit codes

| Code | 의미 |
|---|---|
| 0 | 성공 |
| 1 | YAML 로딩 실패 |
| 2 | config validation 실패 |
| 3 | multi-node 미구현 (Wave 3) |
| 4 | reanalyze: spectrum.npz 없음 |

## 우선순위

CLI > YAML > `--auto` > 디폴트 (`compute = {1, 1, 0}`)

## 예시

### CPU 빠른 회귀 (10 wl)

```bash
python run_simulation.py \
    --config examples/dimer_baseline.yaml \
    --simulation-name dimer_quick \
    --n-wavelengths 10 \
    --n-workers 1 \
    --n-threads 4
```

### SLURM GPU auto

```bash
srun -N1 --gres=gpu:4 python run_simulation.py \
    --config examples/dimer_baseline.yaml \
    --auto
```

### 명시적 multi-GPU

```bash
python run_simulation.py \
    --config examples/dimer_baseline.yaml \
    --n-workers 4 \
    --n-gpus-per-worker 1
```

### Migration

```bash
python -m pymnpbem_simulation.migration.py_to_yaml \
    /path/to/config_str_dimer_g0.6.py \
    /path/to/config_sim_dimer_g0.6.py \
    examples/dimer_g0.6.yaml
```
