# pymnpbem_simulation — 사용 가이드

## CLI 진입점

```bash
python run_simulation.py [OPTIONS]
```

또는 모듈 형태:

```bash
python -m pymnpbem_simulation.cli [OPTIONS]
```

## 필수 옵션

| 옵션 | 의미 | 기본값 |
|---|---|---|
| `--config PATH` | YAML config 파일 경로 | (필수) |

## 병렬 옵션 (3-축 모델)

| 옵션 | 의미 | 기본값 |
|---|---|---|
| `--n-workers INT` | 동시 worker 프로세스 수 (wavelength 분배 단위) | 1 |
| `--n-threads INT` | 각 worker 안의 BLAS/OMP thread 수 | 1 |
| `--n-gpus-per-worker INT` | 각 worker 가 사용할 GPU 수 (0 = CPU, 1 = single GPU, 2+ = VRAM pool) | 0 |
| `--multi-node` | MPI multi-node 실행 (mpi4py 필요) | False |
| `--auto` | SLURM/PBS 환경에서 GPU/CPU 자동 감지 | False |

우선순위: CLI > YAML > `--auto` > 기본값.

## 기타 옵션

| 옵션 | 의미 |
|---|---|
| `--output-dir PATH` | 결과 저장 디렉토리 (YAML override) |
| `--simulation-name STR` | 시뮬레이션 이름 (output 폴더명) |
| `--n-wavelengths INT` | wavelength sub-sample 수 (debug 용) |
| `--reanalyze` | 시뮬 skip, postprocess 만 다시 실행 |
| `--verbose` | 상세 로그 출력 |
| `--help` | 도움말 |

## YAML config 구조

```yaml
structure:
  type: dimer_cube              # sphere | cube | rod | dimer_cube | ...
  edge: 47.0                    # nm
  gap: 0.6                      # nm
  n_per_edge: 24                # mesh density
  refine: 3
  e: 0.2                        # rounding fraction (tricube)

simulation:
  type: ret                     # ret | stat
  excitation: planewave         # planewave | dipole | eels
  enei_min: 500                 # nm
  enei_max: 1000
  n_wavelengths: 100
  polarizations: [[1,0,0], [0,1,0]]
  propagation_dirs: [[0,0,1], [0,0,1]]
  interp: curv

materials:
  medium: water
  particle: gold

compute:
  n_workers: 4
  n_threads: 1
  n_gpus_per_worker: 0
  multi_node: false
  hmode: dense

output:
  dir: ./results/dimer_baseline
  name: dimer_baseline
  formats: [npz, json, png]
  save_plots: true

postprocess:
  spectrum_xaxis: energy
  run_eigenmode_analysis: false
```

## 자동 감지 동작 (`--auto`)

- SLURM `--gres=gpu:N` → `SLURM_GPUS_ON_NODE=N` 사용
- PBS `-l gpus=N` → `PBS_GPUFILE` 사용
- `CUDA_VISIBLE_DEVICES` 도 fallback 으로 사용

휴리스틱:
- `G ≥ 1` 인 경우: `n_workers=G, n_gpus_per_worker=1, n_threads=C//G`
- `G == 0` 인 경우: `n_workers=C, n_threads=1`

## Migration: 기존 `.py` config → YAML

```bash
python -m pymnpbem_simulation.migration.py_to_yaml \
    /path/to/config_str.py \
    /path/to/config_sim.py \
    output.yaml
```

자세한 매핑은 [docs/CONFIG_MIGRATION.md](./docs/CONFIG_MIGRATION.md) 참고.

## 출력 구조

```
{output_dir}/{name}/
├── config.yaml                 # 사용된 config snapshot
├── spectrum.npz                # ext, sca, abs, wavelength
├── spectrum.json               # peak / FWHM 분석 결과
├── spectrum.png                # plot
├── fields.npz                  # E_total, E_induced 등 (calculate_fields=True 시)
├── surface_charge.npz          # surface charge
└── logs/
    └── pipeline.log
```

## 검증 baseline

`examples/dimer_baseline.yaml` 의 결과는 다음과 일치해야 한다 (machine precision 등급):

- `~/scratch/pymnpbem_sanity_test/lane_results/baseline_cpu.json` (60.10 min CPU, 100 wl)
- `~/scratch/pymnpbem_sanity_test/spectra_python_postfix_v4.txt`

Tolerance 등급:
- machine: `<1e-12`
- OK: `<1e-9`
- good: `<1e-6`
- warn: `<1e-3`
- BAD: `≥1e-3`
