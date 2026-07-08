# pymnpbem_simulation

Python wrapper for MNPBEM (Metal Nanoparticle Boundary Element Method) simulations.

> English README: [README.en.md](./README.en.md).

기존 MATLAB-기반 `mnpbem_simulation` wrapper 를 폐기하고 Python MNPBEM port (`/home/yoojk20/workspace/MNPBEM`) 를 직접 호출하도록 재작성한 것이다.

핵심 변경:
- MATLAB 코드 생성 단계 제거 → Python 함수 직접 호출
- 데이터 포맷: Python-native (`.npz` / `.h5`), `.mat` 미사용
- Config: YAML + CLI override (argparse 기반)
- 3-축 병렬 (`n_workers × n_threads × n_gpus_per_worker`)
- SLURM/PBS 자동 감지

## Requirements

```bash
conda create -n pymnpbem_sim python=3.11
conda activate pymnpbem_sim

# Python MNPBEM port (editable, in sibling repo)
pip install -e /home/yoojk20/workspace/MNPBEM

# Core scientific stack
pip install numpy==2.0.2 scipy==1.14.1 matplotlib==3.10.7 pandas==2.3.3

# Config / CLI
pip install pyyaml==6.0.2 python-box==7.3.2 tqdm==4.67.1

# I/O
pip install h5py==3.12.1

# Notebook
pip install jupyter==1.1.1

# GPU acceleration (optional)
pip install cupy-cuda12x==13.3.0

# Multi-node MPI (optional)
pip install mpi4py==4.0.1

# Postprocess analysis
pip install scikit-learn==1.7.2 lmfit==1.3.2 plotly==6.5.0
```

## Quick Start

### 권장 패턴: `--str-conf` / `--sim-conf` (mnpbem_simulation 호환)

```bash
# 새 패턴: structure 와 simulation 정의를 .py 두 개로 분리
python run_simulation.py \
    --str-conf examples/auag_dimer_str.py \
    --sim-conf examples/auag_dimer_sim.py \
    --verbose

# 빠른 검증 (3 wavelengths)
python run_simulation.py \
    --str-conf examples/sphere_str.py \
    --sim-conf examples/sphere_sim.py \
    --n-wavelengths 3
```

`sim_conf.py` 안에 `compute = {n_workers, n_threads, n_gpus_per_worker, ...}`,
`output = {dir, name, ...}` 같은 nested dict 로 모든 compute 파라미터를
함께 지정한다. 자세한 사용법은 [docs/CLI_GUIDE.md](./docs/CLI_GUIDE.md) 참조.

### Sweep 패턴: `--sweep-conf <yaml>` (병렬 다중 케이스)

여러 (str_conf, sim_conf) 페어를 동시에 돌려 각 worker 가 자기 GPU 에 pin 된다 (`CUDA_VISIBLE_DEVICES` 격리). 4-GPU 노드에서 4 케이스 비교 = 4x throughput.

```bash
python run_simulation.py --sweep-conf my_sweep.yaml
```

자세한 sweep YAML 포맷은 [HELP.md](./HELP.md#sweep-mode---sweep-conf) 참조.

### Legacy 패턴: `--config <yaml>` (backward-compat)

```bash
# Single-node CPU
python run_simulation.py --config examples/dimer_baseline.yaml --n-workers 4 --n-threads 1

# Auto-detect (SLURM/PBS GPU 환경)
python run_simulation.py --config examples/dimer_baseline.yaml --auto
```

### 변환 도구

```bash
# Legacy mnpbem_simulation .py → YAML
python -m pymnpbem_simulation.migration.py_to_yaml \
    /path/to/config_str.py /path/to/config_sim.py output.yaml

# YAML → --str-conf/--sim-conf .py 쌍
python -m pymnpbem_simulation.migration.yaml_to_str_sim \
    input.yaml out_str.py out_sim.py
```

### Postprocess 분석 config: `--anal-conf` (재현 가능한 분석)

시뮬레이션이 `--str-conf`/`--sim-conf` 로 config-driven 이듯, 후처리 분석
(Fano 위상차·eigenmode·multipole·spectrum)도 `--anal-conf <.py>` 로 하이퍼파라미터를
config 에 담아 재현 가능하게 돌린다. 우선순위: **명시적 CLI 플래그 > `--anal-conf` > 기본값**.

```bash
# 분석 하이퍼파라미터를 config 로 (examples/fano_anal.py 참고)
python run_postprocess.py \
    --anal-conf examples/fano_anal.py \
    --result /path/to/case/spectrum.npz          # result 는 config 에 넣어도 됨

# CLI 로 개별 override (config 값보다 우선)
python run_postprocess.py --anal-conf examples/fano_anal.py \
    --result .../spectrum.npz --analyzers spectrum,fano-analysis --xaxis energy
```

`anal-conf` .py 는 `args = {...}` dict 로 argparse dest 키(`analyzers`, `fano_features`,
`fano_pol`, `n_modes`, `max_l`, `export_formats`, `xaxis`, `eig_cache`, `case_dir`,
`result`, `output` 등)를 담는다. 콤마-문자열 옵션은 Python list 로도, `polarizations`
는 중첩 list 로도 쓸 수 있다.

### 전체 파이프라인 한번에: `master.py` (시뮬 → 분석)

**넘긴 config 로 동작이 자동 결정된다** (`--skip-*` 플래그 없음): `--str-conf`+`--sim-conf`(세트)면
시뮬, `--anal-conf` 가 있으면 분석. 분석은 특정 후처리(위상차 등) 할 때만 붙이고 대부분은 str+sim 만.

```bash
# 시뮬만  (str + sim 은 세트)
python master.py --str-conf S.py --sim-conf M.py --verbose

# 시뮬 + 분석/저장  (--anal-conf 추가; 특정 후처리 할 때만)
python master.py --str-conf S.py --sim-conf M.py --anal-conf A.py

# 계산 안 하고 후처리만  (str 없이; sim-conf 가 결과 위치를 알려줌)
python master.py --sim-conf M.py --anal-conf A.py

# sweep(다중); --anal-conf 있으면 각 케이스 분석
python master.py --sweep-conf sweep.yaml [--anal-conf A.py]
```

`--str-conf` 는 `--sim-conf` 와 세트다(str 단독 불가). sigma(표면전하 σ / 표면전류) 캐시는 시뮬 중
자동 저장(`simulation.save_sigma_cache`, 기본 `true`)되어 후처리가 BEM 재-solve 없이 재사용한다.
sim/postprocess 로 그대로 넘길 추가 플래그는 `--sim-extra "..."` / `--anal-extra "..."`.

자세한 CLI 옵션은 [docs/CLI_GUIDE.md](./docs/CLI_GUIDE.md), [HELP.md](./HELP.md)
또는 `python run_simulation.py --help` 참조.

## Project layout

```
pymnpbem_simulation/
├── pymnpbem_simulation/    # 메인 패키지
│   ├── cli.py              # CLI entry
│   ├── config.py           # YAML 로더
│   ├── auto_detect.py      # SLURM/PBS GPU 감지
│   ├── env_setup.py        # MNPBEM_GPU 등 환경변수
│   ├── util.py             # 공통 유틸
│   ├── structures/         # 12+ 구조 빌더
│   ├── simulation/         # 시뮬 모드 (planewave/dipole/eels × stat/ret)
│   ├── postprocess/        # numpy 직접 처리
│   ├── dispatch/           # CPU/GPU/multi-node 분배
│   ├── io/                 # .npz / .h5 출력
│   └── migration/          # .py config → YAML 변환
├── tests/                  # pytest 회귀 테스트
├── examples/               # 예제 YAML config
└── docs/                   # 디자인 문서
```

## Status

프로덕션 사용 중. Python MNPBEM 포트(`/home/yoojk20/workspace/MNPBEM`)를 직접 호출하는
end-to-end 파이프라인이 안정 동작하며, Au/Au@Ag/core-shell dimer sweep 등 대규모 캠페인에 쓰이고 있다.

- **시뮬레이션 모드**: planewave / dipole / EELS × quasistatic(stat) / retarded(ret), 진공 및 기판(layered Green/Sommerfeld)
- **구조 빌더 12+**: sphere, dimer, core-shell, custom 유전체 셸(`refractive_index_paths`), monomer, advanced_dimer_cube(rounded-edge) 등
- **3-축 병렬** (`n_workers × n_threads × n_gpus_per_worker`) + SLURM/PBS 자동 감지 + GPU pin 격리 sweep
- **GPU 가속**(cupy) + **multi-GPU VRAM-share**(cuSolverMg 분산 dense LU) — 단일 GPU VRAM(48GB) 초과 mesh 지원
- **sigma 캐시**: 표면전하(σ) 덤프/재로드로 BEM 재-solve 없이 스펙트럼·필드·관측량 재계산 + spectrum sweep RESUME
- **postprocess**: Fano 분석(qs full-eig 기반 bright/dark + dipole 위상 + Lorentzian fit), 표면전하 시각화, eigenmode 분석
- **검증**: MATLAB MNPBEM 대비 72-demo 회귀 (max rel err ~10⁻³·⁹, median ~10⁻¹³·⁵)

## License

MIT (TBD).
