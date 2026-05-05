# pymnpbem_simulation

Python wrapper for MNPBEM (Metal Nanoparticle Boundary Element Method) simulations.

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

- Phase 1 (cleanup + 분석): 완료 (2026-05-02)
- Phase 2 Wave 1 (foundation): 진행 중 — skeleton + dimer baseline
- Phase 2 Wave 2-4 (feature 확장 + 회귀): 계획됨

## License

MIT (TBD).
