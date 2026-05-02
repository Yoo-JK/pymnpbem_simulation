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

```bash
# Single-node CPU
python run_simulation.py --config examples/dimer_baseline.yaml --n-workers 4 --n-threads 1

# Auto-detect (SLURM/PBS GPU 환경)
python run_simulation.py --config examples/dimer_baseline.yaml --auto

# Migrate old .py config to YAML
python -m pymnpbem_simulation.migration.py_to_yaml \
    /path/to/config_str.py /path/to/config_sim.py output.yaml
```

자세한 CLI 옵션은 [HELP.md](./HELP.md) 또는 `python run_simulation.py --help` 참조.

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
