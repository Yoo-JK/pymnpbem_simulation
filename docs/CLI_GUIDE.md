# CLI Guide — `--str-conf` / `--sim-conf` 패턴

이 문서는 `pymnpbem_simulation` 의 새 CLI 사용법과 기존 YAML CLI 와의
관계를 정리한다. 새 패턴은 `mnpbem_simulation` (MATLAB wrapper) 의
인터페이스와 동일한 형태로, structure 정의와 simulation 정의를 두 개의
`.py` config 로 분리한다.

## 1. 두 가지 모드

### Mode A — 새 패턴 (권장, mnpbem_simulation 호환)

```bash
python run_simulation.py \
    --str-conf path/to/<name>_str.py \
    --sim-conf path/to/<name>_sim.py \
    --verbose
```

- `--str-conf <path.py>` : structure 정의 (`structure_type`, dimensions,
  materials, mesh density 등)
- `--sim-conf <path.py>` : simulation + compute + output 정의
  (`simulation_type`, `wavelength_range`, `polarizations`,
  `compute = {n_workers, n_threads, n_gpus_per_worker, ...}`,
  `output = {dir, name}` 등)
- `--verbose` : 로딩한 `str_conf`, `sim_conf`, merged cfg 를 모두 JSON
  으로 dump.

### Mode B — Legacy YAML (backward-compat)

```bash
python run_simulation.py --config path/to/cfg.yaml
```

기존 jk-config 의 `auag_r0.2_g0.6.yaml` 같은 YAML 들이 그대로 동작한다.
새 모드를 사용하지 않을 때만 `--config` 가 필요하다.

## 2. `.py` config 형식

`.py` config 파일은 **`args = {...}`** 단 한 개의 dict 만 정의해야 한다.
실행 시 `exec()` 으로 로딩되며 `args` 가 없거나 dict 가 아니면 실패한다.

### `str_conf` 예시 (`examples/auag_dimer_str.py`)

```python
args = {
    'structure': 'advanced_dimer_cube',
    'core_size': 47,
    'shell_layers': [4],
    'roundings': [0.2, 0.2],
    'mesh_density': 2,
    'gap': 0.6,
    'offset': [0, 0, 0],
    'tilt_angle': 0,
    'tilt_axis': [1, 0, 0],
    'rotation_angle': 0,
    'refine': 3,
    'materials': ['gold', 'silver'],
    'medium': 'water',
    'use_substrate': False,
    'refractive_index_paths': {
        'agcl': {'type': 'constant', 'epsilon': 2.02}}}
```

### `sim_conf` 예시 (`examples/auag_dimer_sim.py`)

```python
args = {
    'simulation_type': 'ret',
    'excitation_type': 'planewave',
    'wavelength_range': [300, 1000, 140],
    'polarizations': [[1, 0, 0], [0, 1, 0]],
    'propagation_dirs': [[0, 0, 1], [0, 0, 1]],
    'interp': 'curv',
    'relcutoff': 3,
    'calculate_cross_sections': True,
    'calculate_fields': False,

    'compute': {
        'use_parallel': True,
        'n_workers': 5,
        'n_threads': 1,
        'wavelength_chunk_size': 10,
        'iterative': True,
        'n_gpus_per_worker': 0,
        'multi_node': False,
        'hmode': 'dense'},

    'output': {
        'dir': './results',
        'name': 'auag_r0.2_g0.6',
        'formats': ['json', 'npz', 'png'],
        'save_plots': True}}
```

## 3. CLI override 우선순위

```
CLI flag  >  sim_conf nested compute/output  >  default
```

자주 쓰는 override:

| Flag                       | 효과                                                      |
| -------------------------- | --------------------------------------------------------- |
| `--n-workers N`            | `compute.n_workers` 덮어쓰기                              |
| `--n-threads N`            | `compute.n_threads` 덮어쓰기                              |
| `--n-gpus-per-worker N`    | `compute.n_gpus_per_worker` 덮어쓰기                      |
| `--vram-share-backend X`   | `cusolvermg` / `magma` / `nccl` (n-gpus > 1 일 때만 의미) |
| `--multi-node`             | `compute.multi_node = True`                               |
| `--auto`                   | SLURM/PBS env 에서 compute plan 자동 감지                 |
| `--simulation-name X`      | `output.name` 덮어쓰기 (run folder)                       |
| `--output-dir DIR`         | `output.dir` 덮어쓰기                                     |
| `--n-wavelengths N`        | wavelength sub-sample (디버깅용)                          |
| `--reanalyze`              | 시뮬 skip, postprocess 만 재실행                          |
| `--verbose`                | str_conf / sim_conf / merged cfg 출력                     |

## 4. 변환 도구

### Legacy `.py` (mnpbem_simulation 형식) → YAML

```bash
python -m pymnpbem_simulation.migration.py_to_yaml \
    legacy_str.py legacy_sim.py output.yaml
```

### YAML → `--str-conf` / `--sim-conf` `.py` 쌍

```bash
python -m pymnpbem_simulation.migration.yaml_to_str_sim \
    input.yaml out_str.py out_sim.py
```

이 도구로 기존 jk-config YAML 들을 `.py` 로 다시 분할하여 새 CLI 패턴
으로 사용할 수 있다.

## 5. 실행 예시

### v1.5.2 권장 setting (4 GPU VRAM share)

`auag_dimer_sim.py` 의 `compute` 블록:

```python
'compute': {
    'n_workers': 1,
    'n_threads': 4,
    'n_gpus_per_worker': 4,
    'vram_share_backend': 'cusolvermg',
    'iterative': True}
```

또는 CLI 로 override:

```bash
python run_simulation.py \
    --str-conf examples/auag_dimer_str.py \
    --sim-conf examples/auag_dimer_sim.py \
    --n-workers 1 --n-threads 4 \
    --n-gpus-per-worker 4 \
    --vram-share-backend cusolvermg \
    --verbose
```

### Multi-node SLURM

```bash
srun python run_simulation.py \
    --str-conf my_str.py --sim-conf my_sim.py \
    --multi-node --auto
```

### 빠른 디버깅 (3 wavelengths only)

```bash
python run_simulation.py \
    --str-conf examples/sphere_str.py \
    --sim-conf examples/sphere_sim.py \
    --n-wavelengths 3 --simulation-name sphere_smoke
```

## 6. 키 매핑 요약

`.py` config 의 flat key 들이 내부 cfg dict 의 어느 section 으로 가는지
는 `pymnpbem_simulation.migration.py_to_yaml._KEY_TO_SECTION` 에 정의
되어 있다. 주요 매핑:

| `.py` flat key        | 내부 cfg section            |
| --------------------- | --------------------------- |
| `structure`           | `structure.type`            |
| `core_size`, `gap`, ...| `structure.<...>`          |
| `materials`           | `materials.particle_list`   |
| `medium`              | `materials.medium`          |
| `simulation_type`     | `simulation.type`           |
| `excitation_type`     | `simulation.excitation`     |
| `wavelength_range`    | `simulation.wavelength_range` |
| `polarizations`       | `simulation.polarizations`  |
| `num_workers`         | `compute.n_workers`         |
| `output_dir`          | `output.dir`                |
| `simulation_name`     | `output.name`               |

`sim_conf.py` 안에서 `compute = {...}`, `output = {...}` 같은 nested
dict 로 직접 작성해도 우선 적용된다 (mnpbem_simulation 의 flat 형식과
새로운 nested 형식 모두 지원).
