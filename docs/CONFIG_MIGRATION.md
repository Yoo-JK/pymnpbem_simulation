# `.py` config → YAML 마이그레이션

기존 `mnpbem_simulation` 의 두 `.py` config (`config_str_*.py` + `config_sim_*.py`) 를 YAML 한 파일로 자동 변환한다.

## 사용법

```bash
python -m pymnpbem_simulation.migration.py_to_yaml \
    /path/to/config_str_*.py \
    /path/to/config_sim_*.py \
    output.yaml
```

`structure` 또는 `simulation` 단독 변환:

```bash
python -m pymnpbem_simulation.migration.py_to_yaml \
    "" \
    /path/to/config_sim_*.py \
    output.yaml
```

## 키 매핑 표

| 기존 `.py` 키 | YAML section | YAML key |
|---|---|---|
| `structure` | `structure` | `type` |
| `structure_name`, `simulation_name` | `output` | `name` |
| `simulation_type` | `simulation` | `type` |
| `excitation_type` | `simulation` | `excitation` |
| `wavelength_range` | `simulation` | `wavelength_range` (+ `enei_min`, `enei_max`, `n_wavelengths` 자동 분해) |
| `polarizations` | `simulation` | `polarizations` |
| `propagation_dirs` | `simulation` | `propagation_dirs` |
| `dipole_position` / `dipole_moment` | `simulation` | 동일 |
| `impact_parameter` / `beam_energy` / `beam_width` | `simulation` | 동일 |
| `interp` | `simulation` | `interp` |
| `refine` | `structure` | `refine` |
| `relcutoff` | `simulation` | `relcutoff` |
| `waitbar` | `simulation` | `waitbar` |
| `use_parallel` | `compute` | `use_parallel` |
| `num_workers` | `compute` | `n_workers` (`'auto'` → -1, `'env'` 보존) |
| `max_comp_threads` | `compute` | `n_threads` (`'auto'` / `'max'` → -1) |
| `wavelength_chunk_size` | `compute` | `wavelength_chunk_size` |
| `use_mirror_symmetry` | `compute` | `mirror` |
| `use_iterative_solver` | `compute` | `iterative` |
| `use_nonlocality` | `compute` | `nonlocal` |
| `use_h2_compression` | `compute` | `hmode` (`bool` → `'aca-gpu'` / `'dense'`) |
| `medium` | `materials` | `medium` |
| `materials` (리스트) | `materials` | `particle_list` |
| `substrate` / `use_substrate` | `materials` | 동일 |
| `refractive_index_paths` | `materials` | 동일 |
| `diameter` / `size` / `gap` / `rounding` / `roundings` | `structure` | 동일 |
| `mesh_density` | `structure` | `mesh_density` |
| `core_size` / `shell_layers` | `structure` | 동일 |
| `offset` / `tilt_angle` / `tilt_axis` / `rotation_angle` | `structure` | 동일 |
| `n_spheres` / `shape_file` / `voxel_size` / `voxel_method` | `structure` | 동일 |
| `output_dir` | `output` | `dir` |
| `output_formats` | `output` | `formats` |
| `save_plots` / `plot_format` / `plot_dpi` | `output` | 동일 |
| `spectrum_xaxis` | `postprocess` | `spectrum_xaxis` |
| `calculate_cross_sections` / `calculate_fields` | `simulation` | 동일 |
| `field_region` / `field_mindist` / `field_nmax` / `field_wavelength_idx` | `simulation` | 동일 |
| `export_field_arrays` / `field_hotspot_count` / `field_hotspot_min_distance` | `simulation` | 동일 |
| `run_eigenmode_analysis` | `postprocess` | `run_eigenmode_analysis` |
| `eigenmode_n` / `eigenmode_top_k` | `postprocess` | 동일 |
| `retarded_eigen_wavelength` | `postprocess` | 동일 |
| `fano_target_wavelengths` / `svd_rank_threshold` | `postprocess` | 동일 |

## 폐기되는 키 (drop)

다음은 MATLAB 전용 옵션이라 Python wrapper에서 사용되지 않으므로 변환 시 제거된다.

- `mnpbem_path`
- `matlab_executable`
- `matlab_options`

## Unmapped 키

위 표에 없는 키는 YAML 의 `extras:` 섹션 아래에 그대로 보존된다. 이는 사용자가 재구성 후 검토할 수 있도록 정보 보존을 위함이다.

## 변환 예시

기존 `config_str_au_r0.2_g0.6.py`:

```python
args = {}
args['structure'] = 'advanced_dimer_cube'
args['core_size'] = 47
args['shell_layers'] = []
args['roundings'] = [0.2]
args['mesh_density'] = 2
args['gap'] = 0.6
args['materials'] = ['gold']
args['medium'] = 'water'
args['refractive_index_paths'] = {'agcl': {'type': 'constant', 'epsilon': 2.02}}
args['use_substrate'] = False
```

기존 `config_sim_au_r0.2_g0.6.py`:

```python
args = {}
args['use_parallel'] = True
args['num_workers'] = 4
args['max_comp_threads'] = 1
args['wavelength_chunk_size'] = 10
args['simulation_name'] = 'au_r0.2_g0.6'
args['simulation_type'] = 'ret'
args['interp'] = 'curv'
args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0], [0, 1, 0]]
args['propagation_dirs'] = [[0, 0, 1], [0, 0, 1]]
args['wavelength_range'] = [500, 1000, 100]
args['refine'] = 3
args['use_mirror_symmetry'] = False
args['use_iterative_solver'] = True
args['use_nonlocality'] = False
args['output_dir'] = '/home/yoojk20/research/mnpbem/dimer'
args['output_formats'] = ['txt', 'csv', 'json']
args['save_plots'] = True
args['run_eigenmode_analysis'] = True
```

변환 후 YAML:

```yaml
output:
  name: au_r0.2_g0.6
  dir: /home/yoojk20/research/mnpbem/dimer
  formats: [txt, csv, json]
  save_plots: true
structure:
  type: advanced_dimer_cube
  core_size: 47
  shell_layers: []
  roundings: [0.2]
  mesh_density: 2
  gap: 0.6
  refine: 3
materials:
  particle_list: [gold]
  medium: water
  refractive_index_paths:
    agcl: {type: constant, epsilon: 2.02}
  use_substrate: false
compute:
  use_parallel: true
  n_workers: 4
  n_threads: 1
  wavelength_chunk_size: 10
  mirror: false
  iterative: true
  nonlocal: false
simulation:
  type: ret
  interp: curv
  excitation: planewave
  polarizations: [[1, 0, 0], [0, 1, 0]]
  propagation_dirs: [[0, 0, 1], [0, 0, 1]]
  wavelength_range: [500, 1000, 100]
  enei_min: 500
  enei_max: 1000
  n_wavelengths: 100
postprocess:
  run_eigenmode_analysis: true
```

## 후속 단계 (수동 조정)

자동 변환 후, 사용자가 다음 항목을 재확인할 것을 권장:

1. **`compute.n_gpus_per_worker`** — 자동 변환 시 0 (CPU). GPU 사용 시 명시.
2. **`structure.type`** — Wave 1은 `dimer_cube`, `sphere` 만 지원. `advanced_dimer_cube` 등은 Wave 2 (M4) 에서 추가.
3. **`materials.particle`** — Wave 1 단일 particle 만 처리. multi-shell (`shell_layers`) 은 Wave 2.
4. **`output.formats`** — `npz` 추가 권장 (Python-native).

## 충돌 사항

- `structure_name` 과 `simulation_name` 둘 다 `output.name` 으로 매핑.
  → 마이그레이션 코드는 `merge_args()` 에서 `args_str` 다음에 `args_sim` 을 update 하므로 sim 우선.

- `args['materials']` (리스트) vs YAML `materials.medium`/`particle` 단일.
  → `materials.particle_list` 로 보존하되, Wave 2 에서 multi-particle 처리 시 사용 예정.
