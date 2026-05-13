# jk-config: 사용자 시뮬 config 모음

`mnpbem_simulation` (MATLAB wrapper) 의 `jk-config` 브랜치에서 사용하던 사용자 config (`.py`) 를
`pymnpbem_simulation` 의 YAML schema 로 마이그레이션한 결과이다.

## 빠른 사용법

```bash
# pymnpbem_simulation 루트에서 실행
PYBIN=/home/yoojk20/miniconda3/envs/mnpbem/bin/python

# 예: dimer (gap=0nm, no substrate) 짧은 smoke
${PYBIN} run_simulation.py \
    --config config/jk/dimer_au_r0.2/au_r0.2_g0.0.yaml \
    --simulation-name au_r0.2_g0.0_smoke \
    --n-wavelengths 5 \
    --n-workers 1 \
    --output-dir ./results

# 예: agg_sph 5-sphere cluster (정상 100파장)
${PYBIN} run_simulation.py \
    --config config/jk/agg_sph_1nm_wo_sub/5_agg.yaml \
    --simulation-name agg_5sph_1nm \
    --n-workers 4 \
    --output-dir ./results
```

`--n-wavelengths` 옵션으로 wavelength 개수 sub-sample 하여 빠른 검증 가능.
`--output-dir` / `--simulation-name` 으로 결과 위치 override.

## 카테고리

| 디렉토리 | 개수 | 설명 |
|---|---|---|
| `agg_sph_1nm_wo_sub` | 7 | Au sphere cluster aggregate (1-7 spheres), 1nm overlap, 기판 없음 |
| `agg_sph_t1_wo_sub` ~ `t4_wo_sub` | 7×4 | sphere cluster series (tilt/variant 1-4) |
| `agg_sph_wo_sub` | 7 | sphere cluster base series |
| `dimer_au_r0.2` | 16 | connected_dimer_cube (Au, rounding=0.2), gap 0..5nm × wo/w substrate |
| `dimer_au_r0.3` | 16 | rounding=0.3 variant |
| `dimer_auag_1nm_r0.2` | 8 | Au-Ag dimer (cores 1nm 차이) |
| `dimer_auag_4nm_r0.2` | 8 | Au-Ag dimer (cores 4nm 차이) |
| `dimer_auagcl_r0.2` | 1 | Au-AgCl dimer (constant epsilon AgCl) |
| `dimer_monomer_r0.2` | 3 | advanced_monomer_cube baseline (au, ag, auag) |
| `excitonic_wo_tmd` | 15 | cube/rod/sphere series (1-5nm), excitonic 연구용 baseline |
| `excitonic_w_tmd` | 1 | cube on metal substrate (TMD layered) |
| `ho_rod` | 33 | core_shell_rod (gold core + polymer/.dat shell), 22x47 ~ 53x106 series |
| `mat2py_rod` | 4 | rod ret/stat × wo/w substrate |
| `mat2py_sphere` | 4 | sphere ret/stat × wo/w substrate |
| `rod_aucu` | 3 | rod (au, auau, au90cu10) |
| `rod_ctab_rod_1nm` ~ `_contact` ~ `_nosub` | 2×3 | core_shell_rod with CTAB shell variants |

## 결과 위치

`output.dir` 은 모든 yaml 에서 `./results` 로 통일되어 있다 (실행 시점에서 `--output-dir` 로 override 가능).
원본 mnpbem_simulation 에서는 `~/research/mnpbem/...` 등 사용자 절대경로였으나 portability 를 위해 변경.

## 자동 변환 + 수동 적응 내역

원본 `.py` config (str + sim 페어) 를 `pymnpbem_simulation.migration.py_to_yaml` 로 자동 변환 후
다음 schema 적응을 수동으로 적용했다:

1. **`materials.medium`** dict (`{type: constant, epsilon: X}`) → preset 이름 또는 numeric string
   (sphere builder 의 `_build_eps_medium` 가 `EpsConst(float(name))` 처리 가능).
2. **`materials.particle_list`** → `particle` (single) 또는 `core` / `shell` (core_shell_*).
3. **`use_substrate=True`** → 원본 `structure` 를 `with_substrate.base` 로 wrap +
   `simulation.type=ret/stat` → `ret_layer`. substrate eps 는 사용자 `refractive_index_paths`
   에서 우선 조회하고, preset 이름이면 그대로, metal 이름이면 `gold.dat` / `silver.dat` 로 fallback.
4. **`compute`** 블록에 `n_gpus_per_worker=0`, `multi_node=False`, `hmode=dense` 기본값 보충.
5. **structure-specific 보정**:
   - cube 류: `mesh_density` (element size nm) → `n_per_edge` 환산
   - sphere 류: 작은 nm 값은 `n_verts=144` default 로 대체
   - rounding 0 (sharp) 은 tricube ZeroDivisionError 회피로 `0.01` 로 클램프
6. **`output`** dir 통일 + `formats` 에 `npz/json/png` 강제 추가.

## 알려진 한계 (실행 시 주의)

### Multi-shell core_shell 구조

- `materials = ['gold', 'silver', 'agcl']` 같은 3-layer 는 pymnpbem builder 가 첫 두 개만 사용 (core, shell).
  `advanced_dimer_cube` 류는 `shell_layers` + `roundings` 길이 매칭으로 동작하지만 epsilon 은 첫 particle 만 적용됨.
  → 원본이 multi-shell 인 경우 사용자가 직접 검토 필요.

### Custom material 경로

- `refractive_index_paths` 가 `.dat` 인 경우만 builder 가 직접 읽음. `.txt` 등은 자동변환 시 `silver` 로 fallback.
  → `ho_rod/*.yaml` 의 `polymer` 는 `polymer.txt` 가 dataset 에 있어 자동 fallback 발동.
  → 정확한 결과를 원하면 `.dat` 형식으로 변환 후 yaml 의 `materials.shell` 경로 수정.

### Metal substrate (예: `gold` 기판)

- `excitonic_w_tmd/cube.yaml` 은 `substrate.eps=gold.dat` 인데, mnpbem 의 layer green function
  이 metal substrate 에 대해 `IndexError` 를 던지는 버그가 관찰됨.
  → 사용자가 dielectric 기판 (예: glass=2.25, silica=1.45^2) 으로 변경하거나 pymnpbem 측 fix 필요.

### Field-only mode

- `dimer_au_r0.2/au_r0.2_g*.yaml` 등 일부는 원본에서 `calculate_cross_sections=False`,
  `calculate_fields=True` 로 field-only 시뮬이었다. 자동 변환은 `simulation.type=ret` 그대로 유지.
  pymnpbem 의 'field' runner 사용을 원하면 `simulation.type=field` 로 변경 + `grid` 블록 명시 필요.
  현재 yaml 은 `ret/planewave` cross-section spectrum 으로 실행됨.

### 사용자 dataset 경로

- `refractive_index_paths` 가 `/home/yoojk20/dataset/mnpbem/refrac/...` 절대경로를 사용하는 경우,
  해당 파일이 호스트에 없으면 빌드 실패.
  → `rod_ctab_*` 시리즈가 이에 해당. 사용자 환경에 dataset 디렉토리가 있어야 함.

## 검증 결과

- **schema validation: 160 / 160 통과** (모든 yaml 이 pymnpbem `validate_config` 통과)
- **smoke run** (n_wavelengths=2-4, 대표 카테고리 23 yaml):
  - 빠르게 완료된 OK: 15 — agg_sph (1nm/wo_sub/t1-t4 series), dimer_monomer, excitonic_wo_tmd
    (sphere/rod 1-5nm), mat2py_sphere/rod (ret+stat × wo/w substrate), rod_aucu/au
  - 무거운 mesh 라 smoke timeout: 6 — `dimer_au_r0.2/au_r0.2_g3.0`, `dimer_auag_*`, `dimer_auagcl`,
    `excitonic_wo_tmd/cube_3nm` 등 (정식 실행 시 더 큰 wavelength 갯수 + 병렬 필요)
  - 환경 의존 fail: 2 — `rod_ctab_rod_contact/au`, `rod_ctab_rod_nosub/au`
    (`gold_olmon.dat` 사용자 dataset 파일 부재)
  - pymnpbem layer-metal bug: 1 — `excitonic_w_tmd/cube.yaml` (gold substrate 사용)

## 원본과의 차이점

| 항목 | 원본 (.py) | jk-config (.yaml) |
|---|---|---|
| `output.dir` | `~/research/mnpbem/<name>` | `./results` (override 가능) |
| `n_workers` | 1-8 사용자 지정 | 그대로 보존 (override 가능) |
| `n_threads` | `max_comp_threads=64` 등 | 보존 |
| `mnpbem_path` | MATLAB toolbox 경로 | 폐기 (pymnpbem 사용) |
| MATLAB 옵션 (`matlab_executable`, `matlab_options`) | 폐기 |
| `output_formats` | `[txt, csv]` 등 | `[npz, json, png]` 으로 강제 추가 |
| `simulation.type` (sub) | `ret` + `use_substrate=True` | `ret_layer` + `with_substrate` wrap |
| `materials` | list `[gold, silver]` | builder 별 single string (`particle` / `core` / `shell`) |
| `medium = {type:constant,epsilon:1.459}` | dict | numeric string `'1.459'` (또는 preset) |
