# jk-config 마이그레이션 인벤토리

## 출처

- mnpbem_simulation `jk-config` 브랜치, `config/` 디렉토리
- `_prev_proj/` 제외 active 페어: 156 str + 160 sim → 페어링 후 160 yaml 산출 (1 sim-only).

## 카테고리별 분류

| 디렉토리 | yaml 개수 | structure type | simulation type | 비고 |
|---|---|---|---|---|
| agg_sph_1nm_wo_sub | 7 | sphere_cluster_aggregate | ret/planewave | 1nm overlap, water+gold |
| agg_sph_t1..t4_wo_sub | 7×4 | sphere_cluster_aggregate | ret/planewave | tilt/series variant |
| agg_sph_wo_sub | 7 | sphere_cluster_aggregate | ret/planewave | base series |
| dimer_auag_1nm_r0.2 | 8 | connected_dimer_cube | ret/planewave (some w/ sub) | Au-Ag 1nm cores |
| dimer_auag_4nm_r0.2 | 8 | connected_dimer_cube | ret/planewave | Au-Ag 4nm cores |
| dimer_auagcl_r0.2 | 1 | connected_dimer_cube | ret/planewave | Au-AgCl |
| dimer_au_r0.2 | 16 | connected_dimer_cube | ret/planewave | gap=0.0..5.0nm × wo/w substrate |
| dimer_au_r0.3 | 16 | connected_dimer_cube | ret/planewave | rounding=0.3 variant |
| dimer_monomer_r0.2 | 3 | advanced_monomer_cube? | ret/planewave | au, ag, auag baselines |
| excitonic_wo_tmd | 15 | cube/rod/sphere | ret/planewave | nm-size series, gold+water |
| excitonic_w_tmd | 1 | cube + substrate | ret_layer/planewave | gold cube on gold substrate |
| ho_rod | 33 | core_shell_rod | ret/planewave | 22x47..53x106 + polymer.dat 사용 |
| mat2py_rod | 4 | rod | ret/stat × wo/w substrate | 22x47nm AuNR |
| mat2py_sphere | 4 | sphere | ret/stat × wo/w substrate | 30nm Au sphere |
| rod_aucu | 3 | rod (custom material) | ret/planewave | au, auau, au90cu10 |
| rod_ctab_rod_1nm | 2 | core_shell_rod | ret/planewave | au/auctab variants |
| rod_ctab_rod_contact | 2 | (same series) | ret/planewave | substrate 접촉 |
| rod_ctab_rod_nosub | 2 | (same series) | ret/planewave | no substrate |

## 자동 변환 vs 수동 적응

자동 변환 (py_to_yaml.py): 키 매핑만 함. 수동 적응이 필수:

- `materials.medium` 이 dict (`{type: constant, epsilon: X}`) → preset 이름 또는 string-of-eps 로 변환
- `materials.particle_list` → `particle` (single) 또는 `core`/`shell` (core_shell_*)
- `use_substrate=True` + `substrate.{material, position}` → `with_substrate` 빌더 wrap + `simulation.type=ret_layer`
- 사용자 정의 material (`polymer`, `agcl`, `ito`, `ctab`, `cu` 등) → `refractive_index_paths` 의 `.dat` 경로로 해석. `.txt` 파일이나 unsupported 이름은 silver 로 fallback (사용자가 README 참조 후 수정 권유).
- `compute` 블록에 `n_gpus_per_worker`, `multi_node`, `hmode` 기본값 보충
- `output.dir` 을 `./results` 로 통일 (실행 시 `--output-dir` 로 override 가능)
- structure-specific 보정: cube 류는 `mesh_density` (nm) → `n_per_edge` 환산, sphere 류는 작은 값 (nm) → 144 default

## 변환되지 않은 파일

- `config/ho_rod/config_sim_53x106_20_8.py` 페어 부재 (config_str_53x106_20_8.py 가 없음). sim-only 변환됨.

## 알려진 한계

1. **Multi-shell core_shell_***: pymnpbem builder 가 [core, shell] 2-layer 만 지원. `[gold, polymer]` 같은 페어는 OK 지만 `[gold, silver, agcl]` 같은 3-layer 는 첫 두 개만 사용됨.
2. **Custom material refractive index**: builder 의 `_build_eps_particle` 이 'gold'/'silver' presets 와 `.dat` 파일만 지원. `.txt` 파일은 silver 로 자동 fallback. 사용자가 .dat 로 변환하거나 builder 를 확장해야 정확.
3. **Substrate 가 metal (`gold` 등)**: `_build_eps_substrate` 가 `gold` 라는 string 을 preset 으로 모름. 자동으로 `gold.dat` 로 매핑 시도 (`_resolve_substrate_eps`).
4. **field-only mode**: `calculate_cross_sections=False` + `calculate_fields=True` 인 dimer_au 류는 자동변환 시 type=`ret` 그대로 (pymnpbem 의 'field' runner 는 별도). 사용자가 `simulation.type=field` 로 변경 후 grid 블록 명시해야 정확.
