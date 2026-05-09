# Feature Parity Audit: mnpbem_simulation (MATLAB) vs pymnpbem_simulation

mnpbem_simulation 의 MATLAB-based wrapper 와 pymnpbem_simulation (pure Python port)
간 기능 비교 표.

상태:
- `OK` — pymnpbem 에 동등 / 더 나은 구현 존재
- `partial` — 일부 기능만 존재, 보강 필요
- `TODO` — 누락 (구현 후보)
- `skip` — MATLAB-only 의존 (port 불가) 또는 의도적 미포함

우선순위:
- `H` (high) — Au dimer / Au@Ag dimer / agg sphere / rod 시뮬에 직접 영향
- `M` (medium) — 일반 plasmonic 시뮬에 자주 사용
- `L` (low) — 특수 / matlab-specific

## 1. Postprocess — Visualization

| 기능 | mnpbem (MATLAB wrapper) | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| Spectrum plot (per polarization) | visualizer.py:plot_spectrum | postprocess/plot.py:plot_spectrum | partial | H |
| Spectrum xaxis = energy (eV) toggle | plot_spectrum (xaxis_unit) | (없음) | TODO | H |
| Polarization comparison (ext/sca/abs 3 plots) | plot_polarization_comparison | (없음) | TODO | H |
| Unpolarized spectrum (FDTD-style 2-pol average) | plot_unpolarized_spectrum + SpectrumAnalyzer.calculate | (없음) | TODO | H |
| Comparison plot (pol vs unpolarized) | _plot_spectrum_comparison_with_unpolarized | (없음) | TODO | H |
| All-in-one comparison (3 subplots) | comparison_all_unpolarized | (없음) | TODO | M |
| Field plots (enhancement, 2D slice) | plot_fields, _plot_field_enhancement | plot_field.py:plot_field_2d_slice | partial | H |
| Field intensity plots | _plot_field_intensity | (없음) | TODO | M |
| Field vector plots (2D quiver) | _plot_field_vectors | (없음) | TODO | M |
| Field separate internal/external | plot_field_separate_internal_external | (없음) | TODO | M |
| Field comparison (overlay all pols) | _plot_field_comparison | (없음) | TODO | M |
| Field overlay (all pols on single plot) | _plot_field_overlay | (없음) | TODO | L |
| Unpolarized fields | plot_unpolarized_fields | (없음) | TODO | M |
| Material boundary draw on field plot | _draw_material_boundary | (없음) | TODO | M |
| Surface charge 3D plot | _plot_surface_charge_3d | plot_surface_charge.py:plot_surface_charge_3d | OK | - |
| Surface charge 2D 8-views | _plot_surface_charge_2d_8views | plot_surface_charge_2d_8views | OK | - |
| Surface charge phase analysis (Re/Im/abs/arg) | plot_surface_charge_phase_analysis | plot_surface_charge_phase | OK | - |
| Hotspots 3D | (없음 — pyMNPBEM 측 별도) | plot_hotspots_3d | OK | - |
| Near-field decay | (없음) | plot_near_field_decay | OK | - |
| Eigenmode plots (mode patterns grid) | _plot_mode_patterns_grid | (없음) | TODO | M |
| Eigenmode magnitude spectra | _plot_magnitude_spectra | (없음) | TODO | M |
| Eigenmode phase spectra | _plot_phase_spectra | (없음) | TODO | M |
| Eigenmode delta phi comparison | _plot_delta_phi_comparison | (없음) | TODO | L |
| Multipole character table | _plot_multipole_character_table | (없음) | TODO | M |
| Fano fit plot | _plot_fano_fit | (없음 — fit 만 존재) | TODO | M |
| Cross-validation summary | _plot_cross_validation_summary | (없음) | TODO | L |

## 2. Postprocess — Analysis

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| Peak finder (scipy.signal.find_peaks) | SpectrumAnalyzer._find_peaks | spectrum.py:argmax only | partial | H |
| FWHM 계산 | _calculate_fwhm | spectrum.py:_compute_fwhm | OK | - |
| Multi-peak detection | _find_peaks (prominence) | (없음) | TODO | M |
| Enhancement factors (pol1/pol2 ratio) | _calculate_enhancement | (없음) | TODO | M |
| Avg/max cross sections | analyze (avg_*, max_*) | (없음) | TODO | M |
| Unpolarized check (orthogonal pols) | _check_unpolarized_conditions | (없음) | TODO | H |
| Unpolarized spectrum compute | _calculate_unpolarized_spectrum | (없음) | TODO | H |
| Hotspot finder | FieldAnalyzer._find_hotspots | field_analyzer.py:hotspot_location | partial | H |
| High-field region analysis | _analyze_high_field_regions | (없음) | TODO | M |
| Near-field integration | calculate_near_field_integration | (없음) | TODO | M |
| Near-field decay | (없음) | field_analyzer.py:near_field_decay | OK | - |
| Field statistics | _calculate_statistics | (없음) | TODO | M |
| Edge artifact detection | edge_filter.py:find_edge_artifacts | (없음) | TODO | L |
| Geometry cross-section | geometry_cross_section.py:GeometryCrossSection | (없음) | TODO | M |
| QS eigenmode analysis | QSEigenAnalyzer (eigenmode_analyzer.py) | postprocess/eigenmode.py:qs_eigenmodes | OK | - |
| SVD decomposition | SVDAnalyzer | eigenmode.py:svd_decomposition | OK | - |
| Multipole projection | MultipoleAnalyzer | postprocess/multipole.py | OK | - |
| Retarded eigenmode | RetardedEigenAnalyzer | eigenmode.py:retarded_eigenmodes | OK | - |
| Mode comparator (cross-validate) | ModeComparator | (없음) | TODO | L |
| Fano fit | FanoFitter | fano_fit.py:fano_fit | OK | - |
| Multi-peak Fano | (없음) | multi_fano_fit | OK | - |
| Core-shell separator | CoreShellSeparator | (없음) | TODO | H (Au@Ag) |

## 3. Simulation Runners

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| planewave + ret | matlab template | planewave_ret.py | OK | - |
| planewave + stat | matlab | planewave_stat.py | OK | - |
| planewave + ret + layer (substrate) | matlab | planewave_ret_layer.py | OK | - |
| planewave + ret + iter | matlab | planewave_ret_iter.py | OK | - |
| planewave + stat + iter | matlab | planewave_stat_iter.py | OK | - |
| planewave + ret + layer + iter | matlab | planewave_ret_layer_iter.py | OK | - |
| planewave + ret + mirror | matlab | planewave_ret_mirror.py | OK | - |
| planewave + stat + layer | matlab | (없음) | TODO | L |
| dipole + ret | matlab | dipole_ret.py | OK | - |
| dipole + stat | matlab | dipole_stat.py | OK | - |
| dipole + ret + layer | matlab | dipole_ret_layer.py | OK | - |
| dipole + ret + iter | matlab | (없음) | TODO | L |
| dipole + stat + iter | matlab | (없음) | TODO | L |
| eels + ret | matlab | eels_ret.py | OK | - |
| eels + stat | matlab | eels_stat.py | OK | - |
| eels + ret + layer | matlab | eels_ret_layer.py | OK | - |
| nonlocal eps | matlab | (with_nonlocal wrapper) | partial | M |
| Field calculation grid | matlab | field_calculator.py + grid_builder.py | OK | - |

## 4. Structures (Geometry Builders)

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| sphere | matlab trisphere | sphere.py | OK | - |
| cube | matlab tricube | cube.py | OK | - |
| rod | matlab trirod | rod.py | OK | - |
| ellipsoid | matlab triellipsoid | ellipsoid.py | OK | - |
| triangle | matlab tritriangle | triangle.py | OK | - |
| dimer (sphere/cube) | matlab | dimer_sphere.py + dimer_cube.py | OK | - |
| core_shell sphere | matlab | core_shell_sphere.py | OK | - |
| core_shell cube | matlab | core_shell_cube.py | OK | - |
| core_shell rod | matlab | core_shell_rod.py | OK | - |
| dimer_core_shell_cube (Au@Ag dimer) | matlab | dimer_core_shell_cube.py | OK | - |
| advanced_monomer_cube | matlab | advanced_monomer_cube.py | OK | - |
| advanced_dimer_cube | matlab | advanced_dimer_cube.py | OK | - |
| connected_dimer_cube | matlab | connected_dimer_cube.py | OK | - |
| sphere_cluster (aggregate) | matlab | sphere_cluster.py | OK | - |
| from_shape (.mat / .stl import) | matlab geometry_generator | from_shape.py | OK | - |
| with_substrate | matlab | with_substrate.py | OK | - |
| with_mirror | matlab | with_mirror.py | OK | - |
| with_nonlocal | matlab | with_nonlocal.py | OK | - |
| Cylinder rod (별도 builder) | matlab | rod.py (cylinder shape supported) | OK | - |

## 5. Materials

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| Drude / Lorentz / EpsConst | matlab | (mnpbem core 직접 호출) | OK | - |
| Table-based eps (Johnson&Christy 등) | RefractiveIndexLoader | (mnpbem core 직접 호출) | OK | - |
| AgCl, dielectric coatings | matlab | (mnpbem core) | OK | - |
| Nonlocal hydrodynamic Drude | NonlocalGenerator | nonlocal_eps.py:make_hydrodynamic_drude_eps | OK | - |
| Material auto-detection (metal/dielectric) | material_manager:_is_metal | (없음) | TODO | L |

## 6. CLI / Dispatch / Orchestration

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| Single-node dispatch | matlab | dispatch/single_node.py | OK | - |
| Multi-GPU per worker | matlab | dispatch/multi_gpu.py | OK | - |
| Multi-node MPI | matlab | dispatch/mpi_node.py | OK | - |
| Sweep launcher (4 worker per-GPU) | (없음) | (sweep launcher) | OK | - |
| --reanalyze (postprocess only) | run_postprocess.py | cli.py --reanalyze | OK | - |
| --auto compute plan | matlab | cli.py --auto | OK | - |
| --verbose | matlab | cli.py --verbose | OK | - |
| --n-wavelengths sub-sample | (없음) | cli.py | OK | - |
| SLURM scripts | matlab | slurm_scripts/ | OK | - |
| PBS scripts | matlab | pbs_scripts/ | OK | - |
| Config snapshot save | matlab | cli.py:save_yaml | OK | - |
| Run metadata save | matlab | save_run_metadata | OK | - |
| Config py->yaml migration | (matlab .py format) | migration/py_to_yaml.py | OK | - |

## 7. Output Formats

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| .npz spectrum | (없음) | io/writer.py | OK | - |
| .json spectrum_analysis | data_exporter (json) | postprocess (json) | OK | - |
| .csv spectrum | data_exporter (csv) | postprocess/export.py | OK | - |
| .txt spectrum (header + data) | data_exporter._save_txt | (없음) | TODO | M |
| .txt field data (per pol/wl) | DataExporter._export_single_field | (없음) | TODO | M |
| .png plot | matlab + visualizer | plot.py 등 | OK | - |
| .pdf plot | visualizer (plot_format=['png','pdf']) | (PNG only by default) | TODO | M |
| .h5 export | (없음) | export.py:export_h5 | OK | - |
| .mat export | matlab native | (없음) | skip | - |
| .eps / .svg | (visualizer 옵션) | (없음) | TODO | L |

## 8. Postprocess Specific Analyses (이미 구현)

| 기능 | mnpbem | pymnpbem | 상태 | 우선순위 |
|---|---|---|---|---|
| QS eigenmodes (Boundary integral 방식) | eigenmode_analyzer.QSEigenAnalyzer | postprocess.eigenmode | OK | - |
| Retarded eigenmodes | retarded_eigen.RetardedEigenAnalyzer | postprocess.eigenmode.retarded_eigenmodes | OK | - |
| SVD rank determination | SVDAnalyzer.determine_rank | (없음 in pymnpbem) | TODO | L |
| Mode classification (dipole/quad/etc) | MultipoleAnalyzer.classify | (없음) | TODO | M |
| Mode comparator (cross-validation) | ModeComparator | (없음) | TODO | L |

---

# 우선순위 요약

## High priority (필수, 사용자 케이스 직접 영향) — Series A/B/C

- **Series A (visualization)**: Spectrum xaxis (energy), polarization comparison, unpolarized spectrum + comparison plot
- **Series B (analysis)**: Multi-peak detection, enhancement factors, unpolarized check
- **Series C (Au@Ag)**: Core-shell separator (core/shell mask + cutaway plot)

## Medium priority (일반 plasmonic 자주 사용) — Series D/E/F

- **Series D (visualization)**: field intensity, field comparison, mode patterns grid, multipole character table, fano fit plot
- **Series E (output)**: txt spectrum/field exporter, pdf plot
- **Series F (analysis)**: high-field region analysis, near-field integration, geometry cross-section, fano fit plot, multi-peak detection improvements

## Low priority — Series G

- planewave_stat_layer runner, dipole iterative variants
- mode comparator, SVD rank, mode classification
- eps / svg plots

---

# 시리즈 실행 계획

| 시리즈 | 대상 | 설명 | 파일 |
|---|---|---|---|
| A1 | postprocess/plot.py | spectrum xaxis=energy 지원 | plot.py |
| A2 | postprocess/plot.py | polarization comparison plot | plot.py |
| A3 | postprocess/spectrum.py | unpolarized 계산 + check | spectrum.py + new |
| A4 | postprocess/plot.py | unpolarized + comparison plots | plot.py |
| B1 | postprocess/spectrum.py | multi-peak detection (scipy find_peaks) | spectrum.py |
| B2 | postprocess/spectrum.py | enhancement factors / avg / max | spectrum.py |
| C1 | postprocess/core_shell.py | CoreShellSeparator port (cube/rod) | new |
| D1 | postprocess/plot_field.py | field intensity (없는 경우) + comparison | plot_field.py |
| D2 | postprocess/plot_eigenmode.py | mode patterns / magnitude / phase | new |
| D3 | postprocess/plot.py | multipole character bar chart + fano fit plot | plot.py |
| E1 | postprocess/export.py | txt spectrum / field exporter | export.py |
| F1 | postprocess/field_analyzer.py | high-field region analysis | field_analyzer.py |
| F2 | postprocess/geometry_cross_section.py | geometry cross-section util | new |
