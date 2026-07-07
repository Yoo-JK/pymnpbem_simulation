# Experiment ↔ Simulation Analysis — Handoff / Onboarding

새 세션(실험값 분석 프로젝트)이 첫날 이 문서만 읽으면 (1) 기존 계산결과 위치, (2) pymnpbem /
pymnpbem_simulation 개요, (3) post-analysis(위상차 분석 포함) 방법을 다 파악하도록 정리한 인수인계
문서. 세부 근거는 각 섹션이 가리키는 auto-memory 파일(§8)에 있다.

---

## 0. TL;DR
- **계산결과**: `~/research/pymnpbem/<family>/<case>/` — 각 케이스에 `spectrum.npz` + `sigma/`(표면전하·전류 캐시) + `config.yaml`.
- **툴**: `pymnpbem`(=`~/workspace/MNPBEM`, GPU BEM 코어) + `pymnpbem_simulation`(=`~/workspace/pymnpbem_simulation`, 래퍼).
- **분석 실행**: `python run_postprocess.py --anal-conf A.py --result <case>/spectrum.npz` (또는 `master.py`로 시뮬→분석 한번에).
- **실측 데이터**: `~/scratch/paper_figure_collection/raw/` (digitized 산란 스펙트럼 CSV).
- **env**: `/home/yoojk20/miniconda3/envs/mnpbem/bin/python` (MNPBEM+cupy 포함).

---

## 1. 계산 결과 저장 위치 (`~/research/pymnpbem/`)

| family | 구조 | 완료 case | 크기 |
|---|---|---|---|
| `au_dimer/` | Au dimer (nosub + sub/ITO) | 24 | 24 GB |
| `auag_dimer_4nm/` | Au@Ag dimer, **4nm Ag 셸** (r0.2/r0.3 × gap + rot15/30/45 회전) | 36 | 53 GB |
| `auagcl_dimer_4nm/` | Au@AgCl dimer (const 셸) | 4 | 2.1 GB |
| `monomer/au_r0.2` | Au 단일 큐브 | 1 | 0.4 GB |
| `auagagcl_dimer/`, `auagago_dimer/` | Au@Ag@AgCl / @AgO 3중셸 | **0 (미완)** | — |
| (미완) `auag_dimer_1nm` | Au@Ag **1nm 셸** | 0 (config만) | — |

**한 케이스 폴더가 담는 것** (예 `au_dimer/nosub/au_r0.2_g0.6/`):
- `spectrum.npz` — keys: `wavelength`(nm), `ext`, `sca`, `abs`; shape `(n_wl, n_pol)`. 편광 축 평균 = unpolarized.
- `sigma/` — 파장·편광별 `wl_{nm:07.2f}_p{pol}_d{dir}.npz` + `manifest.json`. **BEM 전체 해**(`sig1,sig2`=표면전하, `h1,h2`=표면전류) → §4의 재계산에 사용.
- `config.yaml` — 그 run이 실제로 쓴 resolved config (재현 기준). `run_metadata.json`, `postprocess/`, `structure.png`, (있으면) `field.npz`, `spectra_eV`.

**로드**: `np.load(case+'/spectrum.npz')`; sigma는 `pymnpbem_simulation.sigma_cache.load_sigma(case, wl_nm, pols, props)`.

**실측(digitized) 데이터** — `~/scratch/paper_figure_collection/raw/`:
- `digitized_energy_curve.csv` = Au monomer (Energy_eV, Intensity_norm), 1.45–2.60 eV
- `black_curve_redigitized.csv` = Au dimer+substrate r0.2 g0.6 (Energy_eV, Scattering), 1.39–2.66 eV
- 둘 다 산란, peak-normalize 로 비교.

---

## 2. `pymnpbem` — Python MNPBEM 포트 (`~/workspace/MNPBEM`)
- MATLAB MNPBEM → Python + GPU 로 포팅한 **BEM(경계요소법) 나노광학 솔버**. 입자 경계에서 Maxwell을
  풀어 표면전하 σ 를 구하고 → 소산/산란/흡수 단면적·근접장 계산.
- quasistatic(stat) / retarded(ret), 진공 / 기판(layered Green, Sommerfeld).
- GPU 가속(cupy) + fp32(complex64)/fp64. 큰 mesh는 multi-GPU VRAM-share.
- **주의(성능)**: 분산(VRAM-share) init 은 fp32 config 여도 내부 행렬이 complex128 로 새면 A6000의
  약한 fp64 때문에 LU가 ~13배 느려짐 → LOWPREC 시 c64 캐스트 필요 (bem_ret_layer 분산 경로).

## 3. `pymnpbem_simulation` — 래퍼 (`~/workspace/pymnpbem_simulation`)
config-driven end-to-end 파이프라인. 상세는 [README.md](../README.md) / [README.en.md](../README.en.md).
- **시뮬**: `python run_simulation.py --str-conf S.py --sim-conf M.py` (또는 `--config x.yaml`,
  또는 `--sweep-conf sweep.yaml` 다중). → `spectrum.npz` + sigma 캐시 저장(`simulation.save_sigma_cache`, 기본 on).
- **분석**: `python run_postprocess.py --anal-conf A.py --result <case>/spectrum.npz` (§4).
- **한번에**: `python master.py --str-conf S --sim-conf M --anal-conf A` → 시뮬→후처리 순차.
- 구조 빌더 12+ (sphere/dimer/core-shell/custom 셸/monomer/advanced_dimer_cube).
- excitation: **planewave/dipole/EELS × stat/ret/(sub)layer** 전부 구현됨(REGISTRY). 편광은 E-field 벡터
  `polarizations` + 진행방향 `propagation_dirs`. s/p(TE/TM)는 기판일 때 코어가 자동 분해(수직입사=degenerate).
- SLURM/PBS: `slurm_scripts/`, `pbs_scripts/`, `auto_detect.py`. yaml↔py 변환: `migration/`.

## 4. Post-analysis
`run_postprocess.py --analyzers ...` (analyzer 하이퍼파라미터는 `--anal-conf A.py`로 config화 가능, 예 `examples/fano_anal.py`):
- `spectrum` — ext/sca/abs plot + export(csv/json/npz), eV/nm 축.
- `fano` — 단일/다중 Lorentzian Fano fit.
- `fano-analysis` — bright/dark eigenmode + multi-Lorentzian (qs full-eig 기반).
- `eigenmode` / `multipole` — 고유모드 패턴, multipole 분해.
- 모듈: `postprocess/{spectrum,fano_fit,fano_analysis,mode_phase,plot_mode_phase,mode_compare,eigenmode,multipole,plot_surface_charge,field_analyzer}.py`.

**sigma 캐시 재계산 (BEM 재-solve 없이)** — `~/scratch/spectrum_from_cache.py`:
- `sigma/*.npz`(sig1,sig2,h1,h2) = BEM 전체 해 → `CompStruct(p, wl, sig1=, sig2=, h1=, h2=)` 복원.
- free-space `PlaneWaveRet(pol,prop)` / 기판 `PlaneWaveRetLayer(pol,prop,layer)` 로 `exc.extinction/scattering(sig)`.
- 검증: free-space 7.9e-5, layer 7.7e-10. **partial/field-only run 에서 spectrum 복구**할 때 씀. (greentab 불필요.)
- 같은 캐시로 near-field replay 가능(FieldCalculator field-only 경로).

## 5. Phase 차이 분석 (Fano 모드 위상) — [[project_fano_phase_analysis]]
케이스: `au_dimer/sub/au_r0.2_g0.6_sub` 의 1.43 / 1.8 eV feature.
- **규약-불변 모드 dipole `f_m = a_m·d_m`** 를 써야 함 (개별 eigenvector 위상은 임의). `a_m`=모달진폭(`u_L·σ`),
  `d_m`=모달 dipole(`Σ u_R[f]·x_f·A_f`). u_L·u_R 위상이 상쇄돼 규약 무관, `Σf_m` = 총 longitudinal dipole.
- qs 완전기저로 exact: `CompGreenStat(p,p).F` → `scipy.linalg.eig` → 캐시 `~/scratch/_qs_full_eig.npz`, `_dipole_spec.npz`.
- **★함정**: 고정 eigenbasis(reference 파장 ≈868nm/1.43eV에서 계산)는 **그 근처에서만 valid**. σ(1.8eV) 재구성
  R²=0.008(거의 직교) → 먼 파장 모달 위상분석 금지. (이걸 몰라서 잘못된 위상차를 한 번 냈던 전례 있음.)
- **결과**: 1.43/1.8 둘 다 **얕은 Fano** (narrow/broad 진폭비 ≈0.3, 완전 zero 아님). Δφ(narrow−background)≈**π/2
  = asymmetry 지배**, dip 최소서 0.6–0.71π (이상적 π 미달). 기저-무관 총 dipole 위상 arg(D(ω))는 두 dip 사이 ≈π/2 전진.
- 스크립트(`~/scratch/`): `rigorous_phase_qs.py`, `analyze_dipole.py`, `fano_fit_global.py`, `fano_sweep.py`, `render_true_dips.py`.

## 6. 실험 vs 시뮬 비교 노하우 — [[project_exp_sim_validation]]
- **Monomer 검증 통과**: exp peak 2.182 vs sim 2.166 eV (16 meV), r=0.957 → 방법론·monomer 모델 OK.
- **Dimer+기판(g0.6)**: lineshape는 정확하나 **+114~123 meV 계통 redshift**. no-shift r=0.43 → rigid +114meV 시 r=0.97.
- **원인**: 현재 sub 시뮬은 전부 **touching(입자-기판 gap 0.001nm)** → substrate coupling **과대평가** → 과도 redshift.
  실제는 입자가 살짝 떠있거나 ε<3.88(ITO). 물리 재현하려면 **substrate-distance(gap) sweep** 필요.
- **함정**: r0.3 g0.8 sub의 "peak match"는 Fano 쌍봉의 윗가지 → 가짜, 제외. gap best-fit g3.0도 축퇴 해.
- 비교 figure: `paper_figure_collection/compare_*.png`.

## 7. 논문 Figures — `~/scratch/paper_figures/{fig1,fig2,fig3}` (+ `FIGURES_README.md`)
- fig1=개념, fig2=검증+성능, fig3=Au dimer 예제(nosub/sub/monomer). fig3는 무거운 sigma 대신 `plotdata.npz`로 재생성. 상세 [[reference_paper_figures]].

## 8. 관련 auto-memory (세션 시작 시 자동 로드; 세부 근거)
- `project_exp_sim_validation` — 실측 vs 시뮬 (monomer r0.957, dimer+sub +120meV redshift)
- `project_fano_phase_analysis` — 위상차 분석 방법(f_m)·함정·결과
- `project_sigma_cache_recompute` — sigma 캐시로 관측량/필드 재계산
- `reference_paper_figures` — figure 스크립트·데이터·형식
- `project_auag_dimer_ops`, `project_auag_rotated_campaign`, `project_auagcl_sim`, `project_au_dimer_sim_plan` — 각 캠페인 운영 수치·경로

## 9. 새 세션에서 쓰는 법
1. 새 세션 시작 시 이 문서 경로를 알려주기: `~/workspace/pymnpbem_simulation/docs/EXPERIMENT_ANALYSIS_HANDOFF.md`.
2. auto-memory 는 같은 프로젝트 스코프면 자동 로드됨 — "실험 vs Au/Au@Ag dimer 시뮬 비교" 언급하면 §8 메모리를 참조함.
3. 분석 시작점: 비교할 실측 파장/편광 확인 → 해당 sim 케이스 `spectrum.npz` 로드 → peak-normalize·eV 변환 →
   dimer+기판이면 +0.11~0.12 eV offset 감안(§6).
