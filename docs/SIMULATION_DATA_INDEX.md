# 시뮬레이션 데이터 인덱스

지금까지 수행한 모든 시뮬레이션 결과의 위치·상태·크기 목록.
실측 기준일: **2026-07-30**.

데이터는 물리적으로 4개 루트에 흩어져 있다. 이 문서가 단일 진입점이다.

| 루트 | 크기 | 성격 |
|---|---|---|
| `~/research/pymnpbem/` | 87G | **본진.** dimer/monomer 계열 PyMNPBEM 결과 |
| `~/scratch/rod_pymnpbem/` | 122G | **rod 계열 본진** (별도 트리) |
| `~/research/adda/` | 129G | ADDA (DDA) 시뮬 — PyMNPBEM 아님 |
| `~/scratch/*.zip` | 69G | 과거 export 아카이브 (중복) |

## 0. 통합 아카이브 — `~/research/SIMULATION_ARCHIVE/` (127G)

위 루트들을 하나의 트리로 모은 것. **PyMNPBEM 결과를 다룰 때는 여기부터 본다.**

```
SIMULATION_ARCHIVE/
├── README.md      아카이브 자체 설명 (케이스 규약·복구 이력)
├── INDEX.csv      103 케이스 기계판독 목록
├── cube/          dimer·monomer 7 family   (81G, 하드링크)
├── rod/           results + mat + fano     (46G, 실복사)
├── legacy_matlab/ MATLAB MNPBEM 레퍼런스   (364M, 하드링크)
└── misc/          흩어져 있던 소규모 결과  (497M, 실복사)
```

`INDEX.csv` 컬럼: `group, family, case, n_wavelength, n_expected, status,
n_sigma, has_field, spectrum_source, size_mb`.
집계: **complete 91 / partial 4 / 데이터없음 8**.

- `cube/`·`legacy_matlab/` 은 **하드링크**라 원본과 inode 를 공유한다.
  아카이브에서 파일을 고치면 원본도 바뀐다. 한쪽만 지우는 것은 안전하다.
- `rod/`·`misc/` 는 원본이 `~/scratch`(sda3), 아카이브는 `~/research`(nvme) 로
  다른 장치라 하드링크가 불가능해 실제 복사본이다.
- 제외: `~/research/adda/` (129G, 다른 코드), `rod_pymnpbem/txt/` (47G,
  `mat/` 에서 재생성 가능), `pymnpbem/bug_backup/` (6G).

배포용 zip: `~/research/SIMULATION_ARCHIVE.zip` (store 압축, 약 127G).

---

## 1. `~/research/pymnpbem/` — dimer / monomer 본진 (87G)

케이스 디렉터리 규약: `<family>/<case>/` 안에
`spectrum.npz`(완료 표식) · `sigma/`(BEM 전체 해 캐시) · `config.yaml` · `postprocess/`.

`spectrum.npz` 가 있으면 완료, 없으면 중단/부분.

| family | 크기 | 완료 | 부분 | 비고 |
|---|---|---|---|---|
| `monomer/` | 403M | 1 | 0 | `au_r0.2`, 100 파장 그리드 |
| `au_dimer/nosub/` | 6.5G | **16** | 0 | 8건은 sigma 캐시에서 복구 |
| `au_dimer/sub/` | 17G | 16 | 0 | 8개는 필드/표면전하 PNG ~5950장 (≈2G/케이스) |
| `auag_dimer_4nm/` | 53G | 38 | 4 | 부분은 전부 r0.3_sub 계열 |
| `auagcl_dimer_4nm/` | 2.1G | 4 | 1 | const g0.6/0.8/1/5 완료. **sell 변종은 미실행(32K 빈 스텁)** |
| `auagagcl_dimer/` | 775M | 1 | 2 | **g0.6 완료.** g0.8/g1 은 12K 빈 스텁 |
| `auagago_dimer/` | 775M | 1 | 0 | **g0.6 완료** |

### 스펙트럼 복구 (2026-07-30)

중단 케이스 12건의 `spectrum.npz` 를 `sigma/` 캐시에서 재계산했다.
BEM 을 다시 풀지 않고 `PlaneWaveRet.extinction/scattering(sig)` 만 호출하는
경로다. 완료 케이스(`au_dimer/nosub/au_r0.3_g0.0`)로 검증했을 때 원본 대비
**최대 상대오차 2e-15** — 머신 정밀도.

- `au_dimer/nosub` 8건: 100/100 파장 전부 복구 → family 16/16 완료
- `auag_dimer_4nm` 4건: 파장 부분 커버리지로 복구

출처는 `run_metadata.json` 의 `spectrum_source` 로 구분한다
(`reconstructed_from_sigma_cache` 또는 `partial_backup_20260608_211902`).
부분 케이스는 `spectrum_partial: true` + `spectrum_n_wavelengths` 가 함께 기록된다.

### 남은 부분·미실행 케이스

전부 `auag_dimer_4nm/auag_r0.3_*_sub`:

| 케이스 | 파장 | sigma | 상태 |
|---|---|---|---|
| `auag_r0.3_g1.0_sub` | 101 / 140 | 16 (+백업) | 부분 복구 |
| `auag_r0.3_g0.4_sub` | 55 / 140 | 110 | 부분 복구 |
| `auag_r0.3_g0.6_sub` | 45 / 140 | 12 (+백업) | 부분 복구 |
| `auag_r0.3_g50.0_sub` | 9 / 140 | 18 | 부분 복구 |
| `auag_r0.3_g0.6_rot15/rot30/rot45_sub` | 0 | 0 / 0 / 4 | 미착수 (회전 캠페인) |
| `auag_r0.3_g30.0_sub` | 0 | 0 | 미착수 |

`_partial_backup_20260608_211902` 안에 6/8 시점의 sigma 가 남아 있다.
재개 시 이 백업을 먼저 확인할 것.

### 같은 트리 안의 노이즈

- `bug_backup/` (6.0G, spectrum.npz 70개) — 버그 있던 구 실행본. **사용 금지**
- `auag_dimer_1nm/` (4K) — 빈 디렉터리
- `_dist*` / `_memtest*` / `_smoke*` / `_iter_test` / `_plain*` 약 20개 (12K–17M) — 일회성

---

## 2. `~/scratch/rod_pymnpbem/` — rod 계열 (122G)

가장 큰 단일 production 트리. 5x10 / 10x20 / 15x30 rod, gap g0.0–g0.6, Au@Ag core-shell.

| 하위 | 크기 | 내용 |
|---|---|---|
| `txt/` | 48G | .mat → 텍스트 변환본 |
| `mat/` | 27G | `simulation_results.mat` (MATLAB 호환 출력) |
| `results/` | 20G | `spectrum.npz`, field 데이터 (원본) |
| `txt_test/` | 3.2G | **테스트 잔여물 — 삭제 가능** |
| `auag_g0.0_fano/` | 24M | Fano 분석 |
| `audit_txt/`, `audit_mat/` | 124K | 전수조사 스크립트·로그 |

완료 표식 파일: `PIPELINE_DONE`, `MAT_DONE`, `TXT_DONE`.
케이스별 yaml config 동봉.

> 과거 존재하던 `mat_OLD_nm_order_DO_NOT_USE` / `mat_BUGGY_198wl_DO_NOT_USE` 는
> 이미 삭제됨. 남은 정리 대상은 `txt_test/` 뿐.

---

## 3. 흩어진 production 결과 (합계 ≈ 0.5G)

| 경로 | 크기 | 내용 |
|---|---|---|
| `~/scratch/au_g0.6_fp64_sigma/` | 400M | Au g0.6 **FP64 레퍼런스** sigma 캐시 + manifest.json |
| `~/scratch/au_dimer_charge/` | 42M | Au dimer 표면전하 분포 |
| `~/scratch/paper_figures/` | 32M | 논문 figure 스크립트·데이터 |
| `~/scratch/paper_figure_collection/` | 15M | figure 원본 + `raw/` 실측 데이터 |
| `~/scratch/comparison_rod_lying/` | 3.8M | rod 눕힌 배치 비교 |
| `~/scratch/auag_gap_sweep_3/` | 1.7M | Au@Ag gap sweep (후속) |
| `~/scratch/auag_gap_sweep/` | 1.6M | Au@Ag gap sweep |
| `~/scratch/comparison_rod/` | 1.2M | rod 비교 |
| `~/scratch/auag_per_gap/` | 1.1M | Au@Ag per-gap |
| `~/scratch/auag_stacked/` | 692K | 적층 배치 변종 |

### MATLAB 레거시 (PyMNPBEM 이전 레퍼런스)

| 경로 | 크기 | 내용 |
|---|---|---|
| `~/research/mnpbem/` | 350M | ho_rods 27쌍 + dimer/excitonic (.mat) |
| `~/workspace/mnpbem_simulation/` | 14M | codegen 워크플로우 산출물 |

---

## 4. 검증 / 벤치 / 테스트 (합계 ≈ 0.6G)

| 경로 | 크기 | 목적 |
|---|---|---|
| `~/scratch/au_g0.6_fp32_verify/` | 400M | FP32 vs FP64 정밀도 검증 |
| `~/scratch/sub_compare/` | 79M | 기판 유/무 비교 |
| `~/scratch/auagcl_bugtest/` | 42M | Au@AgCl 셸 버그 재현 |
| `~/scratch/auag_g0.2_verify_fp32,_fp64,_refine/` | 26M ×3 | fp32/fp64 + mesh refine |
| `~/scratch/landes_e2e/` | 11M | landes config E2E |
| `~/scratch/cli_md8_test/` | 5.6M | CLI 테스트 |
| `~/scratch/pymnpbem_wave2_m*, _wave3_m10/` | 5.4M | 개발 마일스톤 |
| `~/scratch/au_dimer_compare/` | 2.6M | 비교 케이스 |
| `~/scratch/dimer_bench/` | 2.0M | 벤치마크 |
| `~/scratch/v151_issueA_fix/` | 1.8M | solver 이슈 검증 |
| `~/scratch/resume_test/` | 1.2M | resume 기능 |
| `~/scratch/sub_verify/` | 124K | 기판 검증 |
| `~/workspace/pymnpbem_simulation/results/` | 33M | repo 내 smoke 실행본 |

---

## 5. 아카이브 / stale — 정리 후보 (합계 ≈ 76G)

| 경로 | 크기 | 판정 |
|---|---|---|
| `~/scratch/pymnpbem_dimer_results_part{1,2,3}.zip` | 46.6G | 본진 export 백업. 중복 |
| `~/scratch/eigenmode_analysis_r0.2_g0.6_1.0_5.0_sub.zip` | 16.4G | 고유모드 분석 export |
| `~/scratch/pymnpbem_results_nosigma.zip` | 5.4G | sigma 제거 export |
| `~/scratch/eigenmode_ret_composite_auag_r0.2_g1.0_g5.0.zip` | 3.6G | retarded 고유모드 export |
| `~/research/pymnpbem_backup_fp64/` | 3.9G | 통합 이전 구 FP64 dimer 실행본 |
| `~/scratch/pymnpbem_sub_r0.{2,3}_nosigma.zip` | 1.9G | 기판 케이스 export |
| `~/scratch/rod_pymnpbem/txt_test/` | 3.2G | 테스트 잔여물 |
| `~/research/pymnpbem/bug_backup/` | 6.0G | 버그 실행본 |
| `~/scratch/auag_g5.0_old_5-17_backup/` | 2.4M | 날짜 표기 구 백업 |
| `~/research/pymnpbem_sigma_diff/` | 176K | sigma diff 산출물 |

**zip 삭제 전 확인:** 각 zip 의 대응 케이스가 `~/research/pymnpbem/` 에
`spectrum.npz` 를 갖고 남아 있는지 먼저 확인할 것. 부분 케이스 8개는
zip 쪽에만 온전한 데이터가 있을 수 있다.

---

## 6. 재개 방법

- **중단 케이스 스펙트럼 복구** (BEM 재solve 불필요):
  `~/scratch/spectrum_from_cache.py` — sigma 캐시에서 관측량 재계산.
  검증 오차 7.9e-5(free) / 7.7e-10(layer).
- **auagagcl / auagago 나머지 gap 재개**:
  `~/scratch/run_auagagcl_ago_sweep.sh` (2026-07-27 결정으로 g0.6 쌍만 완료, 6개 취소)
- **22560 face 기판 계열은 FP32 필수** (`MNPBEM_GPU_LOWPREC=1`).
  FP64 는 A6000 에서 OOM. 정확도 차 <0.06% 검증 완료.
- 분석 노하우·실험 대조: [EXPERIMENT_ANALYSIS_HANDOFF.md](EXPERIMENT_ANALYSIS_HANDOFF.md)
