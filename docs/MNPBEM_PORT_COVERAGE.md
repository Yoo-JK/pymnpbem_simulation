# MNPBEM Python Port — Feature Coverage Matrix

본 문서는 MATLAB MNPBEM 의 주요 기능이 Python port (`/home/yoojk20/workspace/MNPBEM/mnpbem/`) 에 어느 정도 구현되어 있는지 정리한 것이다.

조사일: 2026-05-02 (Phase 2 Wave 1 - Task 1)

기준:
- ✅ 있음: Python port 에 구현 완료, 시그니처 명확
- ⚠️ 부분: 일부만 있음, 보강 필요
- ❌ 없음: M5 port 작업 대상

---

## 1. Mirror Symmetry — `comparticlemirror`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `ComParticleMirror` 클래스 | `mnpbem/geometry/comparticle_mirror.py:107` | `class ComParticleMirror(object)` | ✅ 있음 |
| `CompStructMirror` 헬퍼 | `mnpbem/geometry/comparticle_mirror.py:9` | `class CompStructMirror(object)` | ✅ 있음 |
| BEM mirror solvers | `BEMStatMirror` / `BEMRetMirror` / `BEMLayerMirror` / `BEMStatEigMirror` | `mnpbem/bem/bem_*_mirror.py` | ✅ 있음 |
| Excitation mirror | `PlaneWaveStatMirror` / `PlaneWaveRetMirror` / `DipoleStatMirror` / `DipoleRetMirror` | `mnpbem/simulation/*_mirror.py` | ✅ 있음 |
| Green function mirror | `CompGreenStatMirror` / `CompGreenRetMirror` | `mnpbem/greenfun/compgreen_*_mirror.py` | ✅ 있음 |

**EELS + mirror 비호환** 은 MATLAB과 동일하게 미지원 (정상).

---

## 2. Layered Green Function — `compgreentablayer + tabspace`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `CompGreenTabLayer` (multi-tab 핸들러) | `mnpbem/greenfun/compgreentab_layer.py:208` | `__init__(self, p1, p2, tabs)` | ✅ 있음 |
| `_MultiGreenTabLayer` (internal multi) | `mnpbem/greenfun/compgreentab_layer.py:12` | `__init__(self, layer, tabs)` | ✅ 있음 |
| `GreenTabLayer` (single tab) | `mnpbem/greenfun/greentab_layer.py:30` | `__init__(...)` + `set(enei_arr, **options)` | ✅ 있음 |
| `GreenRetLayer` (reflected) | `mnpbem/greenfun/greenret_layer.py` | — | ✅ 있음 |
| `LayerStructure.tabspace()` | `mnpbem/geometry/layer_structure.py:2190` | `tabspace(self, ...)` | ✅ 있음 |
| `BEMRetLayer` solver | `mnpbem/bem/bem_ret_layer.py` | — | ✅ 있음 |
| `BEMStatLayer` solver | `mnpbem/bem/bem_stat_layer.py` | — | ✅ 있음 |
| `SpectrumRetLayer` / `SpectrumStatLayer` | `mnpbem/spectrum/spectrum_*_layer.py` | — | ✅ 있음 |

---

## 3. Field Calculation — `meshfield(mindist, nmax)`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `MeshField` | `mnpbem/simulation/meshfield.py:18` | `__init__(self, p, x, y, z=None, nmax=None, mindist=None, ...)` + `__call__(self, sig, inout=2, fmm=False, fmm_eps=1e-12)` | ✅ 있음 |
| FMM (multipole) acceleration | `mnpbem/simulation/meshfield_fmm.py` | `fmm=True` 옵션 | ✅ 있음 |
| Numba JIT | `mnpbem/simulation/_meshfield_numba.py` | 자동 적용 | ✅ 있음 |

`nmax` (chunk-by-chunk) 와 `mindist` (surface 회피) 모두 MATLAB과 동등하게 지원됨.

---

## 4. Nonlocal — `nonlocal eps + cover layer + refun`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `coverlayer.shift(p1, d, op, ...)` (cover layer 생성) | `mnpbem/greenfun/coverlayer.py:25` | `shift(p1, d, op, ...)` | ✅ 있음 |
| `coverlayer.refine(p, ind)` (refun 생성) | `mnpbem/greenfun/coverlayer.py:159` | `refine(p, ind) -> Callable` | ✅ 있음 |
| `coverlayer.refineret` / `refinestat` | `mnpbem/greenfun/coverlayer.py:280, 362` | helper | ✅ 있음 |
| **`EpsNonlocal` (hydrodynamic 양자모델 dielectric)** | — | — | ❌ 없음 |

**결론**: 비국소 인프라 (cover layer, refun) 는 모두 있으나, MATLAB `epshydrodynamic` 동등 클래스 (`EpsNonlocal`) 가 없다.

**Workaround**: 사용자가 직접 `EpsFun` 으로 hydrodynamic ε 함수를 정의하면 동작 가능 (MATLAB demospecstat19/20 패턴).

**M5 port 대상**: `EpsNonlocal` 클래스 (얇은 wrapper 1개로 충분).

---

## 5. Iterative Solver — `bemiter (BEM*Iter)`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `BEMIter` (base) | `mnpbem/bem/bem_iter.py:11` | `__init__(self, ...)` | ✅ 있음 |
| `BEMStatIter` | `mnpbem/bem/bem_stat_iter.py:15` | `class BEMStatIter(BEMIter)` | ✅ 있음 |
| `BEMRetIter` | `mnpbem/bem/bem_ret_iter.py:15` | `class BEMRetIter(BEMIter)` | ✅ 있음 |
| `BEMRetLayerIter` | `mnpbem/bem/bem_ret_layer_iter.py` | — | ✅ 있음 |

---

## 6. H2 / ACA Compression — `H-matrix Green`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `HMatrix` | `mnpbem/greenfun/hmatrix.py:94` | `__init__(self, ...)` | ✅ 있음 |
| `ClusterTree` (binary cluster) | `mnpbem/greenfun/clustertree.py` | — | ✅ 있음 |
| `ACACompGreenStat` | `mnpbem/greenfun/aca_compgreen_stat.py:11` | `class ACACompGreenStat(object)` | ✅ 있음 |
| `ACACompGreenRet` | `mnpbem/greenfun/aca_compgreen_ret.py:11` | `class ACACompGreenRet(object)` | ✅ 있음 |
| `ACACompGreenRetLayer` | `mnpbem/greenfun/aca_compgreen_ret_layer.py` | — | ✅ 있음 |
| GPU ACA (`aca_block_gpu`) | `mnpbem/greenfun/aca_gpu.py:113` | `aca_block_gpu(fun, ...)` | ✅ 있음 |
| GPU H-matrix | `mnpbem/greenfun/h_matrix_gpu.py` | — | ✅ 있음 |
| `make_kaware_fadmiss(k)` (admiss helper) | `mnpbem/greenfun/hmatrix.py:1118` | — | ✅ 있음 |

H2 압축의 모든 핵심 (cluster + ACA + low-rank assembly) 이 구현되어 있다.

---

## 7. Dipole Excitation — `DipoleRet, DipoleStat`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `DipoleRet` | `mnpbem/simulation/dipole_ret.py:19` | `__init__(self, pt, dip=None, full=False, medium=1, pinfty=None, **options)` + `__call__(self, p, enei)` | ✅ 있음 |
| `DipoleStat` | `mnpbem/simulation/dipole_stat.py:20` | — | ✅ 있음 |
| `DipoleRetMirror` / `DipoleStatMirror` | `mnpbem/simulation/dipole_*_mirror.py` | — | ✅ 있음 |
| `DipoleRetLayer` / `DipoleStatLayer` | `mnpbem/simulation/dipole_*_layer.py` | — | ✅ 있음 |
| `dipole_factory.dipole(...)` | `mnpbem/simulation/dipole_factory.py` | — | ✅ 있음 |

---

## 8. EELS Excitation — `EELSRet, EELSStat`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `EELSBase` | `mnpbem/simulation/eels_base.py` | — | ✅ 있음 |
| `EELSRet` | `mnpbem/simulation/eels_ret.py:24` | `class EELSRet(EELSBase)` + `__call__` | ✅ 있음 |
| `EELSStat` | `mnpbem/simulation/eels_stat.py:25` | `class EELSStat(EELSBase)` | ✅ 있음 |
| `electronbeam(...)` factory | `mnpbem/simulation/electronbeam_factory.py` | — | ✅ 있음 |

---

## 9. Eigenmode — `bemeig (BEMStat eigenmodes)`

| 항목 | 위치 | 시그니처 | 상태 |
|---|---|---|---|
| `BEMStatEig` | `mnpbem/bem/bem_stat_eig.py:22` | `class BEMStatEig(object)` | ✅ 있음 |
| `BEMStatEigMirror` | `mnpbem/bem/bem_stat_eig_mirror.py` | — | ✅ 있음 |
| `plasmonmode(...)` helper | `mnpbem/bem/plasmonmode.py` | top-level export | ✅ 있음 |

**Retarded eigenmode** 는 MATLAB MNPBEM에도 사후 분석에서 SVD/QR 방식으로 처리 (전용 `BEMRetEig` 클래스 없음). 우리도 postprocess 단계에서 dense matrix 추출 후 NumPy `eig`/`svd` 로 처리 예정 (mnpbem_simulation의 `retarded_eigen.py` 동등 로직).

---

## 10. 기타 인프라

| 항목 | 위치 | 상태 |
|---|---|---|
| `BEMRet` / `BEMStat` (dense solver) | `mnpbem/bem/bem_*.py` | ✅ |
| `SpectrumRet` / `SpectrumStat` | `mnpbem/spectrum/spectrum_*.py` | ✅ |
| `PlaneWaveRet` / `PlaneWaveStat` | `mnpbem/simulation/planewave_*.py` | ✅ |
| `tricube` / `trisphere` / `trirod` / `triellipsoid` / `tritorus` | `mnpbem/geometry/particle.py` | ✅ |
| `tripolygon` (custom polygon → tri mesh) | `mnpbem/geometry/particle.py:787` | ✅ |
| `particle_from_mat` (legacy `.mat` import) | `mnpbem/geometry/particle.py:964` | ✅ (legacy import only) |
| `EpsConst` / `EpsTable` / `EpsDrude` / `EpsFun` | `mnpbem/materials/*.py` | ✅ |
| `multi_gpu.solve_spectrum_multi_gpu` | `mnpbem/utils/multi_gpu.py` | ✅ |
| `mpi_dispatch.solve_spectrum_mpi` | `mnpbem/utils/mpi_dispatch.py` | ✅ |
| `bemoptions` / `getbemoptions` | `mnpbem/misc/options.py` | ✅ |

---

## 요약 (빠른 점검)

| 카테고리 | ✅ | ⚠️ | ❌ |
|---|---|---|---|
| Mirror symmetry | 5 | 0 | 0 |
| Layered Green tab | 8 | 0 | 0 |
| Field calculation | 3 | 0 | 0 |
| Nonlocal | 4 | 0 | 1 (`EpsNonlocal`) |
| Iterative solver | 4 | 0 | 0 |
| H2 / ACA | 7 | 0 | 0 |
| Dipole | 5 | 0 | 0 |
| EELS | 4 | 0 | 0 |
| Eigenmode | 3 | 0 | 0 |
| 기타 (인프라) | 13 | 0 | 0 |
| **합계** | **56** | **0** | **1** |

**판단**: Python port 의 기능 커버리지는 약 **98% 완성**. 유일한 누락은 `EpsNonlocal` 클래스 wrapper 1개 (`EpsFun` 으로 우회 가능). M5 port 작업은 사실상 매우 가벼움.

`pymnpbem_simulation` 재구성 시 본 매트릭스의 모든 ✅ 기능을 직접 import 하여 사용 가능. `❌` 1개는 nonlocal 옵션 활성화 시 `EpsFun` 으로 분기 처리.
