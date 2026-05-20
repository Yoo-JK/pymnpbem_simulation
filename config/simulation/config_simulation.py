"""
config_simulation.py — pymnpbem 시뮬레이션 설정 REFERENCE 템플릿.

이 파일은 pymnpbem 시뮬레이션의 simulation / compute / postprocess /
output config 가 받을 수 있는 **모든 옵션**을 문서화한 reference 다.
실제 케이스 config 를 만들 때 이 파일을 복사한 뒤 필요한 옵션만
남기고 값을 채운다.

사용법:
  python run_simulation.py \
      --str-conf config_str_<case>.py \
      --sim-conf config_sim_<case>.py

config 파일은 반드시 ``args`` 라는 dict 를 정의해야 한다.

(MATLAB mnpbem_simulation 의 시뮬레이션 설정에 대응 — Migrated to pymnpbem.)
"""

import os
from pathlib import Path

args = {}

# ===================================================================
# 1. SIMULATION IDENTITY
# ===================================================================
args['simulation_name'] = 'my_simulation'     # 출력 폴더/플롯 이름

# ===================================================================
# 2. BEM SOLVER TYPE  (필수)
# ===================================================================
#   ret              — retarded, dense LU.  기본. 작은 mesh 에 빠름
#   stat             — quasi-static, dense LU
#   ret_layer        — retarded + substrate layer  (use_substrate 시 자동)
#   stat_layer       — quasi-static + substrate
#   ret_iter         — retarded + GMRES (대형 mesh / 메모리 절약)
#   stat_iter        — quasi-static iterative
#   ret_layer_iter   — retarded iterative + substrate
#   ret_mirror       — retarded + mirror 대칭
#   stat_mirror      — quasi-static + mirror
# ※ compute.use_iterative_solver=True 면 ret→ret_iter 자동 승격.
# ※ use_substrate=True 면 ret→ret_layer 자동 승격.
args['simulation_type'] = 'ret'

# ===================================================================
# 3. EXCITATION  (필수)
# ===================================================================
#   planewave — 평면파 (모든 solver 지원)
#   dipole    — 점 쌍극자 (ret/stat/ret_layer/stat_layer 만)
#   eels      — 전자 에너지 손실 분광 (동일)
args['excitation_type'] = 'planewave'

# --- planewave 파라미터 ---
args['polarizations'] = [
    [1, 0, 0],
    [0, 1, 0]]
args['propagation_dirs'] = [
    [0, 0, 1],
    [0, 0, 1]]

# --- dipole 파라미터 (excitation_type='dipole' 일 때) ---
# args['dipole_position'] = [0, 0, 20]    # [x,y,z] (nm)
# args['dipole_moment']   = [1, 0, 0]     # [x,y,z] 쌍극자 모멘트

# --- eels 파라미터 (excitation_type='eels' 일 때) ---
# args['impact_parameter'] = 5.0          # 빔 충돌 매개변수 (nm)
# args['beam_energy']      = 200e3        # 입사 전자 에너지 (eV)
# args['beam_width']       = 0.5          # 빔 폭 (nm)

# ===================================================================
# 4. WAVELENGTH / 수치 정밀도
# ===================================================================
args['wavelength_range'] = [300, 1000, 140]   # [시작 nm, 끝 nm, 점 개수]
                                              #   ※ 사용자 케이스에서
                                              #     절대 변경 금지 항목
args['relcutoff'] = 3           # int — BEM 수치 적분 정밀도 (높을수록 느림)
args['interp']    = 'curv'      # str — mesh 보간: 'flat' / 'curv' / 'polar'
# args['waitbar'] = False       # bool — 진행률 표시

# ===================================================================
# 5. 계산 TOGGLE
# ===================================================================
args['calculate_cross_sections'] = True   # 흡수/산란/소광 단면적 스펙트럼
args['calculate_fields']         = False  # near-field 계산
# args['calculate_spectrum']     = True   # False + calculate_fields=True
#                                         #   = field-only 모드

# ===================================================================
# 6. FIELD 계산 옵션  (calculate_fields=True 일 때)
# ===================================================================
args['field_region'] = {        # 평가 grid (rectangular)
    'x_range': [-80, 80, 161],  # [min, max, n_points]
    'y_range': [0, 0, 1],
    'z_range': [-80, 80, 161],
}
args['field_mindist'] = 0.5     # float — 입자 표면 최소 거리 (nm)
args['field_nmax']    = 2000    # int — 최대 평가 점 수 (메모리 제한)
args['field_wavelength_idx'] = [568, 666, 690]   # 평가할 파장 인덱스
# args['field_wavelengths'] = [550, 600, 650]    # 파장값 직접 지정 (idx 보다 우선)
args['export_field_arrays']        = False       # field 배열 .npz 저장
args['field_hotspot_count']        = 10          # hotspot (강도 peak) 개수
args['field_hotspot_min_distance'] = 3           # hotspot 간 최소 거리 (nm)

# ===================================================================
# 7. COMPUTE — 병렬화 / GPU
# ===================================================================
args['use_parallel']         = True    # bool
args['num_workers']          = 4       # int — 워커 프로세스 수 ('auto' 가능)
args['max_comp_threads']     = 1       # int — 워커당 BLAS 스레드
args['wavelength_chunk_size'] = 10     # int — 파장 청크 (메모리 분할)

args['use_mirror_symmetry']  = False   # bool — mirror 대칭 (ret_mirror)
args['use_iterative_solver'] = False   # bool — GMRES (ret→ret_iter 승격)
args['use_nonlocality']      = False   # bool — 비로컬 cover-layer
# args['use_h2_compression'] = False   # bool — H-matrix ACA-GPU (hmode)

# --- GPU 정밀도 ---
#   gpu_precision = 'fp64' : complex128 dense LU (정확, 기본)
#   gpu_precision = 'fp32' : complex64 dense LU. RTX A6000 에서 ~14x 빠름.
#       검증 결과 (vs FP64): Au dimer spectrum worst 4.5e-4,
#       surface charge worst 4.9e-4, Au@Ag spectrum worst 1.14e-3
#       — 모두 BEM 표준 허용 (1e-3) 이내. 큰 dense LU 작업에 권장.
#   (GPU 실행은 run_simulation.py 의 --n-gpus-per-worker 로 켠다)
args['gpu_precision'] = 'fp64'

# ===================================================================
# 8. ITERATIVE SOLVER 옵션  (use_iterative_solver=True 일 때)
# ===================================================================
# args['iter'] = {
#     'solver':   'gmres',     # 'gmres' / 'bicgstab' / 'cgs'
#     'tol':      1.0e-4,      # 수렴 허용오차
#     'maxit':    200,         # 최대 반복
#     'restart':  50,          # GMRES restart
#     'hmatrix':  False,       # H-matrix ACA 가속
#     'precond':  'hmat',      # 전조건화: 'hmat' / 'none'
# }

# ===================================================================
# 9. OUTPUT
# ===================================================================
args['output_dir'] = os.path.join(Path.home(), 'research/pymnpbem/my_run')
args['output_formats'] = ['npz', 'json', 'png']   # npz/json/csv/txt
args['save_plots']  = True
args['plot_format'] = ['png', 'pdf']
args['plot_dpi']    = 300

# ===================================================================
# 10. POSTPROCESS
# ===================================================================
args['spectrum_xaxis']        = 'energy'   # 'wavelength' / 'energy'
args['run_eigenmode_analysis'] = False     # quasi-static 고유모드 분석
# args['eigenmode_n']     = 10             # 계산할 고유모드 개수
# args['eigenmode_top_k'] = 5              # 시각화할 상위 k개
# args['retarded_eigen_wavelength'] = 550  # retarded 고유모드 파장 (nm)
# args['fano_target_wavelengths']   = [550, 600]   # Fano fit 목표 파장
# args['svd_rank_threshold']        = 1.0e-3       # SVD 유효계수 임계값
