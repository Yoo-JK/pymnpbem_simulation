"""
config_structure.py — pymnpbem 구조(geometry) + 재료 설정 REFERENCE 템플릿.

이 파일은 pymnpbem 시뮬레이션의 structure config 가 받을 수 있는
**모든 옵션**을 문서화한 reference 다. 실제 케이스 config 를 만들 때
이 파일을 복사한 뒤 해당 구조에 필요한 옵션만 남기고 값을 채운다.

사용법:
  python run_simulation.py \
      --str-conf config_str_<case>.py \
      --sim-conf config_sim_<case>.py

config 파일은 반드시 ``args`` 라는 dict 를 정의해야 한다.
아래는 모든 옵션을 보여주는 reference 이므로, 실제 케이스에서는
선택한 structure type 에 해당하는 키만 남긴다.

(MATLAB mnpbem_simulation 의 구조 설정에 대응 — Migrated to pymnpbem.)
"""

args = {}

# ===================================================================
# 1. STRUCTURE IDENTITY
# ===================================================================
args['structure_name'] = 'my_structure'      # 출력 폴더/플롯 이름에 사용

# ===================================================================
# 2. STRUCTURE TYPE  (필수) — 아래 중 하나
# ===================================================================
#   단일 입자 : sphere, cube, rod, ellipsoid, triangle
#   dimer     : dimer_sphere, dimer_cube, dimer_core_shell_cube
#   core-shell: core_shell_sphere, core_shell_cube, core_shell_rod
#   advanced  : advanced_monomer_cube, advanced_dimer_cube,
#               connected_dimer_cube   (gap <= 0, mesh 융합 touching)
#   cluster   : sphere_cluster          (구 1~7개 육각 배치)
#   파일      : from_shape              (vertices/faces 또는 mesh_file)
#   wrapper   : with_substrate, with_mirror, with_nonlocal
args['structure'] = 'advanced_dimer_cube'

# ===================================================================
# 3. 크기 / 치수  (structure type 에 따라 필요한 것만)
# ===================================================================
args['diameter']      = 50      # float — sphere / rod / cluster 의 지름 (nm)
args['size']          = 47      # float — cube / dimer_cube 한 변 (nm). 별칭 'edge'
args['core_size']     = 47      # float — core-shell / advanced_* 의 코어 한 변 (nm)
args['height']        = 50      # float — rod / core_shell_rod 높이 (nm)
args['core_diameter'] = 30      # float — core_shell_sphere / _rod 코어 지름 (nm)
args['axes']          = [10, 15, 20]   # list — ellipsoid 반축 [a, b, c] (nm)
args['side_length']   = 30      # float — triangle 한 변 (nm)
args['thickness']     = 5       # float — triangle 프리즘 두께 (nm)

# ===================================================================
# 4. 셸 / 코팅  (core-shell, advanced_* 구조)
# ===================================================================
args['shell_layers'] = [4]              # list[float] — 각 셸 두께 (nm). [] = 셸 없음
args['roundings']    = [0.2, 0.2]       # list[float] — 레이어별 모서리 반경
                                        #   [core_rounding, shell1_rounding, ...]
# args['rounding']   = 0.2              # float — 모든 레이어 동일 반경 (roundings 대안)
# args['shell_thickness'] = 4           # float — 단일 셸 (레거시; shell_layers 권장)

# ===================================================================
# 5. MESH 밀도
# ===================================================================
args['mesh_density'] = 2        # float — 요소 크기 (nm). 작을수록 조밀/정확/느림
                                #   ※ 사용자 케이스에서 절대 변경 금지 항목
# args['n_per_edge']  = 24      # int — cube 모서리당 분할 (mesh_density 가 우선)
# args['n_per_edges'] = [24,28] # list[int] — 레이어별 분할 [core, shell1, ...]
# args['n_verts']     = 256     # int — 구형 입자 삼각형 꼭짓점 수
# args['rod_mesh']    = [15,20,20]  # list[int] — rod [nphi, ntheta, nz]
args['refine']       = 3        # int — BEM 정제 레벨 (ComParticle interp 정제)

# ===================================================================
# 6. 배치 / 변환  (dimer 구조)
# ===================================================================
args['gap']            = 0.2          # float — dimer 입자 간 갭 (nm).
                                      #   음수 = 접촉/겹침 (connected_dimer 는 <=0)
args['offset']         = [0, 0, 0]    # list — 두 번째 입자 평행이동 [dx,dy,dz] (nm)
args['tilt_angle']     = 0            # float — 두 번째 입자 회전각 (도)
args['tilt_axis']      = [1, 0, 0]    # list — tilt 회전축
args['rotation_angle'] = 0            # float — 두 번째 입자 Z축 회전 (도)
# args['horizontal']   = True         # bool — rod 수평 배치 (기본 True)

# ===================================================================
# 7. 메시 파일 로드  (structure = 'from_shape')
# ===================================================================
# args['mesh_file'] = '/path/to/mesh.npz'   # .npz 또는 .mat
# args['mesh_key']  = 'p'                   # .mat 내부 verts/faces 접두사
# args['vertices']  = None                  # N x 3 직접 입력 (mesh_file 대안)
# args['faces']     = None                  # M x 3 (또는 M x 4)

# ===================================================================
# 8. MATERIALS — 재료 / 매질
# ===================================================================
args['medium'] = 'water'        # str — 주변 매질. water / vacuum / air / glass
                                #   또는 .dat 파일명 / 숫자(상수 eps)
args['materials'] = ['gold', 'silver']
                                # list[str] — 입자 재료 (코어→외부 순).
                                #   gold(au) / silver(ag) / *.dat / 숫자문자열.
                                #   단일 입자는 ['gold'] 처럼 1개.
# args['particle'] = 'gold'     # str — 단일 재료 단축 표기 (materials 대안)

# 커스텀 굴절률 소스 (constant eps 또는 table 파일)
args['refractive_index_paths'] = {
    'agcl': {'type': 'constant', 'epsilon': 2.02},
    # 'ito': {'type': 'table', 'path': '/path/to/ito.dat'},
}

# ===================================================================
# 9. SUBSTRATE — 기판  (structure = 'with_substrate' 또는 use_substrate)
# ===================================================================
args['use_substrate'] = False   # bool — True 면 자동으로 with_substrate 래핑
args['substrate'] = {
    'material': 'glass',        # glass / silica / silicon / water / 파일명 / 숫자
    'gap': 0.001,               # float — 입자 최하단과 기판 거리 (nm)
}

# ===================================================================
# 10. WRAPPER 옵션  (structure = 'with_mirror' / 'with_nonlocal' 일 때)
# ===================================================================
# --- with_mirror : mirror 대칭으로 BEM 가속 (절반/사분 mesh) ---
# args['structure'] = 'with_mirror'
# args['base']   = { ... 위의 일반 structure 옵션들 ... }
# args['mirror'] = {'sym': 'xy'}      # 'x' / 'y' / 'xy' (2x / 2x / 4x 가속)
#
# --- with_nonlocal : 비로컬 cover-layer (hydrodynamic Drude) ---
# args['structure'] = 'with_nonlocal'
# args['base']     = { ... }
# args['nonlocal'] = {
#     'metal': 'gold',     # Drude 금속
#     'delta_d': 0.05,     # cover-layer 두께 (nm)
#     'beta': None,        # hydrodynamic 파라미터 (None = sqrt(3/5)*v_F*hbar)
#     'eps_embed': 1.0,    # 임베딩 매질 eps
# }
