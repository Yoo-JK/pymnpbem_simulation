# pymnpbem_simulation — Architecture

## 디자인 원칙

1. **Pipeline-first** (`/home/yoojk20/.claude/CLAUDE.md`): 가장 단순한 dimer baseline 으로 end-to-end 동작 먼저 확보, 이후 기능 확장.
2. **MATLAB 코드 생성 제거**: 기존 wrapper 의 `simulation_script.m` 합성 단계 (4034 라인) 를 폐기. Python MNPBEM port 의 함수를 직접 호출.
3. **Python-native I/O**: `.mat` 미사용. `.npz` (압축) + `.h5` (대용량 field). 결과 분석은 `.npz` 만 로드.
4. **3-축 병렬**: `n_workers × n_threads × n_gpus_per_worker`. CPU/GPU 동일 모델.
5. **YAML config**: argparse + PyYAML. CLI override 가 YAML 보다 우선.

## 모듈 책임

| 모듈 | 책임 | 의존 |
|---|---|---|
| `cli.py` | argparse, env_setup 호출, dispatch 호출, postprocess 트리거 | (전부) |
| `config.py` | YAML 로드/검증/디폴트, snapshot 저장 | yaml |
| `auto_detect.py` | SLURM/PBS/CUDA_VISIBLE_DEVICES 감지 → (n_w, n_t, n_g) plan | os |
| `env_setup.py` | `OMP_NUM_THREADS`/`MNPBEM_GPU` 등 환경변수 set (반드시 mnpbem import 전) | os |
| `util.py` | seed, json save (NFS retry), tolerance grading | numpy |
| `structures/` | YAML structure section → MNPBEM `ComParticle` 객체 | mnpbem.geometry, materials |
| `simulation/` | excitation + BEM solver + wavelength loop | mnpbem.bem, simulation, spectrum |
| `dispatch/` | n_workers/n_gpus_per_worker 보고 적절한 runner 선택 | simulation |
| `io/` | result dict → `.npz` + `.json` 저장 | numpy |
| `postprocess/` | spectrum 분석 (peak/FWHM), plot | matplotlib |
| `migration/` | 기존 `.py` config (exec 방식) → YAML 변환 | yaml |

## 데이터 흐름

```
   YAML config + CLI args
            │
            ▼
   load_yaml + merge_overrides + apply_defaults + validate
            │
            ▼
   auto_compute_plan / explicit (n_w, n_t, n_g)
            │
            ▼
   env_setup.setup_env(n_t, n_g)   ← 반드시 mnpbem import 전
            │
            ▼
   build_structure → (ComParticle p, epstab, nfaces)
            │
            ▼
   dispatch_single_node:
     - serial CPU
     - CPU process pool (Wave 2)
     - multi-GPU dispatch (Wave 2, mnpbem.utils.multi_gpu)
     - MPI multi-node (Wave 3, mnpbem.utils.mpi_dispatch)
            │
            ▼
   result = {wavelength, ext, sca, abs, fields?, surface_charge?, ...}
            │
            ▼
   io.save_spectrum (.npz, .json)
   postprocess.analyze_spectrum (peak, FWHM, ...)
   postprocess.plot_spectrum (.png)
            │
            ▼
   {output_dir}/{name}/
       config.yaml
       run_metadata.json
       spectrum.npz
       spectrum.json
       spectrum_analysis.json
       spectrum.png
```

## Wave 1 → Wave 2 → Wave 3 → Wave 4 (M1-M10)

- Wave 1 (현재): skeleton + dimer_cube planewave_ret CPU baseline
- Wave 2 (M2-M6): GPU dispatch / field calculation / 12 구조 / dipole+EELS / substrate / postprocess full
- Wave 3 (M7, M9): mirror+iter+nonlocal / multi-node MPI+PBS
- Wave 4 (M10): 회귀 (dimer + sphere/rod 51 case + 72 demo)

## CONVENTIONS 준수 사항

- f-string 미사용 → `.format()`
- 모든 클래스 `(object)` 명시
- 함수 키워드 인자 `=` 양쪽 공백
- docstring 미사용
- match/case 패턴 (`case _: raise ValueError`)
- 텐서 결합 (concat/cat) 미사용 → empty + 슬라이스 대입

## 검증 baseline

`~/scratch/pymnpbem_sanity_test/lane_results/baseline_cpu.json` 의 `dimer 47nm × 2, gap 0.6nm, e=0.2, 6336 faces, 100 wl` 결과:

- wall_min: 60.10 (CPU 1 worker × 4 thread)
- peak ext_x @ 636.36 nm: 39344.20 (rel diff vs MATLAB: 2.4e-4 = good)

Wave 1 의 `tests/test_baseline_dimer.py` 는 wl=10 sub-sample 로 ~6 min 안에 회귀.
