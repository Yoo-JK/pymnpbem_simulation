# pymnpbem_simulation

Python wrapper for MNPBEM (Metal Nanoparticle Boundary Element Method) simulations.

> 한국어 README 는 [README.md](./README.md) 참조. (For the Korean version, see [README.md](./README.md).)

A ground-up rewrite of the former MATLAB-based `mnpbem_simulation` wrapper that instead
calls the Python MNPBEM port (`/home/yoojk20/workspace/MNPBEM`) directly.

Key changes:
- Removes the MATLAB code-generation step → calls Python functions directly
- Data formats: Python-native (`.npz` / `.h5`); no `.mat`
- Config: YAML + CLI override (argparse-based)
- 3-axis parallelism (`n_workers × n_threads × n_gpus_per_worker`)
- SLURM/PBS auto-detection

## Requirements

```bash
conda create -n pymnpbem_sim python=3.11
conda activate pymnpbem_sim

# Python MNPBEM port (editable, in sibling repo)
pip install -e /home/yoojk20/workspace/MNPBEM

# Core scientific stack
pip install numpy==2.0.2 scipy==1.14.1 matplotlib==3.10.7 pandas==2.3.3

# Config / CLI
pip install pyyaml==6.0.2 python-box==7.3.2 tqdm==4.67.1

# I/O
pip install h5py==3.12.1

# Notebook
pip install jupyter==1.1.1

# GPU acceleration (optional)
pip install cupy-cuda12x==13.3.0

# Multi-node MPI (optional)
pip install mpi4py==4.0.1

# Postprocess analysis
pip install scikit-learn==1.7.2 lmfit==1.3.2 plotly==6.5.0
```

## Quick Start

### Recommended: `--str-conf` / `--sim-conf` (mnpbem_simulation compatible)

```bash
# New pattern: structure and simulation definitions split into two .py files
python run_simulation.py \
    --str-conf examples/auag_dimer_str.py \
    --sim-conf examples/auag_dimer_sim.py \
    --verbose

# Quick check (3 wavelengths)
python run_simulation.py \
    --str-conf examples/sphere_str.py \
    --sim-conf examples/sphere_sim.py \
    --n-wavelengths 3
```

Inside `sim_conf.py`, all compute parameters are given as nested dicts such as
`compute = {n_workers, n_threads, n_gpus_per_worker, ...}` and
`output = {dir, name, ...}`. See [docs/CLI_GUIDE.md](./docs/CLI_GUIDE.md) for details.

### Sweep mode: `--sweep-conf <yaml>` (parallel multi-case)

Runs several `(str_conf, sim_conf)` pairs at once, with each worker pinned to its own GPU
(`CUDA_VISIBLE_DEVICES` isolation). On a 4-GPU node, comparing 4 cases gives 4x throughput.

```bash
python run_simulation.py --sweep-conf my_sweep.yaml
```

See [HELP.md](./HELP.md#sweep-mode---sweep-conf) for the sweep YAML format.

### Legacy: `--config <yaml>` (backward-compat)

```bash
# Single-node CPU
python run_simulation.py --config examples/dimer_baseline.yaml --n-workers 4 --n-threads 1

# Auto-detect (SLURM/PBS GPU environment)
python run_simulation.py --config examples/dimer_baseline.yaml --auto
```

### Conversion tools

```bash
# Legacy mnpbem_simulation .py → YAML
python -m pymnpbem_simulation.migration.py_to_yaml \
    /path/to/config_str.py /path/to/config_sim.py output.yaml

# YAML → --str-conf/--sim-conf .py pair
python -m pymnpbem_simulation.migration.yaml_to_str_sim \
    input.yaml out_str.py out_sim.py
```

For all CLI options see [docs/CLI_GUIDE.md](./docs/CLI_GUIDE.md), [HELP.md](./HELP.md),
or `python run_simulation.py --help`.

## Project layout

```
pymnpbem_simulation/
├── pymnpbem_simulation/    # main package
│   ├── cli.py              # CLI entry
│   ├── config.py           # YAML loader
│   ├── auto_detect.py      # SLURM/PBS GPU detection
│   ├── env_setup.py        # environment vars (MNPBEM_GPU, etc.)
│   ├── util.py             # shared utilities
│   ├── structures/         # 12+ structure builders
│   ├── simulation/         # simulation modes (planewave/dipole/eels × stat/ret)
│   ├── postprocess/        # direct numpy processing
│   ├── dispatch/           # CPU/GPU/multi-node distribution
│   ├── io/                 # .npz / .h5 output
│   └── migration/          # .py config → YAML conversion
├── tests/                  # pytest regression tests
├── examples/               # example YAML configs
└── docs/                   # design docs
```

## Status

In production use. The end-to-end pipeline that calls the Python MNPBEM port
(`/home/yoojk20/workspace/MNPBEM`) directly runs stably and is used for large campaigns
such as Au / Au@Ag / core-shell dimer sweeps.

- **Simulation modes**: planewave / dipole / EELS × quasistatic (stat) / retarded (ret), in vacuum and on substrates (layered Green / Sommerfeld)
- **12+ structure builders**: sphere, dimer, core-shell, custom dielectric shells (`refractive_index_paths`), monomer, advanced_dimer_cube (rounded-edge), etc.
- **3-axis parallelism** (`n_workers × n_threads × n_gpus_per_worker`) + SLURM/PBS auto-detect + per-GPU pinned isolated sweeps
- **GPU acceleration** (cupy) + **multi-GPU VRAM-share** (cuSolverMg distributed dense LU) — supports meshes exceeding a single GPU's VRAM (48 GB)
- **sigma cache**: dump/reload surface charges (σ) to recompute spectra, fields and observables without re-solving the BEM system + spectrum sweep RESUME
- **postprocess**: Fano analysis (qs full-eig bright/dark + dipole phase + Lorentzian fit), surface-charge visualization, eigenmode analysis
- **Validation**: 72-demo regression against MATLAB MNPBEM (max rel err ~10⁻³·⁹, median ~10⁻¹³·⁵)

## License

MIT (TBD).
