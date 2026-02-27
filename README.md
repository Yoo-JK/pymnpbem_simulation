# pyMNPBEM Simulation

Python-based MNPBEM (Metal Nanoparticle Boundary Element Method) simulation pipeline.
Performs plasmonic nanoparticle simulations including optical cross section spectra and near-field enhancement calculations.

## Requirements

```bash
conda create -n pymnpbem python=3.11
conda activate pymnpbem
pip install numpy==2.2.3 scipy==1.15.2 matplotlib==3.10.1 jupyter==1.1.1 python-box==7.3.2
pip install h5py==3.13.0
```

The `mnpbem` local package (under `simulation/`) provides the core BEM solver.

## Project Structure

```
pymnpbem_simulation/
├── config/
│   ├── simulation/
│   │   └── config_simulation.py    # Simulation parameters (wavelength, excitation, etc.)
│   └── structure/
│       ├── config_structure.py     # Nanoparticle geometry definition
│       └── guide_structure.txt     # Detailed structure configuration guide
├── simulation/
│   ├── calculate.py                # SimulationManager (core BEM solver interface)
│   └── sim_utils/                  # Simulation utility modules
├── postprocess/
│   ├── postprocess.py              # PostprocessManager (analysis and visualization)
│   └── post_utils/                 # Postprocessing utility modules
├── run_simulation.py               # Entry point: run BEM simulation
├── run_postprocess.py              # Entry point: postprocess results
├── master.sh                       # Full pipeline automation script
└── README.md
```

## Usage

### Full Pipeline (Simulation + Postprocessing)

```bash
./master.sh --str-conf ./config/structure/config_structure.py \
            --sim-conf ./config/simulation/config_simulation.py \
            --verbose
```

### Simulation Only

```bash
python run_simulation.py --str-conf ./config/structure/config_structure.py \
                         --sim-conf ./config/simulation/config_simulation.py \
                         --verbose
```

### Postprocessing Only (Reanalyze Existing Results)

```bash
./master.sh --str-conf ./config/structure/config_structure.py \
            --sim-conf ./config/simulation/config_simulation.py \
            --reanalyze --verbose
```

Or directly:

```bash
python run_postprocess.py --str-conf ./config/structure/config_structure.py \
                          --sim-conf ./config/simulation/config_simulation.py \
                          --verbose
```

## Configuration

### Simulation Config (`config/simulation/config_simulation.py`)

- `simulation_type`: `'stat'` (quasistatic) or `'ret'` (retarded)
- `excitation_type`: `'planewave'`, `'dipole'`, or `'eels'`
- `wavelength_range`: `[start_nm, end_nm, num_points]`
- `use_parallel` / `num_workers`: Python multiprocessing control
- `save_format`: `'npz'` (default), `'mat'`, or `'hdf5'`
- See `config/simulation/config_simulation.py` for full options

### Structure Config (`config/structure/config_structure.py`)

Supported structures:
- Single particles: `sphere`, `cube`, `rod`, `ellipsoid`, `triangle`
- Core-shell: `core_shell_sphere`, `core_shell_cube`, `core_shell_rod`
- Dimers: `dimer_sphere`, `dimer_cube`, `dimer_core_shell_cube`, `advanced_dimer_cube`
- Advanced: `advanced_monomer_cube`, `connected_dimer_cube`, `sphere_cluster_aggregate`
- DDA import: `from_shape`

See `config/structure/guide_structure.txt` for detailed documentation.

## Output

Results are saved to the configured `output_dir`:

- `simulation_results.npz` -- Raw simulation data (cross sections, fields)
- `simulation_processed.{txt,csv,json}` -- Processed spectra in text formats
- `field_analysis.json` -- Near-field hotspot analysis
- `simulation_spectrum.{png,pdf}` -- Spectrum plots
- `field_*.{png,pdf}` -- Near-field distribution maps
- `logs/pipeline.log` -- Execution log
