# pyMNPBEM Simulation Framework

Python-based plasmonic nanoparticle simulation framework using pyMNPBEM (Boundary Element Method).

## Features

- **Optical Simulations**: Scattering, absorption, and extinction cross-section calculations
- **Field Calculations**: Near-field electric field distribution visualization
- **Surface Charge Analysis**: Plasmon mode identification (dipolar, quadrupolar, etc.)
- **Multiple Structure Types**:
  - Single particles: sphere, cube, rod, ellipsoid, triangle
  - Core-shell structures: sphere, cube, rod
  - Dimers: sphere, cube, core-shell cube
  - Advanced dimer cube with multi-shell and transformation controls
  - Sphere cluster aggregates (1-7 particles)
  - DDA shape file import

## Installation

### 1. Create conda environment

```bash
conda create -n pymnpbem python=3.11
conda activate pymnpbem
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install pyMNPBEM

Clone and install from: https://github.com/Yoo-JK/pyMNPBEM

```bash
git clone https://github.com/Yoo-JK/pyMNPBEM.git ~/pyMNPBEM
```

### 4. Configure pyMNPBEM path

Edit `config/simulation/config_simulation.py`:

```python
args['pymnpbem_path'] = '/path/to/pyMNPBEM'
```

## Quick Start

### 1. Configure structure

Edit `config/structure/config_structure.py`:

```python
args['structure'] = 'sphere'
args['diameter'] = 50  # nm
args['materials'] = ['gold']
args['medium'] = 'water'
```

### 2. Configure simulation

Edit `config/simulation/config_simulation.py`:

```python
args['simulation_type'] = 'stat'  # or 'ret' for retarded
args['wavelength_range'] = [400, 800, 100]
args['calculate_fields'] = True
args['calculate_surface_charges'] = True
```

### 3. Run simulation

```bash
python run_simulation.py
```

### 4. Run postprocessing (optional)

```bash
python run_postprocess.py /path/to/run_folder
# or
python run_postprocess.py --latest
```

## Structure Types

### Single Particles
- `sphere`: Nanosphere with diameter parameter
- `cube`: Nanocube with adjustable edge rounding
- `rod`: Nanorod/cylinder with hemispherical caps
- `ellipsoid`: 3D ellipsoid with independent semi-axes
- `triangle`: Triangular prism

### Core-Shell Structures
- `core_shell_sphere`: Spherical core-shell
- `core_shell_cube`: Cubic core-shell
- `core_shell_rod`: Cylindrical core-shell nanorod

### Dimers
- `dimer_sphere`: Two coupled spheres
- `dimer_cube`: Two coupled cubes
- `dimer_core_shell_cube`: Two core-shell cubes

### Advanced Dimer Cube
Multi-shell dimer with full transformation control:
- Multiple shell layers
- Per-layer rounding control
- Gap, offset, tilt, rotation parameters

### Sphere Cluster Aggregate
Compact close-packed clusters (1-7 spheres)

## Output

Simulation results are saved to:
- `data/`: Numerical data (numpy .npy files, TXT, CSV)
- `plots/`: Visualization plots (PNG, PDF)
- `config.json`: Simulation configuration
- `summary.json`: Analysis summary

## Simulation Capabilities

1. **Quasistatic (stat)**: Fast, for small particles (<50nm)
2. **Retarded (ret)**: Full Maxwell equations, for all sizes

### Excitation Types
- Plane wave (most common)
- Dipole source (for LDOS, Purcell factor)
- EELS (electron beam)

### Calculation Options
- Cross-sections: scattering, absorption, extinction
- Electric field distributions on 2D/3D grids
- Surface charge distributions for mode analysis
- Unpolarized light (incoherent averaging)

## Visualization

- Spectrum plots (wavelength or energy axis)
- Field enhancement maps (log/linear scale)
- Surface charge 3D plots
- Mode analysis figures
- Multi-polarization comparisons

## References

- pyMNPBEM: https://github.com/Yoo-JK/pyMNPBEM
- MNPBEM (original MATLAB): https://github.com/Nikolaos-Matthaiakakis/MNPBEM
