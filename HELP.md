# MNPBEM Automation Pipeline

Automated pipeline for MNPBEM (Metallic Nanoparticle Boundary Element Method) simulations.

## 📁 Directory Structure

```
mnpbem_simulation/
├── master.sh                    # Main execution script
├── run_simulation.py           # Simulation code generator
├── run_postprocess.py          # Postprocessing script
├── config/
│   └── config.py              # Configuration file
├── simulation/
│   ├── calculate.py           # Simulation manager class
│   └── sim_utils/             # Simulation utilities
│       ├── __init__.py
│       ├── geometry_generator.py
│       ├── material_manager.py
│       └── matlab_code_generator.py
└── postprocess/
    ├── postprocess.py         # Postprocessing manager class
    └── post_utils/            # Postprocessing utilities
        ├── __init__.py
        ├── data_loader.py
        ├── spectrum_analyzer.py
        └── visualizer.py
```

## 🚀 Quick Start

### 1. Setup

```bash
# Make master.sh executable
chmod +x master.sh

# Install Python dependencies
pip install numpy scipy pandas matplotlib
```

### 2. Configure Simulation

Edit `config/config.py` to set your simulation parameters:

```python
args = {}

# Structure type
args['structure'] = 'dimer_core_shell_cube'

# Geometry parameters
args['core_size'] = 20
args['shell_thickness'] = 5
args['gap'] = 10

# Materials
args['materials'] = ['air', 'silver', 'gold']

# Wavelength range
args['wavelength_range'] = [400, 800, 80]

# ... (see config.py for all options)
```

### 3. Run Simulation

```bash
./master.sh --config ./config/config.py
```

## 📊 Output Files

Results are saved in `./results/` directory:

- `simulation_results.txt` - Raw MATLAB output
- `simulation_results.mat` - MATLAB binary format
- `simulation_processed.txt` - Processed data with analysis
- `simulation_processed.csv` - CSV format for Excel
- `simulation_processed.json` - JSON format with metadata
- `simulation_spectrum.png/pdf` - Spectrum plots
- `simulation_polarization_comparison.png/pdf` - Polarization comparison

## 🎯 Supported Structures

### Single Particles
- `sphere` - Nanosphere
- `cube` - Nanocube with rounded edges
- `rod` - Nanorod/cylinder
- `ellipsoid` - Ellipsoid
- `triangle` - Triangular nanoparticle

### Core-Shell Structures
- `core_shell_sphere` - Core-shell sphere
- `core_shell_cube` - Core-shell cube

### Dimers
- `dimer_sphere` - Two coupled spheres
- `dimer_cube` - Two coupled cubes
- `dimer_core_shell_cube` - Two core-shell cubes

## 🔧 Configuration Options

### Geometry Parameters

Different structures require different parameters:

#### Sphere
```python
args['structure'] = 'sphere'
args['diameter'] = 10  # nm
```

#### Dimer Core-Shell Cube
```python
args['structure'] = 'dimer_core_shell_cube'
args['core_size'] = 20  # nm
args['shell_thickness'] = 5  # nm
args['gap'] = 10  # Gap between cubes
args['rounding'] = 0.25  # Edge rounding
```

### Materials

Available materials:
- `'air'` - Air/vacuum
- `'water'` - Water
- `'glass'` - Glass substrate
- `'gold'` - Gold (from gold.dat)
- `'silver'` - Silver (from silver.dat)
- `'aluminum'` - Aluminum (from aluminum.dat)

### Excitation Types

**Plane Wave**
```python
args['excitation_type'] = 'planewave'
args['polarizations'] = [[1,0,0], [0,1,0], [0,0,1]]
```

**Dipole**
```python
args['excitation_type'] = 'dipole'
args['dipole_position'] = [0, 0, 15]
args['dipole_moment'] = [1, 0, 0]
```

**EELS**
```python
args['excitation_type'] = 'eels'
args['impact_parameter'] = [10, 0]
args['beam_energy'] = 200e3
```

## 📈 Analysis Features

The postprocessing automatically calculates:

- **Peak wavelengths** - Resonance positions
- **Peak values** - Maximum cross sections
- **FWHM** - Full width at half maximum
- **Enhancement factors** - Polarization-dependent enhancement
- **Average/max cross sections** - Statistical measures

## 🐛 Troubleshooting

### MATLAB Not Found
```bash
# Add MATLAB to PATH
export PATH="/path/to/matlab/bin:$PATH"
```

### MNPBEM Path Error
Edit `master.sh` and update the MNPBEM path:
```bash
matlab -nodisplay -nodesktop -r "addpath(genpath('/YOUR/PATH/TO/MNPBEM')); ..."
```

### Permission Denied
```bash
chmod +x master.sh
```

## 📝 Example Configurations

### Gold Nanosphere in Water
```python
args['structure'] = 'sphere'
args['diameter'] = 50
args['materials'] = ['water', 'gold']
args['simulation_type'] = 'ret'
args['wavelength_range'] = [400, 800, 100]
```

### Silver Nanocube Dimer
```python
args['structure'] = 'dimer_cube'
args['size'] = 40
args['gap'] = 5
args['materials'] = ['air', 'silver']
args['polarizations'] = [[1,0,0], [0,1,0]]  # Parallel and perpendicular
```

### Core-Shell Nanoparticle
```python
args['structure'] = 'core_shell_sphere'
args['core_diameter'] = 30
args['shell_thickness'] = 10
args['materials'] = ['air', 'gold', 'silver']  # medium, shell, core
```

## 🔬 Advanced Options

### High Precision
```python
args['mesh_density'] = 16  # Higher mesh density
args['refine'] = 3  # Better integration
args['simulation_type'] = 'ret'  # Full Maxwell equations
```

### Fast Calculation
```python
args['mesh_density'] = 8
args['refine'] = 1
args['simulation_type'] = 'stat'  # Quasistatic approximation
```

## 📚 Citation

If you use this pipeline, please cite the MNPBEM papers:
- U. Hohenester and A. Trügler, Comp. Phys. Commun. 183, 370 (2012)
- U. Hohenester, Comp. Phys. Commun. 185, 1177 (2014)
- J. Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)

## 📧 Support

For MNPBEM-related questions, visit: https://physik.uni-graz.at/mnpbem/