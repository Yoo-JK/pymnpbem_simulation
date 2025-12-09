# pyMNPBEM Simulation - Detailed Help

## Directory Structure

```
pymnpbem_simulation/
├── config/
│   ├── simulation/
│   │   └── config_simulation.py    # Simulation parameters
│   └── structure/
│       └── config_structure.py     # Structure definitions
├── simulation/
│   ├── __init__.py
│   ├── runner.py                   # Main simulation orchestrator
│   └── sim_utils/
│       ├── geometry_builder.py     # Particle geometry builder
│       ├── material_builder.py     # Material/dielectric functions
│       ├── bem_solver.py           # BEM solver wrapper
│       ├── field_calculator.py     # Electric field calculations
│       └── surface_charge.py       # Surface charge analysis
├── postprocess/
│   ├── __init__.py
│   ├── postprocess.py              # Postprocessing manager
│   └── post_utils/
│       ├── data_loader.py          # Load simulation results
│       ├── spectrum_analyzer.py    # Spectral analysis
│       ├── field_analyzer.py       # Field distribution analysis
│       ├── visualizer.py           # Spectrum and field plots
│       ├── surface_charge_visualizer.py  # 3D charge visualization
│       ├── geometry_cross_section.py     # Geometry overlays
│       └── data_exporter.py        # Export to TXT/CSV/JSON
├── run_simulation.py               # Main entry point
├── run_postprocess.py              # Postprocessing entry point
├── requirements.txt                # Python dependencies
└── README.md                       # Quick start guide
```

## Workflow

### 1. Configure Structure (`config/structure/config_structure.py`)

Set the particle geometry:

```python
# Example: Gold nanosphere in water
args['structure'] = 'sphere'
args['diameter'] = 50          # nm
args['mesh_density'] = 144     # Number of mesh vertices
args['materials'] = ['gold']   # Particle material
args['medium'] = 'water'       # Surrounding medium
```

### 2. Configure Simulation (`config/simulation/config_simulation.py`)

Set simulation parameters:

```python
# Path to pyMNPBEM installation
args['pymnpbem_path'] = '/path/to/pyMNPBEM'

# Simulation type
args['simulation_type'] = 'stat'  # 'stat' (quasistatic) or 'ret' (retarded)

# Excitation
args['excitation_type'] = 'planewave'
args['polarizations'] = [[1, 0, 0], [0, 1, 0]]  # x and y polarized

# Wavelength range
args['wavelength_range'] = [400, 800, 100]  # [start, end, num_points]

# Calculations
args['calculate_cross_sections'] = True
args['calculate_fields'] = True
args['calculate_surface_charges'] = True
```

### 3. Run Simulation

```bash
python run_simulation.py
```

### 4. Run Postprocessing

```bash
python run_postprocess.py /path/to/results/folder
# or
python run_postprocess.py --latest
```

## Structure Types Reference

### Single Particles

| Type | Required Parameters |
|------|---------------------|
| `sphere` | `diameter`, `mesh_density` |
| `cube` | `size`, `rounding` (0-1), `mesh_density` |
| `rod` | `diameter`, `height`, `mesh_density` |
| `ellipsoid` | `axes` ([x, y, z] semi-axes), `mesh_density` |
| `triangle` | `side_length`, `thickness` |

### Core-Shell Structures

| Type | Required Parameters |
|------|---------------------|
| `core_shell_sphere` | `core_diameter`, `shell_thickness`, `mesh_density` |
| `core_shell_cube` | `core_size`, `shell_thickness`, `rounding`, `mesh_density` |
| `core_shell_rod` | `core_diameter`, `shell_thickness`, `height`, `mesh_density` |

### Dimers

| Type | Required Parameters |
|------|---------------------|
| `dimer_sphere` | `diameter`, `gap`, `mesh_density` |
| `dimer_cube` | `size`, `gap`, `rounding`, `mesh_density` |
| `dimer_core_shell_cube` | `core_size`, `shell_thickness`, `gap`, `rounding`, `mesh_density` |

### Advanced Dimer Cube

```python
args['structure'] = 'advanced_dimer_cube'
args['core_size'] = 30              # Core size (nm)
args['shell_layers'] = [5, 3]       # Shell thicknesses (inner to outer)
args['materials'] = ['gold', 'silver', 'sio2']  # Core to outer
args['roundings'] = [0.25, 0.2, 0.15]           # Per-layer rounding
args['gap'] = 5                     # Surface-to-surface gap (nm)
args['offset'] = [0, 0, 0]          # Additional offset for particle 2
args['tilt_angle'] = 0              # Tilt angle (degrees)
args['tilt_axis'] = [0, 1, 0]       # Tilt axis
args['rotation_angle'] = 0          # Z-axis rotation (degrees)
```

### Sphere Cluster Aggregate

```python
args['structure'] = 'sphere_cluster_aggregate'
args['n_spheres'] = 3      # 1-7 spheres
args['diameter'] = 50      # nm
args['gap'] = -0.1         # Negative = overlap (contact)
args['mesh_density'] = 144
```

## Simulation Modes

### Quasistatic (`stat`)
- Fast computation
- Suitable for small particles (d < λ/10)
- Uses electrostatic approximation

### Retarded (`ret`)
- Full Maxwell equations
- Required for large particles or broad wavelength ranges
- More accurate but slower

## Materials

### Built-in Materials
- `gold`, `silver`, `aluminum`, `copper`
- `goldpalik`, `silverpalik`, `copperpalik` (Palik data)

### Medium Options
- `air`, `vacuum`: ε = 1.0
- `water`: ε = 1.77 (n ≈ 1.33)
- `glass`, `sio2`: ε ≈ 2.13-2.25

### Custom Materials

```python
# Custom constant dielectric
args['medium'] = {'type': 'constant', 'epsilon': 2.25}

# Custom refractive index file
args['refractive_index_paths'] = {
    'gold': '/path/to/gold_data.dat'  # Format: wavelength, n, k
}
```

## Output Files

### Data Files (`data/`)
- `wavelengths.npy`: Wavelength array
- `scattering.npy`, `absorption.npy`, `extinction.npy`: Cross-sections
- `field_pol{N}_enhancement.npy`: Field enhancement grids
- `charges_pol{N}_values.npy`: Surface charge values
- `spectrum_pol{N}.txt`: Text format spectrum

### Plots (`plots/`)
- `spectrum_pol{N}.png`: Individual spectrum plots
- `spectrum_comparison.png`: Multi-polarization comparison
- `field_pol{N}.png`: Field enhancement maps
- `surface_charge_pol{N}.png`: 3D surface charge plots
- `mode_analysis_pol{N}.png`: Mode identification

### Metadata
- `config.json`: Full configuration snapshot
- `summary.json`: Analysis results summary

## Surface Charge Analysis

Surface charge plots help identify plasmon modes:

- **Dipolar mode**: Single node (positive/negative split)
- **Quadrupolar mode**: Two nodes
- **Hexapolar mode**: Three nodes

The mode analysis automatically computes:
- Dipole moment magnitude
- Quadrupole tensor strength
- Charge asymmetry
- Dominant mode classification

## Tips

### Mesh Density
- Spheres: 144-200 for good accuracy
- Cubes: 12-16 sufficient for most cases
- Increase for complex shapes or high accuracy needs

### Gap Size (Dimers)
- > 10 nm: Weak coupling
- 2-5 nm: Strong coupling, large field enhancement
- < 1 nm: Ultra-strong coupling (may need nonlocal corrections)

### Field Calculation Region
- Set `field_mindist` ≥ 0.5 nm to avoid singularities
- Use appropriate grid resolution for visualization quality

## Troubleshooting

### "Cannot import pyMNPBEM"
- Check `pymnpbem_path` in config_simulation.py
- Ensure pyMNPBEM is properly installed

### Out of Memory
- Reduce `mesh_density`
- Use `use_iterative_solver = True` for large structures
- Reduce wavelength range

### Slow Computation
- Use `simulation_type = 'stat'` for small particles
- Reduce `mesh_density`
- Use symmetry if applicable (`use_mirror_symmetry`)

## References

- pyMNPBEM: https://github.com/Yoo-JK/pyMNPBEM
- MNPBEM (MATLAB): https://github.com/Nikolaos-Matthaiakakis/MNPBEM
- BEM Theory: Hohenester & Trügler, Computer Physics Communications (2012)
