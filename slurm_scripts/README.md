# SLURM Scripts

Cluster submission scripts for `pymnpbem_simulation`. Each script reads the
config path from a `CONFIG` env var, defaulting to a sensible YAML in
`examples/`.

## Scripts

| Script | Layout | Use case |
|---|---|---|
| `cpu_only.slurm` | 1 node, 32 cores, no GPU | CPU baseline / small mesh |
| `single_node_gpu.slurm` | 1 node, 4 GPUs | default (single-node spectrum sweep) |
| `multi_node_2.slurm` | 2 nodes x 4 GPUs (8-way) | medium spectrum, ~2x speedup |
| `multi_node_4.slurm` | 4 nodes x 4 GPUs (16-way) | large mesh + many wavelengths |

## Submitting

```bash
# default config
sbatch slurm_scripts/single_node_gpu.slurm

# override config
sbatch --export=CONFIG=examples/dimer_baseline.yaml \
       slurm_scripts/single_node_gpu.slurm

# multi-node (8-way wavelength split)
sbatch --export=CONFIG=examples/dimer_multinode.yaml \
       slurm_scripts/multi_node_2.slurm
```

## Requirements

- `mnpbem` conda env with mpi4py + OpenMPI for multi-node scripts:
  ```bash
  conda install -n mnpbem -c conda-forge openmpi mpi4py
  # or:
  pip install pymnpbem_simulation[mpi]
  ```
- `--mpi=pmix` flag in `srun` requires SLURM built against PMIx. If your
  cluster uses a different launcher, replace with `--mpi=pmi2` or omit.
- `logs/` directory is auto-created at submit time (relative to repo root).

## How wavelength splitting works

`solve_spectrum_mpi` (in `mnpbem.utils.mpi_dispatch`) splits the wavelength
array across MPI ranks via `np.array_split`. Each rank then dispatches its
slice to `solve_spectrum_multi_gpu` using the node's local GPUs. Results
are gathered to rank 0, which writes the final `spectrum.npz`.

So for `multi_node_2.slurm` with 100 wavelengths and 2 nodes, each node
processes 50 wavelengths in parallel; within a node the 4 GPUs split
those 50 (12-13 wavelengths per GPU).

## Auto-detect

The `--auto` flag triggers `auto_compute_plan` and `detect_multi_node` in
the wrapper. Under SLURM these read:

- `SLURM_GPUS_ON_NODE` -> `n_gpus_per_worker`
- `SLURM_CPUS_PER_TASK` -> `n_threads`
- `SLURM_NNODES > 1` -> `compute.multi_node = True`

You can still pin values explicitly via CLI flags
(`--n-workers`, `--n-gpus-per-worker`).
