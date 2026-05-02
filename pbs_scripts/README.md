# PBS Scripts

PBS Pro / Torque submission scripts (analogous to `slurm_scripts/`).

## Scripts

| Script | Layout | Use case |
|---|---|---|
| `single_node_gpu.pbs` | 1 node, 4 GPUs | default single-node sweep |
| `multi_node_2.pbs` | 2 nodes x 4 GPUs (8-way) | medium spectrum |
| `multi_node_4.pbs` | 4 nodes x 4 GPUs (16-way) | large mesh + many wavelengths |

## Submitting

```bash
# default config
qsub pbs_scripts/single_node_gpu.pbs

# override config
qsub -v CONFIG=examples/dimer_baseline.yaml pbs_scripts/single_node_gpu.pbs

# multi-node
qsub -v CONFIG=examples/dimer_multinode.yaml pbs_scripts/multi_node_2.pbs
```

## Requirements

- `mnpbem` conda env with mpi4py + OpenMPI on every node:
  ```bash
  conda install -n mnpbem -c conda-forge openmpi mpi4py
  ```
- `mpirun` on the PATH (OpenMPI ships its own `mpirun`).
- PBS sets `${PBS_NODEFILE}` listing one line per allocated MPI slot;
  `mpirun` reads it automatically.

## Resource selection syntax

`-l select=N:ncpus=C:ngpus=G:mpiprocs=M` reserves N "chunks" each with C
cores, G GPUs, and M MPI ranks per chunk. We launch one rank per node
(`mpiprocs=1`) so each rank can use all GPUs on its host.

If your site uses Torque (older PBS), the syntax is different
(`-l nodes=N:ppn=C:gpus=G`); adapt as needed.

## Auto-detect under PBS

The wrapper's `--auto` reads:

- `PBS_GPUFILE` -> GPU count per node
- `PBS_NP` -> CPU count
- `PBS_NODEFILE` (multiple unique hosts) -> `compute.multi_node = True`
