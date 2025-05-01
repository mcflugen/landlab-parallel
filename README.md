# landlab-parallel

Some utilities for working with [landlab](https://landlab.csdms.io) in a
parallel environment.

## Install

```bash
pip install -e .
```

To run the example, you'll need to install `mpi4py`, which may be
easier to do in a `conda` environment,

```bash
conda create -n mpi4py mpi4py
conda activate mpi4py
```

## Run

To run the example,
```bash
mpiexec -n 4 python ./run_example.py 64
```
The `run_example.py` command takes a single argument, which specifies
the size of the grid.

If you are on a Mac, you may need to set the fabric interface,
```bash
FI_PROVIDER=tcp mpiexec -n 4 python ./run_example.py 64
```
