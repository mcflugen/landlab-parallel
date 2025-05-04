[![License](https://img.shields.io/github/license/mcflugen/landlab-parallel.svg)](LICENSE)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/mcflugen/landlab-parallel/main.svg)](https://results.pre-commit.ci/latest/github/mcflugen/landlab-parallel/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# landlab-parallel

Some utilities for working with [landlab](https://landlab.csdms.io) in a
parallel environment.

## Installation

```bash
pip install git+https://github.com/mcflugen/landlab-parallel.git
```

> **NOTE**: To run the example, you'll need to install `mpi4py`, which may be
easiest to do in a `conda` environment,

```bash
conda create -n mpi4py mpi4py
conda activate mpi4py
```

## Run

The `run_example.py` command has a simple command-line interface,

```bash
run_example.py SIZE [--mode {raster,odd-r}] [--seed SEED]
```
* `SIZE`: Integer. The grid will have dimensions `(SIZE, 2 * SIZE)`.
* `--mode`: Optional. Grid layout type. Choices:
  * `raster`
  * `odd-r`
* `--seed`: Optional. Integer random seed for initializing grid elevations.

### Example

> **NOTE**: If you are on a Mac, you may need to first set the fabric interface
  used by *libfabric*,
```bash
export FI_PROVIDER=tcp
```

To run an example,
```bash
mpiexec -n 4 python ./run_example.py 128 --mode=odd-r --seed=1945
```
