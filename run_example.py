# FI_PROVIDER=tcp mpiexec -n 4 python run_test.py
import argparse

import numpy as np
from landlab.components import FlowAccumulator
from landlab.components import LinearDiffuser
from landlab.components import StreamPowerEroder
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from landlab_parallel import RasterTiler
from landlab_parallel import Tile
from landlab_parallel import create_landlab_grid
from landlab_parallel import pvtu_dump
from landlab_parallel import vtu_dump


def run(shape, mode="odd-r", seed=None):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    RANK = comm.Get_rank()
    n_partitions = comm.Get_size()

    if RANK == 0:
        rng = np.random.default_rng(seed=seed)
        elevation = rng.uniform(size=shape)
        uplift_rate = np.zeros_like(elevation)
        uplift_rate[1:-1, 1:-1] = 0.004

        tiler = RasterTiler.from_pymetis(shape, n_partitions, mode=mode)

        for _rank in range(1, n_partitions):
            tile = tiler.get_tile(_rank)
            offset = np.asarray([s.start for s in tiler.tiles[_rank]], dtype="i")

            comm.Send(np.array(tile.shape, dtype="i"), dest=_rank, tag=0)
            comm.Send(offset, dest=_rank, tag=1)
            comm.Send(tile.flatten(), dest=_rank, tag=2)
            comm.Send(tiler.scatter(elevation)[_rank].flatten(), dest=_rank, tag=3)
            comm.Send(tiler.scatter(uplift_rate)[_rank].flatten(), dest=_rank, tag=4)

        tile = tiler.get_tile(RANK)
        tile_shape = np.array(tile.shape, dtype="i")
        offset = np.asarray([s.start for s in tiler.tiles[RANK]], dtype="i")
        partition = np.asarray(tile, dtype=int)
        elevation = tiler.scatter(elevation)[RANK]
        uplift_rate = tiler.scatter(uplift_rate)[RANK]
    else:
        tile_shape = np.empty(2, dtype="i")
        offset = np.empty(2, dtype="i")
        comm.Recv(tile_shape, source=0, tag=0)
        comm.Recv(offset, source=0, tag=1)

        partition = np.empty(tile_shape, dtype=int)
        elevation = np.empty(tile_shape, dtype=float)
        uplift_rate = np.empty(tile_shape, dtype=float)
        comm.Recv(partition.reshape(-1), source=0, tag=2)
        comm.Recv(elevation.reshape(-1), source=0, tag=3)
        comm.Recv(uplift_rate.reshape(-1), source=0, tag=4)

    my_tile = Tile(offset, shape, partition, id_=RANK, mode=mode)

    my_ghosts = transform_values(my_tile._ghost_nodes, my_tile.local_to_global)
    their_ghosts = send_receive_ghost_ids(comm, my_ghosts)

    my_ghosts = transform_values(my_ghosts, my_tile.global_to_local)
    their_ghosts = transform_values(their_ghosts, my_tile.global_to_local)

    if mode == "odd-r":
        shift = 0.5 if offset[0] % 2 else 0.0
        xy_of_lower_left = (
            (offset[1] + shift) * 100.0,
            offset[0] * 100.0 * np.sqrt(3.0) / 2.0,
        )
    elif mode == "raster":
        xy_of_lower_left = (offset[1] * 100.0, offset[0] * 100.0)

    grid = create_landlab_grid(
        partition,
        xy_of_lower_left=xy_of_lower_left,
        spacing=100.0,
        id_=RANK,
        mode=mode,
    )

    grid.add_field("topographic__elevation", elevation, at="node")
    grid.add_field("uplift_rate", uplift_rate, at="node")

    uplift = Uplift(grid)
    fa = FlowAccumulator(grid)
    sp = StreamPowerEroder(grid, K_sp=0.0001)
    ld = LinearDiffuser(grid, linear_diffusivity=0.01)

    components = [(uplift, (250.0,)), (ld, (250.0,)), (fa, ()), (sp, (250.0,))]
    for _ in range(2000):
        for component, args in components:
            component.run_one_step(*args)
            send_receive_ghost_data(
                comm,
                my_ghosts,
                their_ghosts,
                (grid.at_node["topographic__elevation"], grid.at_node["drainage_area"]),
            )

    with open(f"{RANK}.vtu", "w") as fp:
        fp.write(vtu_dump(grid, z_coord="topographic__elevation"))

    if RANK == 0:
        tile_data = {0: grid.at_node["topographic__elevation"]}
        for rank in range(1, n_partitions):
            tile_data[rank] = tiler.empty(rank, dtype=float)
            comm.Recv(tile_data[rank], source=rank, tag=0)

        print_output(tiler.gather(tile_data))

        with open("full.pvtu", "w") as fp:
            fp.write(pvtu_dump(grid, [f"{rank}.vtu" for rank in range(n_partitions)]))
    else:
        comm.Send(grid.at_node["topographic__elevation"], dest=0, tag=0)

    return 0


class Uplift:
    def __init__(self, grid, uplift_rate=1.0):
        self.grid = grid

    def run_one_step(self, dt):
        z = self.grid.at_node["topographic__elevation"]
        dz_dt = self.grid.at_node["uplift_rate"]

        z[self.grid.core_nodes] += dz_dt[self.grid.core_nodes] * dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, help="Size of the grid")
    parser.add_argument("--mode", choices=("raster", "odd-r"), help="Grid type")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for initial elevations"
    )

    args = parser.parse_args()

    return run((args.size, 2 * args.size), args.mode, args.seed)


def print_output(array):
    import matplotlib.pyplot as plt
    import tabulate

    if True:
        plt.imshow(array)
        plt.show()
    else:
        print(tabulate.tabulate(array))


def transform_values(d: dict[int, NDArray], xform, inplace=False):
    if inplace:
        for array in d.values():
            array[:] = xform(array)
        return d
    else:
        return {key: xform(array).astype("i") for key, array in d.items()}


def send_receive_ghost_ids(comm, my_ghosts: dict[int, ArrayLike]):
    my_rank = comm.Get_rank()

    my_count = np.empty(1, dtype="i")
    their_count = np.empty(1, dtype="i")

    their_ghosts = {}
    for rank in my_ghosts:
        my_count[0] = len(my_ghosts[rank])
        comm.Sendrecv(
            my_count,
            rank,
            sendtag=my_rank,
            recvbuf=their_count,
            source=rank,
            recvtag=rank,
        )
        their_ghosts[rank] = np.empty(their_count[0], dtype="i")

    for rank in my_ghosts:
        comm.Sendrecv(
            my_ghosts[rank],
            rank,
            sendtag=my_rank,
            recvbuf=their_ghosts[rank],
            source=rank,
            recvtag=rank,
        )

    return their_ghosts


def send_receive_ghost_data(comm, my_ghosts, their_ghosts, data: tuple[NDArray, ...]):
    my_rank = comm.Get_rank()

    for rank in my_ghosts:
        for values in data:
            data_to_send = values[their_ghosts[rank]]
            data_to_receive = np.empty(len(my_ghosts[rank]), dtype=float)

            comm.Sendrecv(
                data_to_send,
                rank,
                sendtag=my_rank,
                recvbuf=data_to_receive,
                source=rank,
                recvtag=rank,
            )

            values[my_ghosts[rank]] = data_to_receive


if __name__ == "__main__":
    SystemExit(main())
