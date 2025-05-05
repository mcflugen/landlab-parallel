# Release Notes

## 0.1.0 (not yet released)

- Initial code import [#1](https://github.com/mcflugen/landlab-parallel/issues/1)
- Added new function, `vtu_dump`, that dumps a *landlab* grid to *vtu* format
  [#2](https://github.com/mcflugen/landlab-parallel/issues/2).
- Added new function, `pvtu_dump`, that compbines a set of *vtu* files
  [#3](https://github.com/mcflugen/landlab-parallel/issues/3).
- Fixed an issue where the lower-left corner of the tiles were not be calculated
  correctly [#4](https://github.com/mcflugen/landlab-parallel/issues/4).
- Added a command-line argument, `mode`, that allows the user to modify
  the type of grid used for the example
  [#5](https://github.com/mcflugen/landlab-parallel/issues/5).
- Added a command-line argument, `seed`, that allows the user to provide
  a seed for the random number generator used to create the initial elevations
  [#6](https://github.com/mcflugen/landlab-parallel/issues/6).
- Added a new component, `Uplift`, that uplifts the landscape
  [#7](https://github.com/mcflugen/landlab-parallel/issues/7).
- Modified the time loop to exchange ghost node data after every component is
  updated and also added drainage area to the list of data fields that are exchanged
  [#8](https://github.com/mcflugen/landlab-parallel/issues/8).
- Fixed a bug where the mode passed from the cli was ignored
  [#9](https://github.com/mcflugen/landlab-parallel/issues/9).
- Mask-out data at closed nodes when writing to *vtu*
  [#10](https://github.com/mcflugen/landlab-parallel/issues/10).
- Fixed an issue where ghost nodes were not being set correctly for some hex grids.
  [#12](https://github.com/mcflugen/landlab-parallel/issues/12).
