# io_utils.py
"""
IO utilities for the oil–wave model.

- Generic HDF5 dumps for restart (portable between machines)
- VTK UnstructuredGrid (.vtu) output (for ParaView) via VTK.

Dependencies:
    pip install h5py vtk
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import h5py
import vtk

from state import SystemState, PhysicalParams


# =========================
#  HDF5 RESTART IO (no approximations)
# =========================

def save_state_hdf5(filename: str, state: SystemState, t: float) -> None:
    """
    Save the full system state to a generic HDF5 file so simulations
    can be restarted (on another machine if needed).

    Contents:
      attrs:
        time
      /params (attrs: all PhysicalParams fields)
      /grid/x, /grid/y, /grid/X2D, /grid/Y2D
      /fields/{T, As, AL, Uo_x, Uo_y}
      /forcing/U_wind, /forcing/U_curr

    Grid is stored exactly, no approximations.
    """
    with h5py.File(filename, "w") as f:
        # global metadata
        f.attrs["time"] = float(t)

        # parameters
        grp_params = f.create_group("params")
        for name, value in vars(state.params).items():
            grp_params.attrs[name] = value

        # grid (1D and 2D views)
        grp_grid = f.create_group("grid")
        grp_grid.create_dataset("x", data=state.x)
        grp_grid.create_dataset("y", data=state.y)

        X2D, Y2D = np.meshgrid(state.x, state.y, indexing="xy")
        grp_grid.create_dataset("X2D", data=X2D)
        grp_grid.create_dataset("Y2D", data=Y2D)

        # fields
        grp_fields = f.create_group("fields")
        grp_fields.create_dataset("T", data=state.T)
        grp_fields.create_dataset("As", data=state.As)
        grp_fields.create_dataset("AL", data=state.AL)
        grp_fields.create_dataset("Uo_x", data=state.Uo_x)
        grp_fields.create_dataset("Uo_y", data=state.Uo_y)

        # forcing
        grp_forcing = f.create_group("forcing")
        grp_forcing.create_dataset("U_wind", data=state.U_wind)
        grp_forcing.create_dataset("U_curr", data=state.U_curr)


def load_state_hdf5(filename: str) -> Tuple[SystemState, float]:
    """
    Load a system state previously saved with save_state_hdf5.
    Returns (state, time).
    """
    with h5py.File(filename, "r") as f:
        t = float(f.attrs["time"])

        # parameters
        grp_params = f["params"]
        params_dict = {k: float(v) for k, v in grp_params.attrs.items()}
        params = PhysicalParams(**params_dict)

        # grid
        x = np.array(f["grid/x"])
        y = np.array(f["grid/y"])

        # reconstruct state
        state = SystemState(x, y, params)

        # fields
        state.T[:, :] = f["fields/T"][:]
        state.As[:, :] = f["fields/As"][:]
        state.AL[:, :] = f["fields/AL"][:]
        state.Uo_x[:, :] = f["fields/Uo_x"][:]
        state.Uo_y[:, :] = f["fields/Uo_y"][:]

        # forcing
        state.U_wind[:] = f["forcing/U_wind"][:]
        state.U_curr[:] = f["forcing/U_curr"][:]

    return state, t


# =========================
#  VTK UNSTRUCTURED OUTPUT (.vtu, exact stretched grid)
#  FOR VISUALIZATION WITH PARAVIEW
# =========================

def write_state_vtu(filename: str, state: SystemState) -> None:
    """
    Write the current state as a VTK UnstructuredGrid (.vtu),
    with the *exact* stretched coordinates used in the solver.

    Uses pure VTK.

    ParaView will then show the correct non-uniform mesh.

    Point data written:
      - oil_thickness
      - small_wave_amp
      - large_wave_amp
      - Uo_x, Uo_y (oil velocity)
      - U_air_x, U_air_y (wind velocity)
      - U_curr_x, U_curr_y (current velocity)
      - x_coord, y_coord (physical coordinates)
    """
    nx = state.nx
    ny = state.ny

    # 1. Build points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nx * ny)

    # Map (j, i) -> pointId = j*nx + i
    for j in range(ny):
        y = float(state.y[j])
        for i in range(nx):
            x = float(state.x[i])
            pid = j * nx + i
            points.SetPoint(pid, x, y, 0.0)

    # 2. Build unstructured grid and cells (quads)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(points)

    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = j * nx + i
            p1 = j * nx + (i + 1)
            p2 = (j + 1) * nx + (i + 1)
            p3 = (j + 1) * nx + i

            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, p0)
            quad.GetPointIds().SetId(1, p1)
            quad.GetPointIds().SetId(2, p2)
            quad.GetPointIds().SetId(3, p3)
            ug.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

    npts = nx * ny

    def add_scalar_point_data(name: str, array_2d: np.ndarray):
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfTuples(npts)
        flat = array_2d.ravel(order="C")
        for idx in range(npts):
            arr.SetValue(idx, float(flat[idx]))
        ug.GetPointData().AddArray(arr)

    def add_uniform_vector2_point_data(name: str, vec2: np.ndarray):
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(3)
        arr.SetNumberOfTuples(npts)
        vx, vy = float(vec2[0]), float(vec2[1])
        for idx in range(npts):
            arr.SetTuple(idx, (vx, vy, 0.0))

        ug.GetPointData().AddArray(arr)

    def add_vector2_point_data(name: str, array_x: np.ndarray, array_y: np.ndarray):
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(3)
        arr.SetNumberOfTuples(npts)
        flat_x = array_x.ravel(order="C")
        flat_y = array_y.ravel(order="C")
        for idx in range(npts):
            arr.SetTuple(idx, (float(flat_x[idx]), float(flat_y[idx]), 0.0))
        ug.GetPointData().AddArray(arr)

    # 3. Add scalar fields
    add_scalar_point_data("oil_thickness", state.T)
    add_scalar_point_data("small_wave_amp", state.As)
    add_scalar_point_data("large_wave_amp", state.AL)

    # 4. Oil velocity (vector2)
    add_vector2_point_data("U_oil", state.Uo_x, state.Uo_y)

    # 5. Air (wind) velocity (uniform vector2)
    add_uniform_vector2_point_data("U_air", state.U_wind)

    # 6. Current velocity (uniform vector2)
    add_uniform_vector2_point_data("U_curr", state.U_curr)

    # 7. Physical coordinates as arrays
    X2D, Y2D = np.meshgrid(state.x, state.y, indexing="xy")
    add_scalar_point_data("x_coord", X2D)
    add_scalar_point_data("y_coord", Y2D)

    # 8. Write to .vtu
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(ug)
    writer.Write()


# =========================
#  SELF-TEST / DEMO MODE
# =========================

def _generate_stretched_axis(max_pos: float, dx0: float, stretch: float) -> np.ndarray:
    """
    Generate 1D coordinates from 0 to ~max_pos with a cell size that
    starts at dx0 and is multiplied by 'stretch' as we move outward.

    We stop when the next point would exceed max_pos and keep the last
    point inside, so the final coordinate is the closest *below* max_pos
    that fits the geometric progression.

    Then reflect to get a symmetric axis [-..., 0, ...].
    """
    positions = [0.0]
    dx = dx0

    while True:
        new = positions[-1] + dx
        if new > max_pos:
            break
        positions.append(new)
        dx *= stretch

    # positive side (excluding 0)
    pos_side = positions[1:]
    # negative side (mirror, excluding 0)
    neg_side = [-p for p in reversed(pos_side)]
    full = neg_side + positions
    return np.array(full, dtype=float)


def _build_test_state() -> SystemState:
    """
    Build a synthetic SystemState for testing IO:

    Grid:
      - x from approx -1000 to +1000 m, stretched from 1 m at center,
        stretch factor 1.2 outward
      - y from approx -1500 to +1500 m, same mechanism

    Fields:
      - oil thickness T = 1 m at (0,0), inversely proportional to distance
        from (0,0): T = 1 / max(r, 1.0)
      - small wave height = 0.1 m everywhere
      - large wave height = 1.0 m everywhere
      - oil velocity: 0.5 m/s at 45° (x+, y+)
      - air (wind) velocity: 5 m/s at 45°
      - current velocity: 0.05 m/s at 45°
    """
    # grid
    xmax_target = 1000.0
    ymax_target = 1500.0
    dx0 = 1.0
    stretch = 1.2

    x = _generate_stretched_axis(xmax_target, dx0, stretch)
    y = _generate_stretched_axis(ymax_target, dx0, stretch)

    params = PhysicalParams()
    state = SystemState(x, y, params)

    print("Test grid:")
    print(f"  nx = {state.nx}, ny = {state.ny}")
    print(f"  x in [{state.x[0]:.3f}, {state.x[-1]:.3f}]")
    print(f"  y in [{state.y[0]:.3f}, {state.y[-1]:.3f}]")

    # velocities (45 degrees between x and y axes)
    sqrt2_inv = 1.0 / np.sqrt(2.0)

    # air (wind)
    U_air_mag = 5.0
    U_air = np.array([U_air_mag * sqrt2_inv, U_air_mag * sqrt2_inv])
    state.set_wind(U_air[0], U_air[1])

    # current
    U_curr_mag = 0.05
    U_curr = np.array([U_curr_mag * sqrt2_inv, U_curr_mag * sqrt2_inv])
    state.set_current(U_curr[0], U_curr[1])

    # oil velocity (we'll set the field directly for test)
    U_oil_mag = 0.5
    U_oil = np.array([U_oil_mag * sqrt2_inv, U_oil_mag * sqrt2_inv])

    ny, nx = state.T.shape
    X2D, Y2D = np.meshgrid(state.x, state.y, indexing="xy")
    R = np.sqrt(X2D**2 + Y2D**2)

    # oil thickness: 1 m at center, inversely proportional to distance
    # T = 1 / max(r, 1) so that near r=0 we get ~1 m
    T = 1.0 / np.maximum(R, 1.0)
    state.T[:, :] = T

    # small and large wave "heights"
    state.As[:, :] = 0.1   # 0.1 m everywhere
    state.AL[:, :] = 1.0   # 1.0 m everywhere

    # oil velocity field (uniform for test)
    state.Uo_x[:, :] = U_oil[0]
    state.Uo_y[:, :] = U_oil[1]

    return state


if __name__ == "__main__":
    """
    Testing / demo mode:

    - Build a synthetic stretched grid and test fields
    - Dump to HDF5 restart file 'test_state.h5'
    - Dump to VTK UnstructuredGrid file 'test_state.vtu'
    """
    print("io_utils: running in test mode...")

    state = _build_test_state()
    t = 0.0

    h5_name = "test_state.h5"
    vtk_name = "test_state.vtu"

    print(f"Writing HDF5 restart dump: {h5_name}")
    save_state_hdf5(h5_name, state, t)

    print(f"Writing VTK UnstructuredGrid file: {vtk_name}")
    write_state_vtu(vtk_name, state)

    print("Done. You can now:")
    print("  - restart from test_state.h5 in your solver")
    print("  - open test_state.vtu in ParaView and inspect fields.")
