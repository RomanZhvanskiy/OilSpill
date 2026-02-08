# main.py
import os
import glob
import argparse
import numpy as np
from state import SystemState, PhysicalParams
from time_stepping import TimeStepper
from io_utils import save_state_hdf5, load_state_hdf5, write_state_vtu
from grid_builder import StretchedGridBuilder


def next_run_history_filename(prefix="run_history_", suffix=".txt"):
    """
    Find the next run_history_XXXX.txt filename.
    If none present, returns run_history_0000.txt.
    If max XXXX == 9999, next is 10000 (more digits allowed).
    """
    pattern = f"{prefix}*{suffix}"
    files = glob.glob(pattern)

    max_idx = -1
    for f in files:
        base = os.path.basename(f)
        # strip prefix and suffix
        if base.startswith(prefix) and base.endswith(suffix):
            middle = base[len(prefix):-len(suffix)]
            if middle.isdigit():
                idx = int(middle)
                if idx > max_idx:
                    max_idx = idx

    next_idx = max_idx + 1
    if next_idx <= 9999:
        idx_str = f"{next_idx:04d}"
    else:
        idx_str = str(next_idx)

    return f"{prefix}{idx_str}{suffix}"


class RunLogger:
    """
    Buffer everything in memory; write to file on demand.
    Timestep lines can be throttled on screen by physical time.
    """
    def __init__(self, filename):
        self.filename = filename
        self.buffer = []
        self.last_print_time = None  # in seconds (simulation time)

    def log(self, msg, t_for_screen=None, always_print=False):
        """
        msg            : string to log.
        t_for_screen   : simulation time for timestep lines (float) or None.
        always_print   : if True, always print to screen regardless of time.
        """
        line = msg if msg.endswith("\n") else msg + "\n"
        self.buffer.append(line)

        # Decide whether to print to screen
        if always_print or t_for_screen is None:
            print(msg)
            if always_print and t_for_screen is not None:
                self.last_print_time = t_for_screen
        else:
            # timestep lines: only print if >= 10 s since last printed timestep
            if (self.last_print_time is None or
                    (t_for_screen - self.last_print_time) >= 10.0):
                print(msg)
                self.last_print_time = t_for_screen

    def flush(self):
        """Write buffered lines to file and clear the buffer."""
        if not self.buffer:
            return
        with open(self.filename, "a") as f:
            f.writelines(self.buffer)
        self.buffer = []


if __name__ == "__main__":
    # ---------- parse command-line options ----------
    parser = argparse.ArgumentParser(
        description="Oil spill / wave model time-stepper"
    )
    parser.add_argument(
        "-StepperOptions",
        "--StepperOptions",
        dest="stepper_options",
        type=str,
        default=None,
        help="Optional string of options passed to TimeStepper.step()"
    )
    args = parser.parse_args()
    stepper_options = args.stepper_options
    # -----------------------------------------------

    # ----- set up run history logging -----
    log_filename = next_run_history_filename()
    logger = RunLogger(log_filename)
    logger.log(f"Logging this run to {log_filename}", always_print=True)
    # --------------------------------------

    # Load all parameters (physics + meshing + runtime) from file
    params = PhysicalParams()

    # ----- domain + grid parameters from PhysicalParams -----
    Xmin = params.Xmin
    Xmax = params.Xmax
    Ymin = params.Ymin
    Ymax = params.Ymax

    x0 = params.x0
    y0 = params.y0

    dx_min = params.dx_min
    dy_min = params.dy_min

    R_x = params.R_x
    R_y = params.R_y

    # Build stretched grid
    grid_builder = StretchedGridBuilder(
        x0=x0,
        y0=y0,
        Xmin=Xmin,
        Xmax=Xmax,
        Ymin=Ymin,
        Ymax=Ymax,
        dx_min=dx_min,
        dy_min=dy_min,
        grid_refinement_ratio_x=R_x,
        grid_refinement_ratio_y=R_y,
    )
    x, y = grid_builder.build()

    debug1 = 1
    if debug1:
        logger.log("x =")
        logger.log(str(x))
        logger.log("y =")
        logger.log(str(y))

    logger.log(f"Grid built: nx = {len(x)}, ny = {len(y)}")
    logger.log(f"  x in [{x[0]:.2f}, {x[-1]:.2f}]")
    logger.log(f"  y in [{y[0]:.2f}, {y[-1]:.2f}]")

    # ----- physics + state -----
    state = SystemState(x, y, params)

    # set uniform wind and current from parameters
    state.set_wind(params.wind_u, params.wind_v)
    state.set_current(params.current_u, params.current_v)

    # time stepping settings
    dt = params.dt
    t_start = params.t_start
    t_end = params.t_end

    stepper = TimeStepper(state, dt)

    # pre-process output times
    dump_times = np.array(params.dump_times if params.dump_times is not None else [])
    pv_times = np.array(params.paraview_times if params.paraview_times is not None else [])

    # ensure sorted
    dump_times.sort()
    pv_times.sort()

    dump_idx = 0
    pv_idx = 0

    # main simulation loop
    t = t_start
    timestep_no = 0

    logger.log(
        f"{'timestep':<10}"
        f"{'time[s]':<12}"
        f"{'max_oil_film[mm]':<22}"
        f"{'oil_volume[m^3]':<20}"
        f"{'avg_small_wave[mm]':<22}"
        f"{'avg_large_wave[m]':<20}"
        f"{'max_oil_vel[m/s]':<18}"
    )
    logger.log("-" * 130)

    # Helper function for conditional formatting
    def fmt(value, width, decimals):
        if abs(value) > 10000:
            return f"{value:<{width}.{decimals}e}"
        else:
            return f"{value:<{width}.{decimals}f}"

    while t <= t_end:
        timestep_no = timestep_no + 1
        t_next = t + dt
        diag = state.diagnostics()

        line = (
            f"{timestep_no:<10d}"
            f"{t_next:<12.3f}"
            f"{fmt(diag['maximum oil film thickness[mm]'], 22, 4)}"
            f"{fmt(diag['volume of oil in domain [m^3]'], 20, 6)}"
            f"{fmt(diag['average small wave amplitude[mm]'], 22, 4)}"
            f"{fmt(diag['average large wave amplitude[m]'], 20, 6)}"
            f"{fmt(diag['maximum oil velocity [m/s]'], 18, 4)}"
        )
        # Log timestep line; only print to screen if >= 10 s since last
        logger.log(line, t_for_screen=t_next)

        # diagnostic crash conditions
        if (
            diag["maximum oil film thickness[mm]"] > 100000.0 or
            diag["average small wave amplitude[mm]"] > 10000.0 or
            diag["average large wave amplitude[m]"] > 100.0 or
            diag["maximum oil velocity [m/s]"] > 100.0
        ):
            crash_time = t_next
            fname = f"state_{int(crash_time):06d}s_before_crash.vtu"
            logger.log(
                f"*** Diagnostic threshold exceeded at t = {crash_time:.3f} s. "
                f"Writing VTK output -> {fname} and stopping simulation. ***",
                t_for_screen=crash_time,
                always_print=True,
            )
            write_state_vtu(fname, state)
            break

        # 1) handle any HDF5 dumps whose scheduled time lies in [t, t_next]
        while dump_idx < len(dump_times) and t <= dump_times[dump_idx] <= t_next:
            dump_t = dump_times[dump_idx]
            fname = f"state_dump_{int(dump_t):06d}s.h5"
            logger.log(f"Dumping HDF5 restart at t = {dump_t:.2f} s -> {fname}")
            save_state_hdf5(fname, state, dump_t)
            dump_idx += 1

        # 2) handle any ParaView outputs in [t, t_next]
        while pv_idx < len(pv_times) and t <= pv_times[pv_idx] <= t_next:
            pv_t = pv_times[pv_idx]
            fname = f"state_{int(pv_t):06d}s.vtu"
            logger.log(f"Writing VTK output at t = {pv_t:.2f} s -> {fname}")
            write_state_vtu(fname, state)
            pv_idx += 1

        # 3) compute injection rate for this step from schedule
        Q = params.injection_rate(t)   # [m/s] at start of step

        # 4) advance solution (pass StepperOptions string)
        stepper.step(t, x0=x0, y0=y0, Q=Q, stepper_options=stepper_options)

        # flush log every 1000 timesteps
        if timestep_no % 1000 == 0:
            logger.flush()

        t = t_next


    logger.log("Simulation complete.", always_print=True)

    logger.flush()
