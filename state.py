# state.py
from dataclasses import dataclass
import numpy as np
import ast
import re

PARAM_FILE = "PhysicalParams.txt"


@dataclass
class PhysicalParams:
    # Values are placeholders; they are always overwritten by PhysicalParams.txt

    # --- physics ---
    g: float = 0.0
    rho_o: float = 0.0
    mu_o: float = 0.0

    k_s: float = 0.0
    k_L: float = 0.0
    omega_L: float = 0.0


    D0_oil: float = 0.0
    kappa_wave: float = 0.0

    C_grav: float = 0.0

    # ---  small-wave energy / Plant / Komen parameters ---
    C_D: float = 0.0               # drag coefficient [-] - also applies to large waves
    B_plant_small: float = 0.0     # Plant growth constant for small waves [-]
    beta0_plant_small: float = 0.0 # Plant cutoff for small waves [-]

    C_ds_small: float = 0.0        # small-wave whitecapping coeff [-]
    p_steep_small: float = 0.0     # steepness power for small-wave whitecapping [-]

    Gamma_s_small: float = 0.0     # oil-induced small-wave damping [1/s]
    T_star_small: float = 0.0      # small-wave oil thickness scale [m]

    # ---  large-wave energy / Plant / Komen parameters ---
    B_plant_large: float = 0.0     # Plant growth constant for large waves [-]
    beta0_plant_large: float = 0.0 # Plant cutoff for large waves [-]

    C_ds_large: float = 0.0        # large-wave whitecapping coeff [-]
    p_steep_large: float = 0.0     # steepness power for large-wave whitecapping [-]

    Gamma_s_large: float = 0.0     # oil-induced large-wave damping [1/s]
    T_star_large: float = 0.0      # large-wave oil thickness scale [m]

    # --- domain / grid ---
    Xmin: float = 0.0
    Xmax: float = 0.0
    Ymin: float = 0.0
    Ymax: float = 0.0

    x0: float = 0.0       # refinement / injection centre
    y0: float = 0.0

    dx_min: float = 0.0   # finest cell sizes at (x0, y0)
    dy_min: float = 0.0

    R_x: float = 0.0      # refinement ratios in x and y
    R_y: float = 0.0

    # --- runtime / forcing ---
    wind_u: float = 0.0
    wind_v: float = 0.0
    current_u: float = 0.0
    current_v: float = 0.0

    dt: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0

    # Q(t) schedule and output times (filled by loader)
    Q_values: np.ndarray | None = None
    Q_times: np.ndarray | None = None

    dump_times: np.ndarray | None = None
    paraview_times: np.ndarray | None = None

    def __post_init__(self):
        # Load from text file on creation; will raise if anything is wrong.
        self.load_from_file(PARAM_FILE)

        # Alias used by small_waves.py (which expects p.gravity)
        self.gravity = self.g



    def load_from_file(self, filename: str):
        """
        Load physical + meshing + runtime parameters from a text file.

        Scalar keys (simple):
            key = ("value": ..., "unit": [units])

        Time series for Q(t):
            effective_thickness_injection_rate = (
                "values": [...],
                "times of values": [...],
                "value_unit": [m^3/s],
                "time_unit": [s]
            )

        Time lists:
            dump_intermediate_results_times = (
                "values": [...],
                "unit": [s]
            )
            paraview_output_times = (
                "values": [...],
                "unit": [s]
            )

        Requirements:
        - All known keys must be present.
        - Units must match exactly (ignoring spaces).
        - Unknown keys are ignored.
        - On missing parameter or wrong unit: raise RuntimeError.
        """
        # SCALAR keys: human-readable name -> (attribute name, expected_unit_string)
        keymap_scalars = {
            # physics
            "gravity":              ("g",                   "m/s^2"),
            "oil_density":          ("rho_o",               "kg/m^3"),
            "oil_viscosity":        ("mu_o",                "Pa*s"),

            "small_wave_k":         ("k_s",                 "1/m"),
            "large_wave_k":         ("k_L",                 "1/m"),
            "large_wave_omega":     ("omega_L",             "1/s"),



            "oil_diffusion":        ("D0_oil",              "m^2/s"),
            "kappa_wave":           ("kappa_wave",          "-"),

            "gravity_spread_coeff": ("C_grav",              "1/(m.s)"),

            # ---  small-wave energy / Plant / Komen params ---
            "drag_coeff":                           ("C_D",               "-"),
            "small_plant_B":                        ("B_plant_small",     "-"),
            "small_plant_beta0":                    ("beta0_plant_small", "-"),
            "small_whitecapping_coeff":             ("C_ds_small",        "-"),
            "small_whitecapping_steepness_power":   ("p_steep_small",     "-"),
            "small_oil_damping_rate":               ("Gamma_s_small",     "1/s"),
            "small_T_star":                         ("T_star_small",      "m"),

            # --- large-wave energy / Plant / Komen params ---
            "large_plant_B":                        ("B_plant_large",     "-"),
            "large_plant_beta0":                    ("beta0_plant_large", "-"),
            "large_whitecapping_coeff":             ("C_ds_large",        "-"),
            "large_whitecapping_steepness_power":   ("p_steep_large",     "-"),
            "large_oil_damping_rate":               ("Gamma_s_large",     "1/s"),
            "large_T_star":                         ("T_star_large",      "m"),


            # domain / grid
            "domain_x_min":         ("Xmin",                "m"),
            "domain_x_max":         ("Xmax",                "m"),
            "domain_y_min":         ("Ymin",                "m"),
            "domain_y_max":         ("Ymax",                "m"),

            "grid_center_x":        ("x0",                  "m"),
            "grid_center_y":        ("y0",                  "m"),

            "dx_min":               ("dx_min",              "m"),
            "dy_min":               ("dy_min",              "m"),

            "grid_refinement_ratio_x": ("R_x",              "-"),
            "grid_refinement_ratio_y": ("R_y",              "-"),

            # runtime / forcing
            "wind_u":               ("wind_u",              "m/s"),
            "wind_v":               ("wind_v",              "m/s"),
            "current_u":            ("current_u",           "m/s"),
            "current_v":            ("current_v",           "m/s"),

            "dt":                   ("dt",                  "s"),
            "simulation_start_time":("t_start",             "s"),
            "simulation_end_time":  ("t_end",               "s"),
        }

        # Complex keys that need special handling
        key_Q = "effective_thickness_injection_rate"
        key_dump = "dump_intermediate_results_times"
        key_pv = "paraview_output_times"

        all_parsed: dict[str, dict] = {}

        # ---------- NEW: multi-line-aware parsing ----------
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Physical parameters file '{filename}' not found. "
                f"Please create it before running the solver."
            )

        i = 0
        n = len(lines)
        while i < n:
            raw_line = lines[i].strip()
            i += 1

            if not raw_line or raw_line.startswith("#"):
                continue
            if "=" not in raw_line:
                continue

            key, rhs_part = [p.strip() for p in raw_line.split("=", 1)]
            rhs = rhs_part

            # If RHS is multi-line, accumulate until parentheses balance
            open_parens = rhs.count("(") - rhs.count(")")
            while open_parens > 0 and i < n:
                cont = lines[i].rstrip("\n")
                i += 1
                rhs += " " + cont.strip()
                open_parens += cont.count("(") - cont.count(")")

            rhs_dict = _parse_rhs(rhs)
            all_parsed[key] = rhs_dict
        # ---------- end multi-line-aware parsing ----------

        # ---- check scalar keys ----
        missing_scalars = [k for k in keymap_scalars.keys() if k not in all_parsed]
        if missing_scalars:
            msg = (
                f"Missing scalar parameter(s) in '{filename}':\n"
                + "\n".join(f"  - {m}" for m in missing_scalars)
                + "\nPlease add them with correct syntax and units."
            )
            raise RuntimeError(msg)

        # Assign scalar parameters
        for key, (attr_name, expected_unit) in keymap_scalars.items():
            spec = all_parsed[key]
            if "value" not in spec or "unit" not in spec:
                raise RuntimeError(
                    f"Bad syntax for scalar parameter '{key}' in {filename}.\n"
                    f"Expected: key = (\"value\": ..., \"unit\": [..])"
                )
            val = float(spec["value"])
            unit_str = str(spec["unit"])
            if not _units_match(unit_str, expected_unit):
                raise RuntimeError(
                    f"Wrong unit for parameter '{key}' in '{filename}'.\n"
                    f"  Found:    {unit_str}\n"
                    f"  Expected: [{expected_unit}] (spaces inside brackets are allowed)"
                )
            setattr(self, attr_name, val)

        # ---- Q(t) schedule ----
        if key_Q not in all_parsed:
            raise RuntimeError(
                f"Missing parameter '{key_Q}' in '{filename}' "
                f"(effective thickness injection schedule)."
            )

        spec_Q = all_parsed[key_Q]
        required_fields_Q = ["values", "times of values", "value_unit", "time_unit"]
        if any(f not in spec_Q for f in required_fields_Q):
            raise RuntimeError(
                f"Bad syntax for '{key_Q}' in '{filename}'. Expected fields:\n"
                f"  'values', 'times of values', 'value_unit', 'time_unit'."
            )

        if not _units_match(spec_Q["value_unit"], "m^3/s"):
            raise RuntimeError(
                f"Wrong value_unit for '{key_Q}'. Expected [m^3/s]."
            )

        if not _units_match(spec_Q["time_unit"], "s"):
            raise RuntimeError(
                f"Wrong time_unit for '{key_Q}'. Expected [s]."
            )

        values_Q = np.array(spec_Q["values"], dtype=float)
        times_Q = np.array(spec_Q["times of values"], dtype=float)
        if values_Q.shape != times_Q.shape:
            raise RuntimeError(
                f"'{key_Q}' has mismatched lengths: "
                f"{len(values_Q)} values vs {len(times_Q)} times."
            )
        order = np.argsort(times_Q)
        self.Q_times = times_Q[order]
        self.Q_values = values_Q[order]

        # ---- dump_intermediate_results_times ----
        if key_dump not in all_parsed:
            raise RuntimeError(
                f"Missing parameter '{key_dump}' in '{filename}'."
            )
        spec_d = all_parsed[key_dump]
        if "values" not in spec_d or "unit" not in spec_d:
            raise RuntimeError(
                f"Bad syntax for '{key_dump}' in '{filename}'. Expected fields:\n"
                f"  'values', 'unit'."
            )
        if not _units_match(spec_d["unit"], "s"):
            raise RuntimeError(
                f"Wrong unit for '{key_dump}'. Expected [s]."
            )
        dump = np.array(spec_d["values"], dtype=float)
        self.dump_times = np.sort(dump)

        # ---- paraview_output_times ----
        if key_pv not in all_parsed:
            raise RuntimeError(
                f"Missing parameter '{key_pv}' in '{filename}'."
            )
        spec_p = all_parsed[key_pv]
        if "values" not in spec_p or "unit" not in spec_p:
            raise RuntimeError(
                f"Bad syntax for '{key_pv}' in '{filename}'. Expected fields:\n"
                f"  'values', 'unit'."
            )
        if not _units_match(spec_p["unit"], "s"):
            raise RuntimeError(
                f"Wrong unit for '{key_pv}'. Expected [s]."
            )
        pv = np.array(spec_p["values"], dtype=float)
        self.paraview_times = np.sort(pv)

    # ---- helper for Q(t) ----
    def injection_rate(self, t: float) -> float:
        """
        Return volumetric injection rate Q(t) [m^3/s] using
        piecewise-linear interpolation of the schedule specified in
        PhysicalParams.txt.

        Outside the provided time range, the first/last value is held constant.
        """
        if self.Q_values is None or self.Q_times is None:
            raise RuntimeError("Injection schedule not initialised.")
        return float(np.interp(t, self.Q_times, self.Q_values))


def _parse_rhs(rhs: str) -> dict:
    """
    Convert RHS like:
        ("value": 900.0, "unit": "[kg/m^3]")
    or:
        ("values": [0, 0.1], "unit": "[s]")
    into a dict using ast.literal_eval.
    """
    rhs = rhs.strip()

    if rhs.startswith("(") and rhs.endswith(")"):
        rhs = rhs[1:-1].strip()

    rhs_dict_literal = "{" + rhs + "}"
    return ast.literal_eval(rhs_dict_literal)


def _units_match(found: str, expected: str) -> bool:
    """
    Check whether the units in 'found' match the expected units string,
    ignoring spaces and outer brackets.
    """
    s = found.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    s = s.replace(" ", "")
    e = expected.replace(" ", "")
    return s == e


class SystemState:
    """
    Holds mesh, fields (T, E_s, E_L, As, AL, Uo), and forcing (wind, current).
    For now: structured rectangular, possibly stretched grid:
    x = f(i), y = g(j).
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, params: PhysicalParams):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.nx = len(self.x)
        self.ny = len(self.y)

        # Cell-based spacings (simple, one-sided estimates at boundaries)
        self.dx = np.empty(self.nx)
        self.dy = np.empty(self.ny)
        self.dx[1:] = np.diff(self.x)
        self.dx[0] = self.dx[1]
        self.dy[1:] = np.diff(self.y)
        self.dy[0] = self.dy[1]

        # 2D fields: indexed as [j, i] -> y, x
        shape = (self.ny, self.nx)
        self.T  = np.zeros(shape)      # oil thickness

        # New: energy / variance fields
        self.E_s = np.zeros(shape)     # small-wave variance
        self.E_L = np.zeros(shape)     # large-wave variance

        # --- Seed initial wave energy so Plant input can amplify it ---
        #dominant large waves with H_rms ≈ 1.0 m
        H_rms_L0 = 1.0
        E_L0 = 0.25 * H_rms_L0**2      # E ≈ (H_rms / 2)^2
        self.E_L[:, :] = E_L0

        # Small waves much smaller, say H_rms ≈ 0.1 m
        H_rms_s0 = 0.1
        E_s0 = 0.25 * H_rms_s0**2
        self.E_s[:, :] = E_s0

        # Amplitude fields for diagnostics / coupling
        self.As = np.sqrt(self.E_s)
        self.AL = np.sqrt(self.E_L)


        # Amplitude fields (kept for diagnostics / other modules)
        self.As = np.zeros(shape)      # small-wave amplitude (derived from E_s)
        self.AL = np.zeros(shape)      # large-wave amplitude

        # Cell-centred oil velocity
        self.Uo_x = np.zeros(shape)  # oil velocity x-component
        self.Uo_y = np.zeros(shape)  # oil velocity y-component

        # Face-centred velocity arrays (staggered grid)
        self.Uo_x_face = np.zeros((self.ny, self.nx + 1))
        self.Uo_y_face = np.zeros((self.ny + 1, self.nx))

        self.params = params

        # Uniform wind & current (can be time-dependent if you like)
        self.U_curr = np.array([0.0, 0.0])   # [Ux, Uy] [m/s]
        self.U_wind = np.array([10.0, 0.0])  # default: 10 m/s in +x

    def set_wind(self, Ux: float, Uy: float):
        self.U_wind = np.array([Ux, Uy], dtype=float)

    def set_current(self, Ux: float, Uy: float):
        self.U_curr = np.array([Ux, Uy], dtype=float)

    @property
    def wind_dir(self):
        mag = np.linalg.norm(self.U_wind)
        if mag == 0.0:
            return np.array([1.0, 0.0])
        return self.U_wind / mag

    @property
    def U_eff(self):
        """
        Effective wind speed along wind direction, relative to current.
        """
        e_w = self.wind_dir
        return float((self.U_wind - self.U_curr) @ e_w)

    def diagnostics(self) -> dict[str, float]:
        """
        Return the current diagnostic quantities that are printed in the
        simulation monitor line, as a dictionary.

        Keys:
            - maximum oil film thickness[mm]
            - volume of oil in domain [m^3]
            - average small wave amplitude[mm]
            - average large wave amplitude[m]
            - maximum oil velocity [m/s]
        """
        # Oil thickness in metres → convert max to mm
        max_thickness_m = float(self.T.max())
        max_thickness_mm = max_thickness_m * 1000.0

        # Volume of oil: thickness × cell area, summed over the domain
        cell_area = self.dx[None, :] * self.dy[:, None]   # shape (ny, nx)
        oil_volume_m3 = float((self.T * cell_area).sum())

        # Average wave amplitudes
        avg_small_wave_mm = float(self.As.mean() * 1000.0)
        avg_large_wave_m  = float(self.AL.mean())

        # Oil speed magnitude
        oil_speed = np.sqrt(self.Uo_x**2 + self.Uo_y**2)
        max_oil_speed = float(oil_speed.max())

        return {
            "maximum oil film thickness[mm]"   : max_thickness_mm,
            "volume of oil in domain [m^3]"    : oil_volume_m3,
            "average small wave amplitude[mm]": avg_small_wave_mm,
            "average large wave amplitude[m]"  : avg_large_wave_m,
            "maximum oil velocity [m/s]"       : max_oil_speed,
        }


# =========================
#  SELF-TEST / BOILERPLATE
# =========================

def _print_params():
    """
    Print selected physical + meshing + runtime parameters with values and units
    as loaded from PhysicalParams.txt.
    """
    keymap_units = {
        # physics
        "g":             "m/s^2",
        "rho_o":         "kg/m^3",
        "mu_o":          "Pa*s",

        "k_s":           "1/m",
        "k_L":           "1/m",
        "omega_L":       "1/s",


        "D0_oil":        "m^2/s",
        "kappa_wave":    "-",

        "C_grav":        "1/(m.s)",

        #  small-wave parameters
        "C_D":                "-",
        "B_plant_small":      "-",
        "beta0_plant_small":  "-",
        "C_ds_small":         "-",
        "p_steep_small":      "-",
        "Gamma_s_small":      "1/s",
        "T_star_small":       "m",

        #  large-wave parameters
        "B_plant_large":      "-",
        "beta0_plant_large":  "-",
        "C_ds_large":         "-",
        "p_steep_large":      "-",
        "Gamma_s_large":      "1/s",
        "T_star_large":       "m",

        # domain / grid
        "Xmin":          "m",
        "Xmax":          "m",
        "Ymin":          "m",
        "Ymax":          "m",

        "x0":            "m",
        "y0":            "m",

        "dx_min":        "m",
        "dy_min":        "m",

        "R_x":           "-",
        "R_y":           "-",

        # runtime / forcing
        "wind_u":        "m/s",
        "wind_v":        "m/s",
        "current_u":     "m/s",
        "current_v":     "m/s",

        "dt":            "s",
        "t_start":       "s",
        "t_end":         "s",
    }

    params = PhysicalParams()
    print(f"\nLoaded parameters from '{PARAM_FILE}':\n")
    for attr, unit in keymap_units.items():
        val = getattr(params, attr)
        print(f"  {attr:15s} = {val:12g}   [{unit}]")

    if params.Q_times is not None and params.Q_values is not None:
        print("\nInjection schedule Q(t):")
        for t, q in zip(params.Q_times, params.Q_values):
            print(f"  t = {t:8.3f} s -> Q = {q:8.5g} m^3/s")

    if params.dump_times is not None:
        print("\nDump times [s]:", params.dump_times)

    if params.paraview_times is not None:
        print("ParaView output times [s]:", params.paraview_times)


if __name__ == "__main__":
    try:
        params = PhysicalParams()
    except Exception as e:
        print("\nERROR while loading physical parameters:")
        print(e)
        raise SystemExit(1)

    _print_params()
    print("\nState parameter test completed successfully.\n")
