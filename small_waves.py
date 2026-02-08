# small_waves.py
import numpy as np
from state import SystemState
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _small_waves_step_energy(
    E_s,        # small-wave variance field, ny x nx
    E_L,        # large-wave variance field, ny x nx
    T,          # oil thickness field, ny x nx
    x, y,       # 1D coordinates
    c_gS,       # group speed of small waves
    e_w_x, e_w_y,
    beta_s,     # Plant growth rate for this band (scalar)
    omega_s,    # small-wave frequency
    C_ds, p_steep, k_s,   # Komen whitecapping params
    Gamma_s, T_star,      # oil damping params
    dt
):
    ny, nx = E_s.shape

    u_adv = c_gS * e_w_x
    v_adv = c_gS * e_w_y

    # Precompute spatially-varying oil damping
    D_oil = 2.0 * Gamma_s * (1.0 - np.exp(-T / T_star))

    E_s_new = np.empty_like(E_s)

    for j in prange(ny):
        for i in range(nx):

            # ---------- Upwind gradient in x ----------
            if u_adv > 0.0:
                if i == 0:
                    dEs_dx = 0.0
                else:
                    dx = x[i] - x[i - 1]
                    dEs_dx = (E_s[j, i] - E_s[j, i - 1]) / dx
            elif u_adv < 0.0:
                if i == nx - 1:
                    dEs_dx = 0.0
                else:
                    dx = x[i + 1] - x[i]
                    dEs_dx = (E_s[j, i + 1] - E_s[j, i]) / dx
            else:
                dEs_dx = 0.0

            # ---------- Upwind gradient in y ----------
            if v_adv > 0.0:
                if j == 0:
                    dEs_dy = 0.0
                else:
                    dy = y[j] - y[j - 1]
                    dEs_dy = (E_s[j, i] - E_s[j - 1, i]) / dy
            elif v_adv < 0.0:
                if j == ny - 1:
                    dEs_dy = 0.0
                else:
                    dy = y[j + 1] - y[j]
                    dEs_dy = (E_s[j + 1, i] - E_s[j, i]) / dy
            else:
                dEs_dy = 0.0

            # Advection term
            adv = -(u_adv * dEs_dx + v_adv * dEs_dy)

            # Local values
            E_s_loc = E_s[j, i]
            E_L_loc = E_L[j, i]

            # ---- wind input: Plant ----
            S_in = beta_s * omega_s * E_s_loc

            # ---- whitecapping: Komen-style ----
            E_tot = E_L_loc + E_s_loc
            if E_tot > 0.0:
                H_rms_tot = 2.0 * np.sqrt(E_tot)
                steep = k_s * H_rms_tot
                S_ds = -C_ds * omega_s * (steep**p_steep) * E_s_loc
            else:
                S_ds = 0.0

            # ---- oil damping ----
            S_oil = - D_oil[j, i] * E_s_loc

            rhs = adv + S_in + S_ds + S_oil

            val = E_s_loc + dt * rhs
            if val < 0.0:
                val = 0.0

            E_s_new[j, i] = val

    return E_s_new


# —————————————— CLEAN, FAST CLASS ——————————————
class SmallWaveSolver:
    """
    Small-wave energy solver with Plant wind input + Komen whitecapping + oil damping.
    """
    def __init__(self):
        pass

    def step(self, state: SystemState, dt: float):
        p = state.params

        # --- dispersion for small band ---
        omega_s = np.sqrt(p.gravity * p.k_s)   # deep water
        c_s     = omega_s / p.k_s
        # Deep-water group speed: c_g = 1/2 c
        c_gS    = 0.5 * c_s


        # --- friction velocity and Plant growth rate ---
        U_eff  = state.U_eff                   # magnitude of effective wind
        u_star = np.sqrt(p.C_D) * U_eff        # friction velocity

        beta_s = p.B_plant_small * (u_star / c_s)**2 - p.beta0_plant_small

        if beta_s < 0.0:
            beta_s = 0.0

        # --- call Numba kernel on energy E_s ---
        state.E_s = _small_waves_step_energy(
            E_s=state.E_s,          # small-wave variance
            E_L=state.E_L,          # large-wave variance
            T=state.T,
            x=state.x,
            y=state.y,
            c_gS=c_gS,
            e_w_x=state.wind_dir[0],
            e_w_y=state.wind_dir[1],
            beta_s=beta_s,
            omega_s=omega_s,
            C_ds=p.C_ds_small,
            p_steep=p.p_steep_small,
            k_s=p.k_s,
            Gamma_s=p.Gamma_s_small,
            T_star=p.T_star_small,
            dt=dt
        )

        # If you still want an amplitude field for diagnostics:
        state.As = np.sqrt(state.E_s)
