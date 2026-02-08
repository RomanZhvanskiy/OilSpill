# large_waves.py
import numpy as np
from state import SystemState
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _large_waves_step_energy(
    E_L,        # large-wave variance field, ny x nx
    E_s,        # small-wave variance field, ny x nx
    T,          # oil thickness field, ny x nx
    x, y,       # 1D coordinates
    c_gL,       # large-wave group speed
    e_w_x, e_w_y,
    beta_in_L,  # Plant-style growth rate coefficient for large band
    omega_L,    # large-wave radian frequency
    C_ds_L,     # Komen whitecapping coefficient for large band
    p_steep_L,  # steepness exponent for large-band whitecapping
    k_L,        # large-wave wavenumber
    gamma_L,    # large-wave oil damping rate [1/s]
    T_star_L,   # large-wave oil thickness scale [m]
    A_star,     # gating amplitude scale [m] (for small-wave control)
    dt
):
    ny, nx = E_L.shape

    # Advection velocities along wind
    u_adv = c_gL * e_w_x
    v_adv = c_gL * e_w_y

    # Precompute oil damping factor for energy (note: factor 2 for energy)
    # S_oil = - 2 * gamma_L * (1 - exp(-T / T_star_L)) * E_L
    D_oil = 2.0 * gamma_L * (1.0 - np.exp(-T / T_star_L))

    E_L_new = np.empty_like(E_L)

    for j in prange(ny):
        for i in range(nx):

            # ---------- Upwind gradient in x ----------
            if u_adv > 0.0:
                if i == 0:
                    dEL_dx = 0.0
                else:
                    dx = x[i] - x[i - 1]
                    dEL_dx = (E_L[j, i] - E_L[j, i - 1]) / dx
            elif u_adv < 0.0:
                if i == nx - 1:
                    dEL_dx = 0.0
                else:
                    dx = x[i + 1] - x[i]
                    dEL_dx = (E_L[j, i + 1] - E_L[j, i]) / dx
            else:
                dEL_dx = 0.0

            # ---------- Upwind gradient in y ----------
            if v_adv > 0.0:
                if j == 0:
                    dEL_dy = 0.0
                else:
                    dy = y[j] - y[j - 1]
                    dEL_dy = (E_L[j, i] - E_L[j - 1, i]) / dy
            elif v_adv < 0.0:
                if j == ny - 1:
                    dEL_dy = 0.0
                else:
                    dy = y[j + 1] - y[j]
                    dEL_dy = (E_L[j + 1, i] - E_L[j, i]) / dy
            else:
                dEL_dy = 0.0

            # Advection term
            adv = -(u_adv * dEL_dx + v_adv * dEL_dy)

            # Local fields
            E_L_loc = E_L[j, i]
            E_s_loc = E_s[j, i]

            # ---- Plant/Janssen wind input for large band, gated by small waves ----
            # A_s = sqrt(E_s); gating saturates as short waves grow.
            if A_star > 0.0 and E_s_loc > 0.0:
                A_s_loc = np.sqrt(E_s_loc)
                gating = 1.0 - np.exp(-A_s_loc / A_star)
            else:
                gating = 0.0

            S_in_L = beta_in_L * omega_L * gating * E_L_loc

            # ---- Komen-style whitecapping using total steepness ----
            E_tot = E_L_loc + E_s_loc
            if E_tot > 0.0:
                H_rms_tot = 2.0 * np.sqrt(E_tot)
                steep_tot = k_L * H_rms_tot
                # use p_steep_L from params instead of hard-coded 2
                S_ds_L = -C_ds_L * omega_L * (steep_tot ** p_steep_L) * E_L_loc
            else:
                S_ds_L = 0.0

            # ---- Oil damping for large waves ----
            S_oil_L = -D_oil[j, i] * E_L_loc

            # RHS of energy equation
            rhs = adv + S_in_L + S_ds_L + S_oil_L

            val = E_L_loc + dt * rhs
            if val < 0.0:
                val = 0.0

            E_L_new[j, i] = val

    return E_L_new


class LargeWaveSolver:
    """
    Large-wave energy solver:
      - Advection along wind with deep-water group speed
      - Plant-style wind input (band-integrated) gated by small-wave saturation
      - Komen-style whitecapping dissipation based on total steepness
      - Oil damping with exponential thickness dependence
    """
    def __init__(self):
        pass

    def step(self, state: SystemState, dt: float):
        p = state.params

        # --- dispersion for large band ---
        # Use supplied omega_L and k_L, but derive c_L and c_gL
        omega_L = p.omega_L
        c_L     = omega_L / p.k_L       # phase speed
        c_gL    = 0.5 * c_L             # deep-water group speed

        # --- friction velocity and Plant growth rate for large band ---
        U_eff  = state.U_eff
        u_star = np.sqrt(p.C_D) * U_eff

        # Plant-type growth rate for large waves
        beta_in_L = p.B_plant_large * (u_star / c_L)**2 - p.beta0_plant_large
        if beta_in_L < 0.0:
            beta_in_L = 0.0

        # Whitecapping coefficient and exponent for large band
        C_ds_L    = p.C_ds_large
        p_steep_L = p.p_steep_large

        # Call the Numba kernel on energy E_L
        state.E_L = _large_waves_step_energy(
            E_L=state.E_L,
            E_s=state.E_s,
            T=state.T,
            x=state.x,
            y=state.y,
            c_gL=c_gL,
            e_w_x=state.wind_dir[0],
            e_w_y=state.wind_dir[1],
            beta_in_L=beta_in_L,
            omega_L=omega_L,
            C_ds_L=C_ds_L,
            p_steep_L=p_steep_L,
            k_L=p.k_L,
            gamma_L=p.Gamma_s_large,
            T_star_L=p.T_star_large,   # large-wave oil thickness scale
            A_star=1e-4,               # gating amplitude scale [m]; can promote to param
            dt=dt
        )

        # Update large-wave amplitude field for diagnostics / other modules
        state.AL = np.sqrt(state.E_L)
