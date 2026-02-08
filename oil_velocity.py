# oil_velocity.py
import numpy as np
from state import SystemState


class OilVelocityCalculator:
    """
    Computes oil velocity Uo based on:
        Uo = U_curr + U_Sto + U_surf
    where:
        U_Sto = a_L^2 * k_L * omega_L (Stokes drift)
        U_surf = gamma(chi) * c_L in wind/wave direction
        chi = T / (a_L + eps)
    """
    def __init__(self):
        pass

    def update(self, state: SystemState):
        p = state.params
        e_w = state.wind_dir

        # long-wave parameters
        kL = p.k_L
        omegaL = p.omega_L
        cL = omegaL / kL

        AL = state.AL
        T = state.T

        aL = 0.5 * AL
        eps = 1e-9

        # Stokes drift magnitude
        U_sto = aL**2 * kL * omegaL   # shape (ny, nx)

        # thickness ratio
        chi = T / (aL + eps)
        gamma = chi / (1.0 + chi)

        # surfing velocity magnitude
        U_surf = gamma * cL

        # project onto x,y
        U_sto_x = U_sto * e_w[0]
        U_sto_y = U_sto * e_w[1]
        U_surf_x = U_surf * e_w[0]
        U_surf_y = U_surf * e_w[1]

        # add background current
        Uc_x, Uc_y = state.U_curr

        state.Uo_x = Uc_x + U_sto_x + U_surf_x
        state.Uo_y = Uc_y + U_sto_y + U_surf_y

##
        Ux = state.Uo_x
        Uy = state.Uo_y
        ny, nx = Ux.shape

        # Compute face-centred velocities by simple averaging
        # Vertical faces (Ux_face): (ny, nx+1)
        for j in range(ny):
            # left boundary face: copy first cell value (or extrapolate)
            state.Uo_x_face[j, 0] = Ux[j, 0]
            # interior faces: average neighbouring cell centres
            for i in range(1, nx):
                state.Uo_x_face[j, i] = 0.5 * (Ux[j, i-1] + Ux[j, i])
            # right boundary face
            state.Uo_x_face[j, nx] = Ux[j, nx-1]

        # Horizontal faces (Uy_face): (ny+1, nx)
        for i in range(nx):
            # bottom boundary face
            state.Uo_y_face[0, i] = Uy[0, i]
            # interior faces
            for j in range(1, ny):
                state.Uo_y_face[j, i] = 0.5 * (Uy[j-1, i] + Uy[j, i])
            # top boundary face
            state.Uo_y_face[ny, i] = Uy[ny-1, i]
