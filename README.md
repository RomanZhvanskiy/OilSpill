# Oil--Wave Coupled Slick Model (2D x--y, time-dependent)

A simulation-ready, Eulerian oil-slick model with **two-way coupling**
between an evolving surface oil film and a simplified **two-band wave
field**:

-   **Small waves**
-   **Large waves**

The model captures a closed physical feedback loop:

-   wind drives waves\
-   oil damps waves\
-   waves advect and drift oil\
-   oil thickness evolves (advection + diffusion + gravity spreading +
    source)\
-   oil thickness feeds back into wave damping

The goal is to retain **physical interpretability** while remaining
**numerically implementable** without a full spectral wave model.

------------------------------------------------------------------------

# 1. Model overview

We simulate a horizontal 2-D domain `(x, y)` forward in time `t` with
four evolving fields:

1.  **Oil thickness** `T(x, y, t)`
2.  **Oil velocity** `Uo(x, y, t)`
3.  **Small-wave amplitude** `As(x, y, t)`
4.  **Large-wave amplitude** `AL(x, y, t)`

This forms a **fully coupled oil--wave system**.

------------------------------------------------------------------------





# 2. Governing equations (conceptual form)

## 2.1 Small waves `As`

Advection + wind growth − damping:

    dAs/dt + cg_s · grad(As)
      = Cs * Ueff^2
      - Ds0 * As
      - Ds_oil(T) * As

Oil damping example:

    Ds_oil(T) = Ds_max * (1 − exp(−T / Ts))

Small waves are **strongly suppressed** once thickness approaches `Ts`.

------------------------------------------------------------------------

## 2.2 Large waves `AL`

Large waves grow from wind **only when small waves exist**:




    dAL/dt + cg_L · grad(AL)
      = CL * Ueff^2 * G(As)
      - DL0 * AL







      - DL_oil(T) * AL



Oil damping is weaker:

    DL_oil(T) = DL_max * (1 − exp(−T / (m * Ts)))

with `m >> 1`, meaning **long waves require thicker oil** to be damped.

------------------------------------------------------------------------




## 2.3 Oil velocity `Uo`







Oil motion combines:




-   background current\
-   Stokes drift from large waves\
-   optional **surfing** when oil is thick relative to wave orbital
    amplitude

```{=html}
<!-- -->
```
    Uo = Ucurr + Ustokes(AL) + gamma(chi) * c_phase

Thickness ratio:

    chi = T / (aL + eps)



Behavior:









-   `chi << 1` → `gamma ≈ 0` → pure Stokes drift\
-   `chi >> 1` → `gamma → 1` → partial phase-speed transport

**No direct wind drag on oil** --- wind acts only via waves and
currents.

------------------------------------------------------------------------

## 2.4 Oil thickness `T`

Conservative transport with diffusion, gravity spreading, and source:






    dT/dt + div(Uo * T)
      = div(DT(AL) * grad(T))
      + div(Cgrav * grad(T^3))


      + S(x, y, t)

Wave-enhanced diffusion example:

    DT(AL) = D0 + k_wave * AL^2

Gravity spreading uses a **nonlinear thin-film (Fay-type) term**.

Oil thickness feeds back into **both wave damping terms**, closing the
loop.

------------------------------------------------------------------------




# 3. Simulation loop








## 3.1 Wave spin-up (`t < 0`)

-   Set `T = 0`
-   Evolve `As` and `AL` under wind/current
-   Reach steady or quasi-steady wave field

## 3.2 Coupled evolution (`t ≥ 0`)





Each timestep:




1.  Update **small waves**
2.  Update **large waves**
3.  Compute **oil velocity**
4.  Update **oil thickness**
5.  Apply **wave damping from thickness**
6.  Advance time

------------------------------------------------------------------------

# 4. Discretization

-   **Finite-volume method** on structured or stretched grid\
-   Cell-centered storage for `T`, `As`, `AL`, `Ux`, `Uy`\
-   **High-order upwind** advection\
-   **Second-order central** diffusion & gravity spreading\
-   Localized **source injection** at spill location\
-   Enforced **non-negative thickness**


------------------------------------------------------------------------

# 5. Stability constraints

Advection CFL:

    dt < min(dx / |Ux|, dy / |Uy|)








Explicit diffusion may impose a **stronger timestep restriction**.

Small-wave dynamics evolve fastest → timestep must respect this scale.

------------------------------------------------------------------------

# 6. Code structure

    main.py            # simulation driver
    state.py           # grid, parameters, fields
    time_stepping.py   # timestep integration
    grid_builder.py    # stretched grid construction
    io_utils.py        # configuration & outputs
    small_waves.py     # As evolution
    large_waves.py     # AL evolution
    oil_velocity.py    # Uo closure

Excluded via `.gitignore`:

    DoNotUpload/
    old_code_versions/
    run_outputs/
    __pycache__/

------------------------------------------------------------------------


## References (starting list)

Small waves / wind input / dissipation:
- Plant (1982) — wind input
- Snyder et al. (1981) — growth rates
- Komen et al. (1984) — whitecapping parameterizations
- WAMDI Group (1988) — operational spectral wave model concepts
- Banner & Young (1994) — steepness dissipation
- Phillips (1977), Kinsman (1965) — wave theory foundations

Film / viscous damping:
- Lamb (1932) — viscous wave damping
- Gade (1958) — film damping
- Foda & Cox (1980) — viscous surface films

Oil slick spreading / drift / two-layer approaches:
- Fay (1969, 1971)
- Mackay (1980)
- Elliott (1986), Lehr et al. (1984), Johansen (1984/1985)
- Stolzenbach et al. (1977), Warluzel & Benque (1981)
- Tkalich (2000), Tkalich et al. (1999) — MOSM






---


## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See `LICENSE` for details.
