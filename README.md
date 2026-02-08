# Oil–Wave Coupled Slick Model (2D x–y, time-dependent)

A simulation-oriented, Eulerian oil-slick model with two-way coupling between an evolving surface oil film and a simplified two-band wave field (small waves + large waves). The model is designed to be **implementable on a structured (possibly stretched) 2D grid** without requiring a full spectral wave model, while still capturing the key feedback loop:

- wind drives waves  
- oil damps waves  
- waves advect/drift oil  
- oil thickness evolves (advection + diffusion + gravity spreading + source)  
- thickness feeds back into wave damping

---

## Overview

We simulate a 2D horizontal domain \((x,y)\) forward in time \(t\) with **four evolving fields**:

1. **Oil thickness**: \(T(x,y,t)\)
2. **Oil velocity**: \(\mathbf{U}_o(x,y,t)\)
3. **Small-wave amplitude**: \(A_s(x,y,t)\)
4. **Large-wave amplitude**: \(A_L(x,y,t)\)

This is a **closed loop** model: wave amplitudes affect oil transport; oil thickness affects wave growth/damping.

---

## Governing equations (conceptual PDE set)

### 1) Small-wave amplitude \(A_s(x,y,t)\)

Small waves are advected downwind at their group speed and evolve due to wind input and damping:

\[
\frac{\partial A_s}{\partial t} + \mathbf{c}_{g,s}\cdot\nabla A_s
= I_s(U_{\text{eff}}) - D_{s,0}A_s - D_{s,\text{oil}}(T)A_s
\]

- **Wind input** (simple quadratic dependence on effective wind):
\[
I_s(U_{\text{eff}}) = C_s\,U_{\text{eff}}^2
\]
- **Background damping**:
\[
D_{s,0}A_s
\]
- **Oil-induced damping** (strong for small waves):
\[
D_{s,\text{oil}}(T)=D_{s,\max}\left(1-e^{-T/T_s}\right)
\]
where \(T_s\) is a characteristic thickness for strong short-wave suppression.

---

### 2) Large-wave amplitude \(A_L(x,y,t)\)

Large waves are advected at their group speed, grow from wind **only if small waves exist**, and are damped by water + oil:

\[
\frac{\partial A_L}{\partial t} + \mathbf{c}_{g,L}\cdot\nabla A_L
= I_L(U_{\text{eff}},A_s) - D_{L,0}A_L - D_{L,\text{oil}}(T)A_L
\]

- **Wind input enabled by small waves**:
\[
I_L(U_{\text{eff}},A_s)=C_L\,U_{\text{eff}}^2\,G(A_s)
\]
with an enabling function \(G(A_s)\) such that:
- if \(A_s\approx 0\), then \(G(A_s)\approx 0\) (no long-wave growth)
- if \(A_s\) is appreciable, \(G(A_s)\to 1\)

- **Oil-induced damping** (weaker; requires thicker slick):
\[
D_{L,\text{oil}}(T)=D_{L,\max}\left(1-e^{-T/(mT_s)}\right),\qquad m\gg 1
\]

---

### 3) Oil velocity \(\mathbf{U}_o(x,y,t)\)

Oil moves with background current plus wave-driven drift, with an optional “surfing” correction when the oil layer is thick relative to wave orbital amplitude:

\[
\mathbf{U}_o = \mathbf{U}_{\text{curr}} + \mathbf{U}_{\text{Stokes}}(A_L) + \gamma(\chi)\,\mathbf{c}_{p,L}
\]

- **Stokes drift** from large waves (deep-water form; direction along long-wave direction).
- **Thickness-dependent surfing factor**:
\[
\chi=\frac{T}{a_L+\varepsilon},\quad
\gamma(\chi)\in[0,1]
\]
so:
- thin slick \((\chi\ll1)\Rightarrow \gamma\approx 0\)
- thick layer \((\chi\gg1)\Rightarrow \gamma\to 1\)

**No direct wind term** is applied to \(\mathbf{U}_o\); wind acts via waves and currents.

---

### 4) Oil thickness \(T(x,y,t)\)

Conservative thickness evolution with advection, diffusion (including wave-enhanced mixing), gravity-driven spreading, and injection source:

\[
\frac{\partial T}{\partial t} + \nabla\cdot(\mathbf{U}_oT)
= \nabla\cdot\left(D_T(A_L)\nabla T\right)
+ \nabla\cdot\left(C_{\text{grav}}\nabla T^3\right)
+ S(x,y,t)
\]

- **Wave-enhanced diffusion** example:
\[
D_T(A_L)=D_0+\kappa_{\text{wave}}A_L^2
\]
- **Gravity-driven spreading** (thin-film / Fay-type nonlinear flux): \(\nabla\cdot(C_{\text{grav}}\nabla T^3)\)
- **Source**: localized injection at \((x_0,y_0)\), starting at \(t=0\)

This thickness feeds back into the wave equations through \(D_{s,\text{oil}}(T)\) and \(D_{L,\text{oil}}(T)\).

---

## Simulation loop

A recommended two-phase workflow:

### Spin-up (build steady wave field)
For \(t\in[-T_{\text{spin}},0)\):
- set \(T=0\)
- evolve \(A_s, A_L\) under prescribed wind/current until steady or quasi-steady

### Coupled run
For \(t\ge 0\), at each step:
1. update **small waves** \(A_s^{n+1}\)
2. update **large waves** \(A_L^{n+1}\)
3. compute **oil velocity** \(\mathbf{U}_o^{n+1}\)
4. update **thickness** \(T^{n+1}\) (advection + diffusion + spreading + source)
5. update wave damping terms using new \(T^{n+1}\)
6. advance time

---

## Discretization (finite volume on stretched grid)

Oil thickness is discretized on a structured 2D grid with known monotone cell-center coordinates:
- \(x_i\), \(y_j\) cell centers  
- \(\Delta x_i \approx x_{i+1}-x_i\), \(\Delta y_j \approx y_{j+1}-y_j\)  
- cell area \(A_{ij}=\Delta x_i\Delta y_j\)  
- unknowns stored cell-centered: \(T_{ij}, A_{s,ij}, A_{L,ij}, U_{x,ij}, U_{y,ij}\)

Thickness equation is advanced with a conservative FV balance:
- **advection**: high-order upwind-biased reconstruction at faces
- **diffusion + gravity spreading**: second-order central gradients on non-uniform spacing
- **source**: inject thickness in the nearest cell to \((x_0,y_0)\)

Non-negativity is enforced after each update:
\[
T^{n+1}\leftarrow \max(T^{n+1},0)
\]

---

## Stability notes

Practical constraints:
- CFL for advection:
\[
\Delta t < \min\left(\frac{\Delta x}{|U_x|},\frac{\Delta y}{|U_y|}\right)
\]
- diffusion stability (if explicit diffusion is used) will be more restrictive; consider semi-implicit diffusion if needed.
- small-wave dynamics are “fast”; explicit is usually fine with a sufficiently small \(\Delta t\)

---

## Code layout (current)

From the repo root (example):

- `main.py` — simulation driver / orchestration  
- `state.py` — state container(s): fields, params, grid  
- `time_stepping.py` — integrators / update sequence  
- `grid_builder.py` — stretched grid construction  
- `io_utils.py` — I/O, parameter loading, outputs  
- `small_waves.py` — \(A_s\) evolution terms  
- `large_waves.py` — \(A_L\) evolution terms  
- `oil_velocity.py` — closure for \(\mathbf{U}_o\) from waves + current  

> Folders `DoNotUpload/`, `old_code_versions/`, `run_outputs/` (and typically `__pycache__/`) are excluded via `.gitignore`.

---

## Validation and background

This repo is inspired by (and can be cross-checked against) classic semi-empirical spreading/drift models and Eulerian oil spill frameworks, including:

- Fay spreading stages (gravity–inertia, gravity–viscous, surface tension–viscous)
- Mackay et al. style tuning for spreading area evolution
- Two-layer / Eulerian approaches such as MOSM (e.g., Tkalich and coauthors) that emphasize compatibility with hydrodynamic models and low numerical diffusion requirements for accurate advection.

The wave side is intentionally simplified (two amplitude bands) but conceptually aligned with common wave-growth/dissipation ideas used in operational wave modeling.

---

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

*(You can refine this into BibTeX later.)*

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See `LICENSE` for details.

