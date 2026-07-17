# FeDoT: Full-Wave Electromagnetic Code for Reflectometry
*A 2D Finite-Difference Time-Domain (FDTD) Synthetic Diagnostic for Plasma Profile and Fluctuation Interpretations*

**Author:** Anna MEDVEDEVA  
**Copyright:** © 2023  
**License:** MIT  

---

## 1. Overview & Physics Framework
**FeDoT** is a two-dimensional, full-wave electromagnetic forward-modeling code specifically engineered to simulate microwave reflectometry in magnetic confinement fusion plasmas (e.g., tokamaks). When evaluating plasma density profiles, shear flows, and microscopic turbulence, geometric optics (ray-tracing) often breaks down because the scale length of plasma fluctuations approaches or drops below the probing wave's wavelength ($\lambda$).

FeDoT bridges this gap by directly solving Maxwell's equations self-consistently alongside cold plasma fluid equations on a Yee-like staggered spatial grid. It operates in the **Extraordinary Mode (X-mode)** polarization, where the electric field of the probing wave is oriented perpendicular to the background toroidal magnetic field ($E \perp B_0$).

### Underlying Physics Equations
The code advances the components of the electromagnetic field ($\mathbf{E}, \mathbf{H}$) and the coherent plasma induced current density ($\mathbf{J}$) through time via an explicit numerical integration loop:

1. **Ampere's Law with Plasma Current Contribution:**
   $$\frac{\partial \mathbf{E}}{\partial t} = \frac{1}{\epsilon_0} \left( \nabla \times \mathbf{H} - \mathbf{J} \right)$$

2. **Faraday's Law:**
   $$\frac{\partial \mathbf{H}}{\partial t} = -\frac{1}{\mu_0} \nabla \times \mathbf{E}$$

3. **Cold Plasma Fluid Equation of Motion:**
   $$\frac{\partial \mathbf{J}}{\partial t} = \epsilon_0 \omega_p^2 \mathbf{E} - \mathbf{\omega}_c \times \mathbf{J}$$

Where:
*   $\omega_p = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}$ is the local electron plasma frequency.
*   $\omega_c = \frac{e B_0}{m_e}$ is the local electron cyclotron frequency.
*   The $1/R$ spatial decay of the toroidal magnetic field is modeled explicitly via `B_ampl / (x_start + distance)`.

---

## 2. Code Architecture & Core Modules
              +-----------------------------------+
              |        Simulation Steering        |
              |  (Configuration & Profile Type)  |
              +-----------------+-----------------+
                                |
                                v
              +-----------------------------------+
              |         class grid(...)           |
              |  (Instantiates 2D Complex Arrays) |
              +-----------------+-----------------+
                                |
                                v
              +-----------------------------------+
              |          PML(simul, ...)          |
              |  (Damps fields at grid boundaries)|
              +-----------------+-----------------+
                                |
                     [Loop for k in tmax]
                                |
                                v
              +-----------------------------------+
              |     next_step_numba(..., Jr, Jy)  |
              |  (Accelerated FDTD Core Solver)   |
              +-----------------------------------+

### `class grid`
Manages the memory allocations for the 2D spatial meshes. It stores fields as complex arrays (`complex128`) to track amplitude and phase information seamlessly, while managing background density profiles and plasma frequencies.
*   **Key Fields Matrix:** `Er` ($E_x$), `Ey` ($E_y$), `Hz`, `Jr` ($J_x$), `Jy` ($J_y$).
*   **Methods:** `clear()` resets matrices between iterations; `next_step()` recalculates antenna boundary input and launches the FDTD calculation loop.

### `@jit(...) def next_step_numba(...)`
The mathematical engine of FeDoT. This standalone function operates in strict `nopython=True` mode. It updates fields across a staggered spatial layout using central-difference approximations. It includes specialized conditional blocks that isolate boundary absorption zones (PML) from active core plasma grid updates.

### `def PML(...)`
Implements a **Perfectly Matched Layer** boundary around the boundary edges (`Xpml`, `Ypml`). It uses a polynomial grading profile (`sigmam * (depth/thickness)**n`) to match wave impedance and eliminate parasitic reflections off the simulation box boundaries.

---

## 3. Input Parameters Reference

| Variable | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `NX`, `NY` | Integer | `800`, `800` | Number of spatial grid points in the $X$ (radial) and $Y$ (poloidal) directions. |
| `Xpml`, `Ypml` | Integer | `100`, `100` | Grid depth allocated for boundary absorption layers. |
| `f0` | Float | `80E9` ($80\text{ GHz}$) | Initial microwave launcher probing frequency. |
| `n_max` | Float | `6e19` ($\text{m}^{-3}$) | Peak plasma density setting for analytical profile models. |
| `tmax` | Integer | `5500` | Maximum time steps executed per run. |
| `ind_phi_stable`| Integer | `5000` | The time-step index where initial transient wave activity clears and phase analysis begins. |
| `B_ampl` | Float | `2.5 * 1.65` | Magnetic field magnitude component used for tracking X-mode cutoffs. |
| `profile_type` | Integer | `1` (Linear) / `3` (HDF5) | **0:** Vacuum / **1:** Analytical linear ramp with edge smoothing / **3:** Numerical 2D turbulence map from HDF5 file (`Turbulence_map.h5`). |

---

## 4. Diagnostics & Analytical Routines

FeDoT features built-in diagnostics to confirm microwave propagation limits before executing full physical time steps:

*   **`N2(w, wc, wp)` / `Nx(w, wc, wp)`**: Computes the complex and real local refractive index maps for X-mode propagation:
    $$N_X^2 = 1 - \frac{\omega_p^2}{\omega^2} \left( \frac{\omega^2 - \omega_p^2}{\omega^2 - \omega_c^2 - \omega_p^2} \right)$$
*   **`COL(w, wc, wp)`**: Locates the spatial index of the non-linear cut-off layer boundary by scanning for indices where $N_X^2$ switches sign ($N^2 \cdot N_{next} < 0$).
*   **`phi_WKB(...)`**: Calculates an analytical reference dephasing via the Wentzel-Kramers-Brillouin (WKB) approximation to validate the code's full-wave field results:
    $$\phi_{WKB} = 2 \int \frac{\omega}{c} N_x(r) \, dr - \frac{\pi}{2}$$

---

## 5. Execution & Data Formats

### Running Simulations
Execute the framework directly from a terminal shell:
python3 FeDoT_reflectometry.py

Output Files

The code saves output arrays into the designated path_Working directory as standard NumPy binary matrices (.npy):

sourceEy_[Freq][Comment].npy: Time history tracking wave emission directly at the launch source.

Ey_ant_[Freq][Comment].npy: Spatiotemporal matrix tracking wave activity crossing the antenna plane.

Ey_final_[Freq][Comment].npy: A final 2D snapshot of the spatial wave field configuration.

amp_avg_ / amp_avg_ .npy: Scalar datasets tracking evaluated phase shifts and signal degradation markers due to turbulence interactions.

The code is optimized utilizing **Numba's Just-In-Time (`@jit`) compilation** to execute critical, nested finite-difference loops at native C-like speeds without sacrificing Python's scripting flexibility.
