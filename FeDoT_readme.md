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

The code saves output arrays into the designated path_Working directory as standard NumPy binary matrices (.npy)

# X-Mode Profile Reconstruction Routine (`X_mode_profile_reconstruction.py`)
*An Inversion Code for Microwave Reflectometry using the Bottollier-Curtet Algorithm*

---

## 1. Overview & Algorithmic Framework
While FeDoT acts as the forward-modeling engine, this companion routine solves the inverse problem: **reconstructing the 1D electron density profile from the phase shift measured by X-mode microwave reflectometry.** 

Because X-mode propagation depends non-linearly on both the plasma density and the spatially varying toroidal magnetic field ($B(R) \propto 1/R$), simple Abel inversion breaks down. This script implements the **Bottollier-Curtet (BC) integration algorithm**, which steps radially inward from the plasma edge, updating the local refractive index and correcting for the phase-delay singularity at the cutoff layer at each frequency channel.

---

## 2. Mathematical Workflow & Singularity Handling

The routine evaluates the round-trip phase shift $\phi(\omega)$ accumulated from the vacuum boundary to the reflection layer $R_c$:

$$\phi(\omega) = \frac{2\omega}{c} \int_{R_c}^{R_{\text{edge}}} N_X(r, \omega) \, dr - \frac{\pi}{2}$$

The inversion loop handles the core physical challenges via three main steps:

### A. Phase Extraction & Geometry Correction
The raw phase is modified to isolate the true plasma contribution by subtracting the vacuum path length and the specific vessel coordinates.
### B. The Bottollier-Curtet Iteration Step
The internal phase accumulation up to the current layer ($S_n$) is calculated numerically using a trapezoidal rule optimization across all previously resolved plasma layers:

$$S_n(f_{i+1}) = -\frac{4\pi f_{i+1}}{c} \sum_{j=1}^{i} ( \frac{N_X(r_j, f_{i+1}) + N_X(r_{j+1}, f_{i+1})}{2} \cdot (r_{j+1} - r_j)) + \frac{\pi}{2}$$

### C. The 3/4 Singularity Factor
Near the cutoff layer, the refractive index approaches zero ($N_X \to 0$), introducing a mathematical singularity. The code applies a localized analytic correction factor of $3/4$ to accurately project the next radial position step:

$$r_{i+1} = r_i - \frac{3}{4} \left( \frac{c}{4\pi f_{i+1}} \right) \frac{2 \left(\phi_{i+1} - S_n[i] + \frac{\pi}{2}\right)}{N_X(r_i, f_{i+1})}$$

---

## 3. Core Functions Reference

### `density_profile_bc(...)`
The primary inversion function that accepts raw phase and frequency arrays alongside the magnetic field configuration to produce the localized spatial coordinates and electron densities.
*   **Key Inputs:** Phase array (`phi_f_VW`), frequency sweep array in GHz (`F_VW`), first cutoff frequency (`F0`), magnetic profile vector (`B`), spatial grid (`R_B`).
*   **Outputs:** Real-space major radius array (`rx`), electron density profile array (`nex`).
*   **Feature:** Includes a `show_plots=True` debug switch that provides a real-time animated layout showing the profile building point-by-point alongside its phase gradient.

### `calculate_theoretical_phase(...)`
Generates synthetic data to validate the inversion algorithm. It solves the forward path problem by establishing an independent, fine-meshed root-finding search grid (`N_CUTOFF_SEARCH`) to pin down the exact cutoff coordinates before performing numerical quadrature integration (`scipy.integrate.quad`) of the X-mode refractive index.

---

## 4. Discretization Control & Settings

The initialization variables are isolated at the top of the routine to guarantee numerical stability across complex plasma profile gradients:

| Parameter | Default Value | Computational Target |
| :--- | :--- | :--- |
| `N_FREQ_POINTS` | `1000` | Grid channels for the probing sweep array; higher density mitigates integration step errors. |
| `N_B_FIELD_POINTS` | `3000` | Matrix size for the background magnetic field mesh; eliminates discretization noise in spline interpolations. |
| `N_CUTOFF_SEARCH` | `5000` | Precision scale for the internal forward-calculation root finder; guarantees exact interception of the cutoff boundary. |
| `N_RECON_GRID` | `100` | Smooth spatial points mapped across tokamak edge bounds to initialize boundary steps. |

Ey_final.npy: A final 2D snapshot of the spatial wave field configuration.

amp_avg_ / amp_avg_ .npy: Scalar datasets tracking evaluated phase shifts and signal degradation markers due to turbulence interactions.

The code is optimized utilizing **Numba's Just-In-Time (`@jit`) compilation** to execute critical, nested finite-difference loops at native C-like speeds without sacrificing Python's scripting flexibility.
