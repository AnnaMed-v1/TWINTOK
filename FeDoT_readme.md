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

The code is optimized utilizing **Numba's Just-In-Time (`@jit`) compilation** to execute critical, nested finite-difference loops at native C-like speeds without sacrificing Python's scripting flexibility.
