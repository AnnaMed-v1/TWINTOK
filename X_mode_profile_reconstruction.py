# -*- coding: utf-8 -*-
"""

@author: Anna Glasser
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
# =============================================================================
# DISCRETIZATION CONTROL PARAMETERS
# =============================================================================
# Rule 1: High frequency resolution prevents integration step errors in BC.
N_FREQ_POINTS = 1000       # Channels for the probing sweep array (freqs_test)

# Rule 2: High B-field spatial resolution prevents numeric noise in the loop.
N_B_FIELD_POINTS = 3000    # Points for the magnetic field mesh (R_B_profile)

# Rule 3: Dense internal search grid ensures the exact cutoff root is captured.
N_CUTOFF_SEARCH = 5000     # Internal points for the root-finding grid (R_search)

# Reconstruction internal grid resolution.
N_RECON_GRID = 100        # Spatial points for the internal edge calculation (R_grid)


# -------------------------------------------------------------------------
# Physical Constants 
# -------------------------------------------------------------------------
me   = 0.91094e-30     # Electron mass (kg)
e    = 0.16022e-18     # Electron charge (C) 
eps0 = 0.88542e-11     # Vacuum permittivity (F/m)
c    = 3.0e8           # Speed of light (m/s)

def density_profile_bc(phi_f_VW, F_VW, F0, B, R_B, show_plots=False):
    """
    Reconstructs the plasma density profile from X-mode reflectometry
    using the Bottollier-Curtet algorithm.
    """
  
    # -------------------------------------------------------------------------
    # Data Filtering and Interpolation
    # -------------------------------------------------------------------------
    # Only use frequencies greater than the initial cutoff F0
    valid_indices = F_VW > F0
    F_VW = F_VW[valid_indices]
    phi_f_VW = phi_f_VW[valid_indices]
    
    npt_calcul = len(F_VW)  # Target number of reconstruction points
    F_r = np.linspace(F_VW[0], F_VW[-1], npt_calcul)
    
    # Interpolate phase onto the new frequency grid and invert sign
    interp_phi = interp1d(F_VW, phi_f_VW, kind='linear', fill_value="extrapolate")
    phi_r = -interp_phi(F_r)
    f = F_r * 1e9  # Convert GHz to Hz
    
    # -------------------------------------------------------------------------
    # Initialization & First Edge Cut-Off Point
    # -------------------------------------------------------------------------
    # Create a spatial grid between tokamak wall coordinates to find edge B-field
    R_grid = np.linspace(1.2, 2.3, N_RECON_GRID)
    
    # Using cubic interpolation to ensure perfectly smooth gradients for the loop steps
    interp_B = interp1d(R_B, B, kind='cubic', fill_value="extrapolate")
    B_lin = interp_B(R_grid)
    
    # Electron cyclotron frequency profile (GHz)
    Fce = (e * B_lin) / (2 * np.pi * me) * 1e-9
    
    # Find the major radius rx[0] corresponding to the initial cutoff frequency F0
    interp_R_from_Fce = interp1d(Fce, R_grid, kind='linear', fill_value="extrapolate")
    
    # Initialize output profiles
    rx = np.zeros(npt_calcul)
    nex = np.zeros(npt_calcul)
    Nx = np.zeros(npt_calcul)
    Fcex = np.zeros(npt_calcul)
    Fpex = np.zeros(npt_calcul)
    Sn = np.zeros(npt_calcul)
    
    rx[0] = min(interp_R_from_Fce(F0), 2.2) # To stay inside AUG tokamak
    Bx_0 = interp_B(rx[0])
    
    # Correct phase for vacuum path length / vessel geometry
    phi = phi_r + 4 * np.pi * f * (1.0432 - (2.3 - np.sqrt((2.3 - rx[0])**2 + 4e-4))) / c
    phi = np.abs(phi - np.max(phi))
    
    # First point boundary conditions (Density is 0 at the extreme edge)
    Fpex[0] = 0.0
    nex[0] = 0.0
    Fcex[0] = f[0]
    
    # Refractive index step for the 2nd frequency layer
    Nx[0] = np.sqrt(1 - Fpex[0]**2 * (f[1]**2 - Fpex[0]**2) / 
                    (f[1]**2 * (f[1]**2 - Fpex[0]**2 - Fcex[0]**2)))
    
    # Calculate the second radial position using the 3/4 edge correction
    rx[1] = np.sqrt(rx[0]**2 + 4e-4) - (3/4) * np.abs(c / (4 * np.pi * f[1]) * (1 / Nx[0]) * phi[1])
    rx[1] = min(2.3 - (np.sqrt(2.3 - rx[1])**2 - 4e-4), 2.3)
    
    Bx_1 = interp_B(rx[1])
    Fcex[1] = (e * Bx_1) / (2 * np.pi * me)
    Fpex[1] = np.abs(np.sqrt(f[1]**2 - f[1] * Fcex[1]))
    nex[1] = np.abs(4 * np.pi**2 * me * eps0 * (Fpex[1])**2 / e**2)
    
    Sn[0] = 0.0  # Phase contribution from previous layers initialized to 0
    
    # -------------------------------------------------------------------------
    # 4. Main Reconstruction Loop (Bottollier-Curtet Iteration)
    # -------------------------------------------------------------------------
    if show_plots:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    nptx = len(f)
    for i in range(1, nptx - 1):
        # Vectorized calculation of refractive indices of all previous layers 
        # evaluated at the current higher probing frequency f[i+1]
        Fpex_past = Fpex[:i+1]
        Fcex_past = Fcex[:i+1]
        
        num = Fpex_past**2 * (f[i+1]**2 - Fpex_past**2)
        den = f[i+1]**2 * (f[i+1]**2 - Fpex_past**2 - Fcex_past**2)
        
        Nx_1 = np.minimum(np.abs(np.sqrt(1 - num / den)), 1.0)
        
        # Trapezoidal rule optimization using median midpoints
        K = 0.5 * (Nx_1[:-1] + Nx_1[1:])
        I = np.sum(K * np.diff(rx[:i+1]))
        
        # Accumulated internal phase delay up to the current layer boundary
        Sn[i] = -(4 * np.pi * f[i+1] / c) * I + np.pi / 2
        
        # Local refractive index at the edge of the known layer
        Nx[i] = np.minimum(np.abs(np.sqrt(1 - num[-1] / den[-1])), 1.0)
        
        # Calculate the next radial step with the 3/4 singularity correction factor
        rx[i+1] = rx[i] - (3/4) * (c / (4 * np.pi * f[i+1])) * 2 * (phi[i+1] - Sn[i] + np.pi/2) / Nx[i]
        rx[i+1] = min(rx[i+1], 2.3)
        
        # Update Plasma Parameters at the new position rx[i+1]
        Bx_next = interp_B(rx[i+1])
        Fcex[i+1] = (e * Bx_next) / (2 * np.pi * me)
        Fpex[i+1] = np.sqrt(f[i+1]**2 - f[i+1] * Fcex[i+1])
        nex[i+1] = np.abs(4 * np.pi**2 * me * eps0 * (Fpex[i+1])**2 / e**2)
        
        # Real-time plotting block 
        if show_plots and i % 10 == 0:
            ax1.clear()
            ax1.plot(rx[:i+2], nex[:i+2], 'b-')
            ax1.set_xlabel('Radius (m)')
            ax1.set_ylabel('Density ($m^{-3}$)')
            ax1.set_title(f'Frequency: {f[i]*1e-9:.2f} GHz')
            
            ax2.clear()
            # Gradient approximation
            dphi_dr = np.gradient(phi[:i+2])
            ax2.plot(rx[:i+2], dphi_dr, 'r-')
            ax2.set_xlabel('Radius (m)')
            ax2.set_ylabel('dphi (rad)')
            
            plt.pause(0.01)
            
    if show_plots:
        plt.ioff()
        plt.show()
        
    return rx, nex

def calculate_theoretical_phase(freqs_ghz, target_ne_func, B_field_func, R_edge, R_backwall=1.0432):
    """
    Calculates the theoretical X-mode phase profile phi(f) for a given density profile.
    """
    phi_theoretical = np.zeros_like(freqs_ghz)
    
    for idx, f_ghz in enumerate(freqs_ghz):
        omega = 2 * np.pi * f_ghz * 1e9
        
        # 1. Find the reflection (cutoff) radius for this specific frequency
        def cutoff_condition(R):
            R = np.asarray(R)
            result = np.full_like(R, omega, dtype=float)
            inside_plasma = R <= R_edge
            if np.any(inside_plasma):
                ne = np.array([target_ne_func(r) for r in R[inside_plasma]])
                B = np.array([B_field_func(r) for r in R[inside_plasma]])
                omega_pe2 = (ne * e**2) / (eps0 * me)
                omega_ce = (e * B) / me
                cutoff_val = omega**2 - omega * omega_ce - omega_pe2
                result[inside_plasma] = cutoff_val
            return result
        
        # Search for cutoff from the edge moving inward using the globally controlled parameter
        R_search = np.linspace(R_edge, 1.3, N_CUTOFF_SEARCH)
        cond_values = cutoff_condition(R_search)
        
        if np.all(cond_values > 0):
            R_cutoff = R_backwall
        else:
            zero_crossings = np.where(np.diff(np.sign(cond_values)))[0]
            R_cutoff = R_search[zero_crossings[0]] if len(zero_crossings) > 0 else R_backwall

        # 2. Integrate the refractive index from the cutoff layer out to the backwall
        def refractive_index_integrand(R):
            if R >= R_edge:
                return 1.0 
            
            ne = target_ne_func(R)
            B = B_field_func(R)
            
            omega_pe2 = (ne * e**2) / (eps0 * me)
            omega_ce = (e * B) / me
            
            denom = omega**2 - omega_pe2 - omega_ce**2
            if abs(denom) < 1e-6: denom = 1e-6
                
            num = omega_pe2 * (omega**2 - omega_pe2)
            val = 1.0 - (num / (omega**2 * denom))
            
            return np.sqrt(max(0.0, val))

        if R_cutoff < R_edge:
            plasma_integral, _ = quad(refractive_index_integrand, R_cutoff, R_edge, epsabs=1e-3, limit=100)
            vacuum_integral = 1.0 * (R_backwall - R_edge)
            total_integral = plasma_integral + vacuum_integral
        else:
            total_integral = 1.0 * (R_backwall - R_cutoff)

        phi_theoretical[idx] = (2 * omega / c) * total_integral - np.pi/2

    return phi_theoretical

# =============================================================================
# SYNTHETIC TEST SETUP
# =============================================================================

# 1. Define Frequencies using global sweep resolution
freqs_test = np.linspace(50, 105, N_FREQ_POINTS) 

# 2. Define 1/R Toroidal Magnetic Field (B = 2.5T at R = 1.65m)
R_axis = 1.65
B_axis = 2.5
def B_field(R): return B_axis * R_axis / R

R_edge_plasma = 2.12
R_B_profile = np.linspace(1.0, 2.5, N_B_FIELD_POINTS)
B_profile = B_field(R_B_profile)

B_at_edge = B_field(R_edge_plasma)
f0_physical = (e * B_at_edge) / (2 * np.pi * me) * 1e-9
f0_test = f0_physical+0.5

# 3. Create a synthetic phase shift phi(f) 
def true_density_profile(R):
    if R > R_edge_plasma or R < 1.2: 
        return 0.0
    n0 = 4.5e19 
    return n0 * (1 - ((R - R_axis) / (R_edge_plasma - R_axis))**2)

print("Calculating theoretical phase profile from analytical density")
raw_phase = calculate_theoretical_phase(freqs_test, true_density_profile, B_field, R_edge_plasma)

# 4. Run the Reconstruction
R_recon, ne_recon = density_profile_bc(raw_phase, freqs_test, f0_test, B_profile, R_B_profile)

# 5. Plot the Reconstructed Density Profile
plt.figure(figsize=(9, 5))
R_grid_fine = np.linspace(1.5, 2.15, 200)
ne_true_values = [true_density_profile(r) for r in R_grid_fine]

plt.plot(R_grid_fine, ne_true_values, 'k--', linewidth=2, label='Original profile')
plt.plot(R_recon, ne_recon, 'crimson', linewidth=2.5, label='Reconstructed profile')
plt.axvline(R_axis, color='gray', linestyle=':', label='Magnetic axis')
plt.xlabel('Major radius $R$ (m)', fontsize=12)
plt.ylabel('Electron density $n_e$ ($m^{-3}$)', fontsize=12)
plt.title('Forward phase calculation vs. profile reconstruction', fontsize=13)
plt.xlim(1.5, 2.2)
plt.gca().invert_xaxis()
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()
