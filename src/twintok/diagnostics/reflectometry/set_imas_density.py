import numpy as np
import scipy.interpolate as interp

def prepare_imas_2d_grid(core_ids, edge_ids, grid_cfg):
    """
    Combines IMAS core and edge profiles, maps to a 2D grid, applies border
    tapering to 0 density, and builds the magnetic field map B(R,Z).
    """
    NX, NY = grid_cfg['NX'], grid_cfg['NY']
    dx, dy = grid_cfg['dx'], grid_cfg['dy']
    Xpml, Ypml = grid_cfg['Xpml'], grid_cfg['Ypml']
    
    # 1. Combine core and edge density profiles along flux coordinate rho
    rho_core = core_ids.profiles_1d[0].grid.rho_tor_norm
    ne_core = core_ids.profiles_1d[0].electrons.density
    
    rho_edge = edge_ids.profiles_1d[0].grid.rho_tor_norm
    ne_edge = edge_ids.profiles_1d[0].electrons.density
    
    # Concatenate and remove duplicates for monotonic interpolation grid
    rho_combined = np.concatenate([rho_core, rho_edge])
    ne_combined = np.concatenate([ne_core, ne_edge])
    sort_idx = np.argsort(rho_combined)
    rho_combined, ne_combined = rho_combined[sort_idx], ne_combined[sort_idx]
    _, unique_idx = np.unique(rho_combined, return_index=True)
    
    ne_interp = interp.interp1d(
        rho_combined[unique_idx], ne_combined[unique_idx], 
        bounds_error=False, fill_value=0.0
    )
    
    # 2. Build 2D Spatial Grid (R, Z)
    R_grid = grid_cfg['R_start'] + np.arange(NX) * dx
    Z_grid = grid_cfg['Z_center'] + (np.arange(NY) - NY / 2.0) * dy
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')
    
    # Normalized radial coordinate approximation (normalized to minor radius a)
    rho_2D = np.sqrt(((RR - grid_cfg['R0']) / grid_cfg['a'])**2 + (ZZ / grid_cfg['a'])**2)
    density_2D = ne_interp(rho_2D)
    density_2D = np.nan_to_num(density_2D, nan=0.0)
    
    # 3. Smooth Tapering to 0 at simulation box edges (PML interface)
    x_idx = np.arange(NX)
    # Taper function dropping smoothly to 0 inside the PML zones
    taper_x = 0.5 * (1.0 + np.tanh((x_idx - Xpml) / 10.0)) * \
              0.5 * (1.0 + np.tanh((NX - Xpml - x_idx) / 10.0))
    
    y_idx = np.arange(NY)
    taper_y = 0.5 * (1.0 + np.tanh((y_idx - Ypml) / 10.0)) * \
              0.5 * (1.0 + np.tanh((NY - Ypml - y_idx) / 10.0))
    
    taper_2D = np.outer(taper_x, taper_y)
    density_2D *= taper_2D
    
    # 4. Magnetic Field for X-mode propagation (1/R vacuum field + axis value)
    B_2D = grid_cfg['B0'] * (grid_cfg['R0'] / RR)
    
    return RR, ZZ, density_2D, B_2D
