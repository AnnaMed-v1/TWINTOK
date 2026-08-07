import scipy.stats as stats

def compute_sol_phase_skewness(phase_time_series):
    """
    phase_time_series: 2D array of shape (N_radial_positions, N_time_points)
    Returns phase variance and skewness profiles across radial positions.
    """
    N_radii = phase_time_series.shape[0]
    skewness_profile = np.zeros(N_radii)
    std_profile = np.zeros(N_radii)
    
    for i in range(N_radii):
        phi_t = phase_time_series[i, :]
        
        # Detrend long-term drifts to isolate fluctuations
        phi_fluc = sig.detrend(phi_t)
        
        std_profile[i] = np.std(phi_fluc)
        # Calculate third standardized moment (skewness)
        skewness_profile[i] = stats.skew(phi_fluc)
        
    return std_profile, skewness_profile
