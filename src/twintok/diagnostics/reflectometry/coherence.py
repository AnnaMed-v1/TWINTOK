def compute_radial_coherence_length(Z_signals, R_cutoffs):
    """
    Z_signals: 2D array of shape (N_radii, 1024_time_points)
    R_cutoffs: 1D array of radial positions corresponding to probing frequencies
    """
    N_radii, N_time = Z_signals.shape
    coherence_matrix = np.zeros((N_radii, N_radii))
    
    # 1. Calculate Cross-Coherence Matrix
    for i in range(N_radii):
        for j in range(N_radii):
            sig_i = Z_signals[i, :] - np.mean(Z_signals[i, :])
            sig_j = Z_signals[j, :] - np.mean(Z_signals[j, :])
            
            numerator = np.abs(np.mean(sig_i * np.conj(sig_j)))**2
            denominator = np.mean(np.abs(sig_i)**2) * np.mean(np.abs(sig_j)**2)
            
            coherence_matrix[i, j] = numerator / denominator if denominator > 0 else 0.0
            
    # 2. Extract Radial Coherence Length L_c at each reference radius
    L_c = np.zeros(N_radii)
    
    for i in range(N_radii):
        r_ref = R_cutoffs[i]
        coherence_profile = coherence_matrix[i, :]
        dr = np.abs(R_cutoffs - r_ref)
        
        # Fit decay function: gamma(dr) = exp(-dr / L_c)
        # Avoid log(0) by filtering points where coherence > 0.1
        valid = coherence_profile > 0.1
        if np.sum(valid) > 2:
            p = np.polyfit(dr[valid], np.log(coherence_profile[valid]), 1)
            L_c[i] = -1.0 / p[0] if p[0] < 0 else 0.0
        else:
            L_c[i] = 0.0
            
    return coherence_matrix, L_c
