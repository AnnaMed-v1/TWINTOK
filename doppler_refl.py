import scipy.signal as sig

def analyze_doppler_asymmetry(time_array, A_t, Phi_t, fs):
    """
    Calculates complex signal spectrum Z(t) = A(t)*exp(i*Phi(t)), extracts 
    Doppler shift and computes spectral asymmetry index.
    """
    # Build analytic complex synthetic signal
    Z_signal = A_t * np.exp(1j * Phi_t)
    
    # Power Spectral Density (two-sided for complex signal)
    f_axis, PSD = sig.welch(
        Z_signal, fs=fs, return_onesided=False, nperseg=256, scaling='spectrum'
    )
    
    f_axis = np.fft.fftshift(f_axis)
    PSD = np.fft.fftshift(PSD)
    
    # 1. Mean Doppler Shift Calculation
    f_doppler = np.sum(f_axis * PSD) / np.sum(PSD)
    
    # 2. Asymmetry Index: (P_pos - P_neg) / (P_pos + P_neg)
    pos_mask = f_axis > 0
    neg_mask = f_axis < 0
    
    P_pos = np.trapz(PSD[pos_mask], f_axis[pos_mask])
    P_neg = np.trapz(PSD[neg_mask], f_axis[neg_mask])
    
    asymmetry_index = (P_pos - P_neg) / (P_pos + P_neg)
    
    return f_axis, PSD, f_doppler, asymmetry_index
