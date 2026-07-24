import os
import random
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d, RectBivariateSpline
import imas

# Physics Constants
ELECTRON_MASS_EV = 511000.0  # eV
FARADAY_CONST = 2.62e-13     # Faraday rotation constant factor [rad / (T * m^-2)]


def discretize_los(p1, p2, n_points=300):
    """Discretizes a Line of Sight between two 3D points (R, Z, phi)."""
    x1, y1, z1 = p1.r * np.cos(p1.phi), p1.r * np.sin(p1.phi), p1.z
    x2, y2, z2 = p2.r * np.cos(p2.phi), p2.r * np.sin(p2.phi), p2.z

    s = np.linspace(0, 1, n_points)
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    z = z1 + s * (z2 - z1)

    r = np.sqrt(x**2 + y**2)
    path_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    dl = path_length / (n_points - 1)

    # Unit direction vectors along the path
    dx_dl = (x2 - x1) / path_length
    dy_dl = (y2 - y1) / path_length
    dz_dl = (z2 - z1) / path_length

    dr_dl = (x * dx_dl + y * dy_dl) / r

    return r, z, path_length, dl, dr_dl, dz_dl


def compute_synthetic_signal(
    eq_slice, core_slice, channels, 
    modality="interferometer", pass_mode="SP", 
    n_points=300, noise_level=0.0
):
    """
    Computes line-integrated electron density or Faraday rotation.
    
    Parameters:
        modality: 'interferometer' or 'polarimeter'
        pass_mode: 'SP' (Single Pass, factor=1) or 'DP' (Double Pass, factor=2)
    """
    # Double-pass multiplier (DP doubles path integration length)
    pass_factor = 2.0 if pass_mode.upper() == "DP" else 1.0

    # 2D Equilibrium Splines
    z_mesh = eq_slice.profiles_2d[0].z[0, :]
    r_mesh = eq_slice.profiles_2d[0].r[:, 0]
    psi_spline = RectBivariateSpline(r_mesh, z_mesh, eq_slice.profiles_2d[0].psi)

    # 1D Core Interpolators
    rho_eq = eq_slice.profiles_1d.rho_tor_norm
    psi_eq = eq_slice.profiles_1d.psi
    rho_core = core_slice.grid.rho_tor_norm

    psi_interp = interp1d(rho_eq, psi_eq, bounds_error=False, fill_value="extrapolate")
    psi1d_core = psi_interp(rho_core)

    f_ne = interp1d(psi1d_core, core_slice.electrons.density, bounds_error=False, fill_value=0.0)
    f_te = interp1d(psi1d_core, core_slice.electrons.temperature, bounds_error=False, fill_value=0.0)

    if modality == "polarimeter":
        b_r = getattr(eq_slice.profiles_2d[0], 'b_field_r', eq_slice.profiles_2d[0].b_r)
        b_z = getattr(eq_slice.profiles_2d[0], 'b_field_z', eq_slice.profiles_2d[0].b_z)
        br_spline = RectBivariateSpline(r_mesh, z_mesh, b_r)
        bz_spline = RectBivariateSpline(r_mesh, z_mesh, b_z)

    results = []

    for ch in channels:
        p1, p2 = ch.line_of_sight.first_point, ch.line_of_sight.second_point
        r_los, z_los, total_len, dl, dr_dl, dz_dl = discretize_los(p1, p2, n_points)

        psi_los = psi_spline(r_los, z_los, grid=False)
        ne_los = np.maximum(f_ne(psi_los), 0.0)
        te_los = np.maximum(f_te(psi_los), 0.0)

        # Relativistic temperature correction estimate
        te_err = pass_factor * 1.5 * np.sum(te_los * ne_los) * dl / ELECTRON_MASS_EV

        if modality == "interferometer":
            ne_line = pass_factor * np.sum(ne_los) * dl
            noisy_val = ne_line * (1.0 + noise_level * random.uniform(-1, 1)) - te_err
            
            results.append({
                "signal": noisy_val,
                "ne_line_avg": noisy_val / (pass_factor * total_len),
                "error_upper": te_err,
                "length": total_len * pass_factor
            })

        elif modality == "polarimeter":
            wavelength = ch.wavelength[0].value if isinstance(ch.wavelength, list) else ch.wavelength
            br_los = br_spline(r_los, z_los, grid=False)
            bz_los = bz_spline(r_los, z_los, grid=False)

            b_parallel = br_los * dr_dl + bz_los * dz_dl
            coeff = FARADAY_CONST * (wavelength ** 2)

            faraday_angle = pass_factor * coeff * np.sum(ne_los * b_parallel) * dl
            noisy_val = faraday_angle * (1.0 + noise_level * random.uniform(-1, 1))

            results.append({
                "signal": noisy_val,
                "error_upper": te_err * coeff,
                "length": total_len * pass_factor
            })

    return results


def run_synthetic_diagnostic(
    scenario_shot, scenario_run, md_shot, md_run,
    modality="interferometer", pass_mode="SP",
    user=os.environ['USER'], database="iter",
    n_points=300, noise=0.0, output_run=None
):
    """
    Main driver function to read Scenario & Machine Description, calculate signals, 
    and store output in IMAS IDS.
    """
    # 1. Fetch Geometry from Machine Description Shot
    md_entry = imas.DBEntry(imas.imasdef.MDSPLUS_BACKEND, database, md_shot, md_run, user)
    if md_entry.open()[0] != 0:
        raise RuntimeError(f"Failed to open Machine Description: shot {md_shot}, run {md_run}")

    geom_ids = md_entry.get(modality)
    md_entry.close()

    # 2. Fetch Scenario Data
    scenario_entry = imas.DBEntry(imas.imasdef.MDSPLUS_BACKEND, database, scenario_shot, scenario_run, user)
    if scenario_entry.open()[0] != 0:
        raise RuntimeError(f"Failed to open Scenario: shot {scenario_shot}, run {scenario_run}")

    eq_ids = scenario_entry.get("equilibrium")
    core_ids = scenario_entry.get("core_profiles")
    scenario_entry.close()

    # Initialize Output IDS
    out_ids = getattr(imas, modality)()
    out_ids.ids_properties.comment = f"Synthetic {modality} ({pass_mode} mode)"
    out_ids.ids_properties.homogeneous_time = 1
    out_ids.ids_properties.provider = user
    out_ids.ids_properties.creation_date = datetime.now().strftime("%Y-%m-%d")

    n_channels = len(geom_ids.channel)
    out_ids.channel.resize(n_channels)

    for ch_idx, ch_geom in enumerate(geom_ids.channel):
        out_ids.channel[ch_idx].name = ch_geom.name
        out_ids.channel[ch_idx].identifier = ch_geom.identifier
        out_ids.channel[ch_idx].line_of_sight = ch_geom.line_of_sight

    # Process Time Slices
    out_ids.time = eq_ids.time

    for t_idx, t_val in enumerate(eq_ids.time):
        eq_slice = eq_ids.time_slice[t_idx]
        core_slice = core_ids.profiles_1d[t_idx]

        calc_results = compute_synthetic_signal(
            eq_slice, core_slice, geom_ids.channel, 
            modality=modality, pass_mode=pass_mode,
            n_points=n_points, noise_level=noise
        )

        for ch_idx, res in enumerate(calc_results):
            ch = out_ids.channel[ch_idx]
            if modality == "interferometer":
                ch.n_e_line.data.append(res["signal"])
                ch.n_e_line.time.append(t_val)
                ch.n_e_line_average.data.append(res["ne_line_avg"])
                ch.n_e_line_average.time.append(t_val)
                ch.n_e_line.data_error_upper.append(res["error_upper"])
            elif modality == "polarimeter":
                ch.faraday_angle.data.append(res["signal"])
                ch.faraday_angle.time.append(t_val)
                ch.faraday_angle.data_error_upper.append(res["error_upper"])

    if output_run is not None:
        out_entry = imas.DBEntry(imas.imasdef.MDSPLUS_BACKEND, database, scenario_shot, output_run, user)
        out_entry.create()
        out_entry.put(out_ids)
        out_entry.close()

    return out_ids


if __name__ == "__main__":
    # Example usage:
    # 1. TIP (Toroidal Interferometer: Double Pass 'DP')
    tip_data = run_synthetic_diagnostic(
        scenario_shot=100001, scenario_run=1, md_shot=1, md_run=0,
        modality="interferometer", pass_mode="DP"
    )

    # 2. DIP (Single Pass 'SP')
    dip_data = run_synthetic_diagnostic(
        scenario_shot=100001, scenario_run=1, md_shot=1, md_run=0,
        modality="interferometer", pass_mode="SP"
    )

    # 3. POP (Polarimeter: Single Pass 'SP')
    pop_data = run_synthetic_diagnostic(
        scenario_shot=100001, scenario_run=1, md_shot=1, md_run=0,
        modality="polarimeter", pass_mode="SP"
    )
