import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import curve_fit
from pathlib import Path


# Wavelengths, all vacuum. Copied from Lime
ha_n2 = [6549.850000, 6564.610000, 6585.280000]
hb = [4862.683000]
o3_b = [5008.240000]

# Continuum windows
ha_n2_window = [(6450, 6500), (6620, 6670)]
hb_window = [(4800, 4850), (4890, 4940)]
o3_b_window = [(4890, 4940), (5030, 5080)]

LINE_GROUPS = {
    "ha_n2": (ha_n2, ha_n2_window),
    "hb": (hb, hb_window),
    "o3_b": (o3_b, o3_b_window),
}

emps = Table.read("../data/emp-candidates.fits", format="fits").to_pandas()


def make_gaussian_model(centers):
    centers = np.asarray(centers)
    n_lines = len(centers)

    def model(wave, z, sigma, *args, m=None, n=None):
        wave = np.asarray(wave)

        if len(args) == n_lines + 2:
            fluxes = np.asarray(args[:n_lines])
            m = args[n_lines] if m is None else m
            n = args[n_lines + 1] if n is None else n
        elif len(args) == n_lines and m is not None and n is not None:
            fluxes = np.asarray(args)
        else:
            raise ValueError(
                f"Expected {n_lines} fluxes and 2 continuum parameters (m, n), got {len(args)} positional arguments."
            )

        # Linear continuum
        continuum = m * wave + n

        # Gaussians evaluated across all lines: shape (n_lines, len(wave))
        # Broadcasting: wave[None, :] - (centers * (1 + z))[:, None]
        shifted_centers = (centers * (1 + z))[:, None]
        exponent = -0.5 * ((wave[None, :] - shifted_centers) / sigma) ** 2
        gaussians = np.sum((fluxes[:, None] / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent), axis=0)

        return continuum + gaussians

    return model


def initial_guesses(z, centers, continuum_windows, data, sigma=1.5):
    """
    Estimate initial parameters for the multi-Gaussian emission line + linear continuum fit.

    Parameters
    ----------
    z : float
        Initial redshift guess.
    centers : list or array-like
        Rest-frame line central wavelengths (in Angstroms).
    continuum_windows : list of 2 tuples/lists
        Two rest-frame wavelength windows [(w1_min, w1_max), (w2_min, w2_max)]
        used to estimate the continuum.
    data : astropy.table.Table, pandas.DataFrame, dict, or tuple of (wave, flux)
        Observed spectrum containing wavelength and flux.
    sigma : float, optional
        Initial line width guess in Angstroms (default is 1.5).

    Returns
    -------
    p0 : list
        Initial parameter list: [z, sigma, flux_1, ..., flux_N, m, n]
    """
    # Extract wavelength and flux from input data
    if isinstance(data, (tuple, list)) and len(data) == 2:
        wave, flux = np.asarray(data[0]), np.asarray(data[1])
    elif hasattr(data, '__getitem__'):
        wave = np.asarray(data['wavelength'] if 'wavelength' in data else data['wave'])
        flux = np.asarray(data['flux'])
    else:
        raise ValueError("data must be a table/dict with 'wavelength'/'flux' or a (wave, flux) tuple.")

    centers = np.asarray(centers)
    rest_wave = wave / (1 + z)

    # Continuum estimation from the two rest-frame windows
    w1, w2 = continuum_windows
    mask_w1 = (rest_wave >= w1[0]) & (rest_wave <= w1[1])
    mask_w2 = (rest_wave >= w2[0]) & (rest_wave <= w2[1])

    # Mean observed wavelength and flux in each window
    obs_center_1 = np.mean(wave[mask_w1]) if np.any(mask_w1) else np.mean(w1) * (1 + z)
    obs_center_2 = np.mean(wave[mask_w2]) if np.any(mask_w2) else np.mean(w2) * (1 + z)
    flux_w1 = np.mean(flux[mask_w1]) if np.any(mask_w1) else 0.0
    flux_w2 = np.mean(flux[mask_w2]) if np.any(mask_w2) else 0.0

    # Linear continuum: flux_cont = m * wave_obs + n
    m = (flux_w2 - flux_w1) / (obs_center_2 - obs_center_1)
    n = flux_w1 - m * obs_center_1

    # Estimate line fluxes
    fluxes = []
    # Search half-width around each line center (in rest-frame Angstroms)
    search_half_width = max(3.0 * sigma / (1 + z), 5.0)

    for c in centers:
        line_mask = (rest_wave >= c - search_half_width) & (rest_wave <= c + search_half_width)
        c_obs = c * (1 + z)
        cont_at_line = m * c_obs + n

        if np.any(line_mask):
            peak_flux_density = np.max(flux[line_mask]) - cont_at_line
            # Total integrated flux = peak_height * sigma * sqrt(2*pi)
            flux_guess = max(peak_flux_density, 0.1) * sigma * np.sqrt(2 * np.pi)
        else:
            flux_guess = 1.0

        fluxes.append(float(flux_guess))

    p0 = [float(z), float(sigma)] + fluxes + [float(m), float(n)]
    return p0


def get_bounds(n_lines):
    lower = [-np.inf, 0.0] + [0.0] * n_lines + [-np.inf, -np.inf]
    upper = [np.inf, np.inf] + [np.inf] * n_lines + [np.inf, np.inf]
    return (lower, upper)


def fit_spectrum_lines(spec, z):
    """
    Fit the ha_n2, hb and o3_b emission-line complexes for a single spectrum.

    Parameters
    ----------
    spec : astropy.table.Table
        Spectrum data with 'wavelength', 'flux' and 'ivar' columns.
    z : float
        Redshift to use as the fit's initial guess.

    Returns
    -------
    results : dict
        Mapping from line-group name ("ha_n2", "hb", "o3_b") to a dict with
        keys "popt", "pcov", "perr" (1-sigma errors) and "param_names".
    """
    results = {}
    for name, (centers, window) in LINE_GROUPS.items():
        mask = (
            (spec["wavelength"] / (1 + z) >= window[0][0])
            & (spec["wavelength"] / (1 + z) <= window[1][1])
            & (spec["ivar"] > 0)
        )
        sigma = 1.0 / np.sqrt(spec["ivar"][mask])
        model = make_gaussian_model(centers)
        p0 = initial_guesses(z, centers, window, (spec["wavelength"], spec["flux"]))

        popt, pcov = curve_fit(
            model,
            spec["wavelength"][mask],
            spec["flux"][mask],
            p0=p0,
            sigma=sigma,
            absolute_sigma=True,
            bounds=get_bounds(len(centers)),
        )

        param_names = (
            ["z", "sigma"]
            + [f"flux_{i}" for i in range(len(centers))]
            + ["m", "n"]
        )
        results[name] = {
            "popt": popt,
            "pcov": pcov,
            "perr": np.sqrt(np.diag(pcov)),
            "param_names": param_names,
        }

    return results


def plot_fit_results(result, spec, z, pad_frac=0.3, fignum=None):
    """
    Plot the data and best-fit model for each line group in `result`, one pane per line.

    Parameters
    ----------
    result : dict
        Output of `fit_spectrum_lines`.
    spec : astropy.table.Table
        Spectrum data with 'wavelength' and 'flux' columns.
    z : float
        Redshift used to convert to rest-frame wavelength for plotting.
    pad_frac : float, optional
        Fraction of the continuum-window span to pad the x-range with on each side.
    fignum : int, optional
        Figure number to use for the plot.
    """

    fig, axes = plt.subplots(1, len(LINE_GROUPS), figsize=(3 * len(LINE_GROUPS), 2), num=fignum, clear=True)
    for ax, name in zip(axes, ['hb', 'o3_b', 'ha_n2']):
        centers, window = LINE_GROUPS[name]
        popt = result[name]["popt"]
        z = popt[0]
        rest_wave = spec["wavelength"] / (1 + z)

        w_lo, w_hi = window[0][0], window[1][1]
        pad = pad_frac * (w_hi - w_lo)
        x_lo, x_hi = w_lo - pad, w_hi + pad

        mask = (rest_wave >= x_lo) & (rest_wave <= x_hi)
        model = make_gaussian_model(centers)
        flux_model = model(spec["wavelength"][mask], *popt)

        ax.plot(rest_wave[mask], spec["flux"][mask], color="0.6", lw=1.5, label="data")
        ax.plot(rest_wave[mask], flux_model, color="crimson", lw=0.7, label="fit")
        for c in centers:
            ax.axvline(c, color="tab:orange", ls=":", lw=0.7)
        for w1, w2 in window:
            ax.axvspan(w1, w2, color="tab:blue", alpha=0.1)

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Rest-frame wavelength (Å)")
        ax.set_title(name)

    axes[0].set_ylabel("Flux")
    axes[0].legend()
    fig.tight_layout()
    return fig, axes


def report_and_plot(idx, emps, fignum=None):
    targetid = emps['TARGETID'].iloc[idx]
    spec = Table.read(f'../data/spectra/{targetid}.fits', format='fits')
    z = emps['Z'].iloc[idx]
    N2 = emps['N2'].iloc[idx]
    R3 = emps['R3'].iloc[idx]
    ston = emps['ston'].iloc[idx]

    result = fit_spectrum_lines(spec, z)

    print("My estimate of NII S/N: ", result['ha_n2']['popt'][4] / result['ha_n2']['perr'][4])
    print("EmFit estimate of  S/N: ", ston)
    print("")
    print("My estimate of N2: ", np.log10(result['ha_n2']['popt'][4] / result['ha_n2']['popt'][3]))
    print("EmFit estimate   : ", N2)
    print("")
    print("My estimate of R3: ", np.log10(result['o3_b']['popt'][2] / result['hb']['popt'][2]))
    print("EmFit estimate   : ", R3)

    plot_fit_results(result, spec, z, fignum=fignum)
    plt.show()
    return result


def plot_ha_nii(idx, emps, fignum=None):
    targetid = emps['TARGETID'].iloc[idx]
    spec = Table.read(f'../data/spectra/{targetid}.fits', format='fits')
    z = emps['Z'].iloc[idx]

    result = fit_spectrum_lines(spec, z)

    ha_n2_result = result['ha_n2']
    sigma = ha_n2_result['popt'][1]
    nii_flux = ha_n2_result['popt'][4]
    nii_flux_error = ha_n2_result['perr'][4]
    nii_snr = nii_flux / nii_flux_error
    n2 = np.log10(nii_flux / ha_n2_result['popt'][3])
    # update redshift
    z = ha_n2_result['popt'][0]

    print("NII S/N: ", nii_snr)
    print("N2: ", n2)
    print("EmFit catalog N2: ", emps['N2'].iloc[idx])

    rest_wave = spec['wavelength'] / (1 + z)
    plot_mask = (
        (rest_wave >= ha_n2_window[0][0])
        & (rest_wave <= ha_n2_window[1][1])
    )
    model = make_gaussian_model(ha_n2)
    flux_model = model(spec['wavelength'][plot_mask], *ha_n2_result['popt'])

    y_values = np.concatenate((np.asarray(spec['flux'][plot_mask]), flux_model))

    # focus on NII 6583
    y_min = np.nanmin(y_values)
    y_max = 2 * nii_flux * np.sqrt(2 * np.pi) * sigma   # np.nanmax(y_values)
    y_pad = 0.15 * max(y_max - y_min, np.finfo(float).eps)

    fig, ax = plt.subplots(num=fignum, clear=True, figsize=(6, 4))
    ax.plot(rest_wave[plot_mask], spec['flux'][plot_mask], color='black', lw=0.7, label='data')
    ax.plot(rest_wave[plot_mask], flux_model, color='crimson', lw=1.2, label='fit')
    ax.axvline(6585.28, color='tab:blue', ls='--', lw=0.9, label='[N II] 6583')
    ax.set_xlim(ha_n2_window[0][0], ha_n2_window[1][1])
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel('Rest-frame wavelength (Angstrom)')
    ax.set_ylabel('Flux')
    ax.set_title('H-alpha + [N II]')
    ax.legend()
    fig.tight_layout()
    plt.show()

    return result, fig, ax



def stack_spectra(targetids, emps, spectra_dir="../data/spectra", dwave=0.8, norm_window=(5950, 6050)):
    """
    Build an inverse-variance weighted stack of rest-frame spectra.

    Parameters
    ----------
    targetids : list of int
        TARGETIDs identifying the spectra to stack (files named f"{targetid}.fits"
        in `spectra_dir`, with 'wavelength', 'flux', 'ivar' columns).
    emps : pandas.DataFrame or astropy.table.Table
        Catalog used to look up the redshift (column 'Z') for each TARGETID.
    spectra_dir : str or Path, optional
        Directory containing the per-target spectrum fits files.
    dwave : float, optional
        Rest-frame wavelength step (in Angstroms) of the common output grid.
    norm_window : tuple of float, optional
        Rest-frame wavelength range (in Angstroms) used to normalize each
        spectrum by its mean flux before stacking (default: a line-free window
        around 6000 A).

    Returns
    -------
    stacked : astropy.table.Table
        Table with columns 'wavelength' (common rest-frame grid), 'flux'
        (inverse-variance weighted mean flux), 'ivar' (summed inverse
        variance of the flux estimate) and 'n_contrib' (number of spectra
        contributing, i.e. ivar > 0, at each grid point).
    """
    spectra_dir = Path(spectra_dir)
    emps_catalog = emps.to_pandas() if isinstance(emps, Table) else emps

    rest_waves, rest_fluxes, rest_ivars = [], [], []
    for targetid in targetids:
        matches = emps_catalog.loc[emps_catalog["TARGETID"] == targetid, "Z"]
        if matches.empty:
            raise ValueError(f"TARGETID {targetid} not found in emps.")
        z = float(matches.iloc[0])

        spec = Table.read(spectra_dir / f"{targetid}.fits", format="fits")
        wave = np.asarray(spec["wavelength"])
        flux = np.asarray(spec["flux"])
        ivar = np.asarray(spec["ivar"])

        # Shift to rest frame, conserving flux (and its variance) under the transform.
        rest_wave = wave / (1 + z)
        rest_flux = flux * (1 + z)
        rest_ivar = ivar / (1 + z) ** 2

        # Normalize by the mean continuum flux in a line-free window so all
        # spectra contribute on a comparable scale before weighting/stacking.
        norm_mask = (rest_wave >= norm_window[0]) & (rest_wave <= norm_window[1])
        norm = np.mean(rest_flux[norm_mask])
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"TARGETID {targetid}: invalid normalization flux ({norm}) in {norm_window}.")

        rest_waves.append(rest_wave)
        rest_fluxes.append(rest_flux / norm)
        rest_ivars.append(rest_ivar * norm ** 2)

    # Common rest-frame grid spanning the overlap of all input spectra.
    wave_min = max(w.min() for w in rest_waves)
    wave_max = min(w.max() for w in rest_waves)
    wave_grid = np.arange(wave_min, wave_max, dwave)

    ivar_sum = np.zeros_like(wave_grid)
    flux_ivar_sum = np.zeros_like(wave_grid)
    n_contrib = np.zeros_like(wave_grid, dtype=int)

    for wave, flux, ivar in zip(rest_waves, rest_fluxes, rest_ivars):
        order = np.argsort(wave)
        wave, flux, ivar = wave[order], flux[order], ivar[order]

        flux_i = np.interp(wave_grid, wave, flux)
        ivar_i = np.interp(wave_grid, wave, ivar, left=0.0, right=0.0)
        ivar_i[ivar_i < 0] = 0.0

        flux_ivar_sum += flux_i * ivar_i
        ivar_sum += ivar_i
        n_contrib += ivar_i > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        stacked_flux = np.where(ivar_sum > 0, flux_ivar_sum / ivar_sum, np.nan)

    return Table(
        {
            "wavelength": wave_grid,
            "flux": stacked_flux,
            "ivar": ivar_sum,
            "n_contrib": n_contrib,
        }
    )
