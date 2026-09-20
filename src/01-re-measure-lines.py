"""Parallel remeasurement of emission lines in DESI spectra.

The input table must contain ``TARGETID`` and the initial redshift column
passed to :func:`measure_sources` (``Z`` by default).  The output is a
checkpointable Astropy table with one row per source.
"""

from __future__ import annotations

import importlib.util
import os
import concurrent.futures
from pathlib import Path

import numpy as np
from astropy.table import Table


N_PROCESSES = max(1, (os.cpu_count() or 2) - 1)
CHECKPOINT_EVERY = 25
DEFAULT_SPECTRA_DIR = Path(__file__).resolve().parents[1] / "data" / "spectra"


def _load_line_functions():
    module_path = Path(__file__).with_name("emline-inspect-functions.py")
    spec = importlib.util.spec_from_file_location("emline_inspect_functions", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _empty_result(targetid, input_z, status="error", error=""):
    """Return a consistently shaped result row, including failed fits."""
    return {
        "TARGETID": targetid,
        "input_z": input_z,
        "z_ha_n2": np.nan,
        "zerr_ha_n2": np.nan,
        "flux_nii6549": np.nan,
        "fluxerr_nii6549": np.nan,
        "flux_ha": np.nan,
        "fluxerr_ha": np.nan,
        "flux_nii6585": np.nan,
        "fluxerr_nii6585": np.nan,
        "z_hb": np.nan,
        "zerr_hb": np.nan,
        "flux_hb": np.nan,
        "fluxerr_hb": np.nan,
        "z_o3_b": np.nan,
        "zerr_o3_b": np.nan,
        "flux_o3_b": np.nan,
        "fluxerr_o3_b": np.nan,
        "status": status,
        "error": error,
    }


def measure_source(targetid, z, spectra_dir=DEFAULT_SPECTRA_DIR):
    """Measure one source and return its line fluxes, errors, and redshifts.

    Parameters
    ----------
    targetid : int
        Source identifier; its spectrum is ``{targetid}.fits``.
    z : float
        Initial redshift used by the line fits.
    spectra_dir : path-like, optional
        Directory containing the source spectra.
    """
    functions = _load_line_functions()
    result = _empty_result(targetid, float(z), status="ok")
    spec = Table.read(Path(spectra_dir) / f"{targetid}.fits", format="fits")
    fits = functions.fit_spectrum_lines(spec, float(z))

    for group in ("ha_n2", "hb", "o3_b"):
        fit = fits[group]
        result[f"z_{group}"] = float(fit["popt"][0])
        result[f"zerr_{group}"] = float(fit["perr"][0])

    ha_n2 = fits["ha_n2"]
    result["flux_nii6549"] = float(ha_n2["popt"][2])
    result["fluxerr_nii6549"] = float(ha_n2["perr"][2])
    result["flux_ha"] = float(ha_n2["popt"][3])
    result["fluxerr_ha"] = float(ha_n2["perr"][3])
    result["flux_nii6585"] = float(ha_n2["popt"][4])
    result["fluxerr_nii6585"] = float(ha_n2["perr"][4])

    for group, name in (("hb", "hb"), ("o3_b", "o3_b")):
        result[f"flux_{name}"] = float(fits[group]["popt"][2])
        result[f"fluxerr_{name}"] = float(fits[group]["perr"][2])

    return result


def _measure_source_worker(args):
    targetid, z, spectra_dir = args
    try:
        return measure_source(targetid, z, spectra_dir)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return _empty_result(targetid, z, error=f"{type(exc).__name__}: {exc}")


def _table_from_rows(rows):
    return Table(rows=rows, names=list(_empty_result(0, np.nan).keys()))


def _write_checkpoint(rows, output_path):
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    _table_from_rows(rows).write(temporary_path, format="fits", overwrite=True)
    temporary_path.replace(output_path)


def measure_sources(
    sources,
    output_path="line-measurements.fits",
    spectra_dir=DEFAULT_SPECTRA_DIR,
    redshift_column="Z",
    n_processes=N_PROCESSES,
    checkpoint_every=CHECKPOINT_EVERY,
    redo=False,
):
    """Measure a source table in parallel and periodically save progress.

    ``sources`` must contain ``TARGETID`` and ``redshift_column``. Existing
    rows in ``output_path`` are skipped unless ``redo=True``. Failed sources
    are also checkpointed, so a rerun does not repeatedly attempt them.
    Set ``redo=True`` to replace every existing measurement.
    """
    sources = sources if isinstance(sources, Table) else Table(sources)
    if "TARGETID" not in sources.colnames:
        raise ValueError("sources must contain a TARGETID column")
    if redshift_column not in sources.colnames:
        raise ValueError(f"sources must contain a {redshift_column!r} column")
    if n_processes < 1 or checkpoint_every < 1:
        raise ValueError("n_processes and checkpoint_every must be positive")

    output_path = Path(output_path)
    rows = []
    completed_ids = set()
    if output_path.exists() and not redo:
        previous = Table.read(output_path, format="fits")
        rows = [dict(zip(previous.colnames, row)) for row in previous]
        completed_ids = set(previous["TARGETID"].tolist())

    jobs = [
        (source["TARGETID"].item(), float(source[redshift_column]), str(spectra_dir))
        for source in sources
        if source["TARGETID"].item() not in completed_ids
    ]

    if not jobs:
        if not output_path.exists():
            _write_checkpoint(rows, output_path)
        return _table_from_rows(rows)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(_measure_source_worker, job) for job in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if count % checkpoint_every == 0:
                _write_checkpoint(rows, output_path)

    _write_checkpoint(rows, output_path)
    return _table_from_rows(rows)

if __name__ == "__main__":
    emps = Table.read(
        Path(__file__).resolve().parents[1] / "data" / "emp-candidates-v261018.fits",
        format="fits",
    )

    measure_sources(
        emps,
        output_path="../data/line-measurements-v261018.fits",
        n_processes=32,
        checkpoint_every=25,
        redo=False
        )

