import os

from astropy.table import Table
from sparcl.client import SparclClient

emps = Table.read("../data/emp-candidates-v261018.fits").to_pandas()
emps["n_records"] = 1
N_to_download = len(emps)

client = SparclClient(connect_timeout=10.0)

out_dir = "../data/spectra"


for i, targetid in enumerate(emps["TARGETID"]):
    print(f"Downloading {i+1}/{N_to_download}: {targetid}")
    if os.path.exists(f"{out_dir}/{targetid}.fits"):
        print(f"WARNING: {out_dir}/{targetid}.fits already exists, skipping")
        continue

    res = client.retrieve_by_specid(
        specid_list=[int(targetid)],
        include=["flux", "wavelength", "ivar"],
        dataset_list=["DESI-DR1"],
    )

    if len(res.records) > 1:
        print(f"WARNING: id:{targetid} has {len(res.records)} records")
        emps.loc[emps["TARGETID"] == targetid, "n_records"] = len(res.records)

    t = Table()
    t["flux"] = res.records[0].flux
    t["wavelength"] = res.records[0].wavelength
    t["ivar"] = res.records[0].ivar
    t.write(f"{out_dir}/{targetid}.fits", format="fits", overwrite=True)
