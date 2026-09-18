# Exploring DESI data

## Value Added Catalog for Emission Lines

One of the VACs is the `EmFit` catalog with emission line measurements for the strong lines in z<0.45 galaxies.

The data needs to be downloaded and it's a `fits` file of ~18 GB.

- [EmFit Documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/emfit/)
- [emfit-dr1-v2.3.1.fits](https://data.desi.lbl.gov/public/dr1/vac/dr1/emfit/v2.3/emfit-dr1-v2.3.1.fits)
- [Observability tool](https://iris-observability.lam.fr/chart/obsid:lco/)



## Log

- On 2026-10-18 I discovered an error in notebook `00-EMP-sample-selection.ipynb` which means file `data/emp-candidates-v261012.fits` is incorrect. Use new version: `data/emp-candidates-v2610-18.fits` and re-download the sources. Also extend redshift to 0.3.