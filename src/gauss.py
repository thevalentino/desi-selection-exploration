import numpy as np
from scipy.integrate import quad


def normal(x, s):
    return np.exp(-(x**2) / 2 / s**2) / np.sqrt(2 * np.pi) / s


fwhm = 0.4
s = fwhm / 2.355
# flux = 8e-18
bin_size = 0.016
ston = 1.3

# print(quad(normal, -np.inf, np.inf, args=s)[0])
# print(quad(normal, -bin_size / 2, bin_size / 2, args=s)[0] / bin_size)

bin_centers = np.arange(-fwhm / 2, fwhm / 2 + bin_size, bin_size)
f = normal(bin_centers, s)
norm = 0.76 / bin_size / np.sum(f)

peak = (f * norm).max()
err_i = peak / ston

err_tot = np.sqrt(len(bin_centers)) * err_i

total_sn = np.sum(f) * norm / err_tot
