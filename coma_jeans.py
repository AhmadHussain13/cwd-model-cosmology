import numpy as np
from scipy.integrate import quad
G = 6.6743e-11
r_s = 1.54e+22
rho_s = 3.7e-23
R_vir = 4.63e+22
def rho_nfw(r):
    x = r / r_s
    return rho_s / (x * (1.0 + x)**2)
def M_nfw_enclosed(r):
    x = r / r_s
    return 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
def dPhi_dr(r):
    r = max(r, 1e-300)
    return G * M_nfw_enclosed(r) / (r**2)
def sigma_r2(r):
    nu = rho_nfw(r)
    if nu <= 0:
        return 0.0
    integrand = lambda s: rho_nfw(s) * dPhi_dr(s)
    integral = quad(integrand, r, R_vir, limit=200)[0]
    integral = max(integral, 0.0)
    return integral / nu
if __name__ == "__main__":
    r_test = r_s
    print("sigma_r(r_s) [km/s] = {:.1f}".format(np.sqrt(sigma_r2(r_test))/1e3))
