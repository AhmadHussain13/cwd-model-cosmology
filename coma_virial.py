import numpy as np
G = 6.6743e-11
def sigma_virial(M, R, alpha=3.0):
    """General virial-like relation:
    sigma^2 = G M / (alpha R)
    Common choices: alpha=3 (spherical isotropic), alpha=5 (uniform sphere)
    """
    return np.sqrt(G * M / (alpha * R))
if __name__ == "__main__":
    M = 2.4e+45
    R = 4.63e+22
    print("Virial sigma (alpha=3.0): {:.1f} km/s".format(sigma_virial(M,R,alpha=3.0)/1e3))
    print("Virial sigma (alpha=5.0): {:.1f} km/s".format(sigma_virial(M,R,alpha=5.0)/1e3))
