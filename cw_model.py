from __future__ import annotations

"""cw_model.py

Circular-velocity components for common Milky Way / galaxy mass models and a simple composite wrapper that sums them in quadrature.

Units:

Radii R are in kpc

Velocities are returned in km/s

Masses are in Msun

Densities in Msun/kpc^3

Constants use G = 4.30091e-6  [kpc (km/s)^2 / Msun]

Components implemented:

NFW halo (init via (rho_s, r_s) or (M200, c200, H0))

Burkert halo (rho0, r0)

Pseudo-Isothermal (ISO) core halo (rho0, rc)

Exponential disk (Freeman 1970) with exact Bessel formula

Uses SciPy if available; otherwise a built-in Cephes-style approximation

Hernquist bulge (M, a)

Optional exponential gas disk (same kernel as stellar disk)

CompositeModel combines any subset of components and produces the total rotation curve.

Example

>>> import numpy as np
>>> R = np.linspace(0.1, 30, 200)
>>> halo = NFW.from_Mc(M200=1.2e12, c200=10.0)  # Msun, dimensionless
>>> disk = ExponentialDisk(M=6e10, Rd=3.0)      # Msun, kpc
>>> bulge = HernquistBulge(M=8e9, a=0.6)        # Msun, kpc
>>> model = CompositeModel(halo=halo, disk=disk, bulge=bulge)
>>> vc = model.vc_total(R)                      # km/s
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

# ---------------------------------------------
# Constants & helpers
# ---------------------------------------------

G = 4.30091e-6  # kpc (km/s)^2 / Msun
TWOPI = 2.0 * np.pi
FOURPI = 4.0 * np.pi

def _asarray(R: np.ndarray | float) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    return R

# ---------------------------------------------
# Modified Bessel functions (fallback if SciPy absent)
# ---------------------------------------------

try:  # pragma: no cover
    from scipy.special import i0 as _i0, i1 as _i1, k0 as _k0, k1 as _k1  # type: ignore

    def I0(x):
        return _i0(x)

    def I1(x):
        return _i1(x)

    def K0(x):
        return _k0(x)

    def K1(x):
        return _k1(x)

    _HAVE_SCIPY = True

except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

    # Cephes-style polynomial/rational approximations
    # Source: public-domain Cephes / Numerical Recipes style
    # Accurate to ~1e-7 relative for typical galaxy RC y-values
    def I0(x):
        x = np.abs(np.asarray(x, dtype=float))
        y = x * x
        p = 1.0 + y * (
            3.5156229
            + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))))
        )
        q = 0.0  # kept for symmetry with K0 form
        # asymptotic for large x
        mask = x > 3.75
        if np.any(mask):
            z = (x[mask] - 3.75) / (x[mask])
            p_asym = (
                0.39894228
                + z * (0.01328592 + z * (0.00225319 + z * (-0.00157565 + z * (0.00916281 + z * (-0.02057706 + z * (0.02635537 + z * (-0.01647633 + z * 0.00392377)))))))
            )
            p_asym *= np.exp(x[mask]) / np.sqrt(x[mask])
            p = np.where(mask, p_asym, p)
        return p

    def I1(x):
        ax = np.abs(np.asarray(x, dtype=float))
        y = ax * ax
        p = ax * (
            0.5
            + y * (
                0.87890594
                + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))
            )
        )
        # asymptotic
        mask = ax > 3.75
        if np.any(mask):
            z = (ax[mask] - 3.75) / ax[mask]
            p_asym = (
                0.39894228
                + z * (-0.03988024 + z * (-0.00362018 + z * (0.00163801 + z * (-0.01031555 + z * (0.02282967 + z * (-0.02895312 + z * (0.01787654 - z * 0.00420059)))))))
            )
            p_asym *= np.exp(ax[mask]) / np.sqrt(ax[mask])
            p = np.where(mask, p_asym, p)
        return np.where(np.asarray(x) < 0.0, -p, p)

    def K0(x):
        x = np.asarray(x, dtype=float)
        y = x * x
        # small x
        p = (
            -np.log(x / 2.0) * I0(x)
            + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))))
        )
        # large x
        mask = x > 2.0
        if np.any(mask):
            z = 2.0 / x[mask]
            p_asym = (
                1.25331414
                + z * (-0.07832358 + z * (0.02189568 + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208)))))
            )
            p_asym *= np.exp(-x[mask]) / np.sqrt(x[mask])
            p = np.where(mask, p_asym, p)
        return p

    def K1(x):
        x = np.asarray(x, dtype=float)
        y = x * x
        p = (
            np.log(x / 2.0) * I1(x)
            + (1.0 / x)
            + x * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * -0.00004686)))))
        )
        mask = x > 2.0
        if np.any(mask):
            z = 2.0 / x[mask]
            p_asym = (
                1.25331414
                + z * (0.23498619 + z * (-0.03655620 + z * (0.01504268 + z * (-0.00780353 + z * (0.00325614 + z * -0.00068245)))))
            )
            p_asym *= np.exp(-x[mask]) / np.sqrt(x[mask])
            p = np.where(mask, p_asym, p)
        return p

# ---------------------------------------------
# Critical density helper
# ---------------------------------------------

def rho_crit(H0: float = 70.0) -> float:
    """Return critical density rho_c in Msun/kpc^3 for given H0 [km/s/Mpc]."""
    # Convert H0 to km/s/kpc
    H = H0 / 1000.0  # km/s/kpc
    return 3.0 * H * H / (8.0 * np.pi * G)

# ---------------------------------------------
# Halo profiles
# ---------------------------------------------

@dataclass
class NFW:
    """NFW halo.

    Parameters
    ----------
    rho_s : float
        Characteristic density [Msun/kpc^3].
    r_s : float
        Scale radius [kpc].
    """
    rho_s: float
    r_s: float

    @classmethod
    def from_Mc(cls, M200: float, c200: float, H0: float = 70.0) -> "NFW":
        """Construct from (M200, c200).

        M200 in Msun, c200 dimensionless. Uses spherical overdensity of 200 * rho_crit.
        """
        rho_c = rho_crit(H0)
        r200 = (3.0 * M200 / (FOURPI * 200.0 * rho_c)) ** (1.0 / 3.0)  # kpc
        r_s = r200 / c200
        f_c = np.log(1.0 + c200) - c200 / (1.0 + c200)
        rho_s = M200 / (FOURPI * r_s**3 * f_c)
        return cls(rho_s=float(rho_s), r_s=float(r_s))

    def mass_enclosed(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        x = R / self.r_s
        f = np.log(1.0 + x) - x / (1.0 + x)
        return FOURPI * self.r_s**3 * self.rho_s * f

    def vc(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        M = self.mass_enclosed(R)
        with np.errstate(divide="ignore", invalid="ignore"):
            V2 = G * M / R
        V2 = np.where(R > 0.0, V2, 0.0)
        return np.sqrt(np.maximum(V2, 0.0))

@dataclass
class Burkert:
    """Burkert cored halo.

    rho0 : central density [Msun/kpc^3]
    r0   : core radius [kpc]
    """
    rho0: float
    r0: float

    def mass_enclosed(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        x = R / self.r0
        # Analytic cumulative mass for Burkert profile
        term = np.log(1.0 + x) + 0.5 * np.log(1.0 + x * x) - np.arctan(x)
        M = 2.0 * np.pi * self.rho0 * self.r0**3 * term
        return M

    def vc(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        M = self.mass_enclosed(R)
        with np.errstate(divide="ignore", invalid="ignore"):
            V2 = G * M / R
        V2 = np.where(R > 0.0, V2, 0.0)
        return np.sqrt(np.maximum(V2, 0.0))

@dataclass
class IsoCore:
    """Pseudo-Isothermal (ISO) cored halo: rho(r)=rho0/(1+(r/rc)^2)."""
    rho0: float  # Msun/kpc^3
    rc: float    # kpc

    def vc(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        x = R / self.rc
        with np.errstate(divide="ignore", invalid="ignore"):
            V2 = FOURPI * G * self.rho0 * self.rc**2 * (1.0 - (1.0 / x) * np.arctan(x))
        V2 = np.where(R > 0.0, V2, 0.0)
        return np.sqrt(np.maximum(V2, 0.0))

# ---------------------------------------------
# Baryons
# ---------------------------------------------

@dataclass
class ExponentialDisk:
    """Razor-thin exponential disk (Freeman 1970).

    Parameters
    ----------
    M : float
        Total mass of the disk [Msun].
    Rd : float
        Scale length [kpc].
    """
    M: float
    Rd: float

    def vc(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        y = R / (2.0 * self.Rd)
        # Freeman kernel: V^2 = 2 G M / Rd * y^2 [I0(y) K0(y) - I1(y) K1(y)]
        # Handle y=0 -> V=0
        kernel = I0(y) * K0(y) - I1(y) * K1(y)
        V2 = 2.0 * G * self.M / self.Rd * (y * y) * kernel
        V2 = np.where(R > 0.0, V2, 0.0)
        return np.sqrt(np.maximum(V2, 0.0))

@dataclass
class HernquistBulge:
    """Hernquist bulge.

    Parameters
    ----------
    M : float
        Total bulge mass [Msun]
    a : float
        Scale length [kpc]
    """
    M: float
    a: float

    def vc(self, R: np.ndarray | float) -> np.ndarray:
        R = _asarray(R)
        V2 = G * self.M * R / (R + self.a) ** 2
        V2 = np.where(R > 0.0, V2, 0.0)
        return np.sqrt(np.maximum(V2, 0.0))

# ---------------------------------------------
# Composite wrapper
# ---------------------------------------------

@dataclass
class CompositeModel:
    """Combine components and return circular-velocity curves.

    Any component can be omitted (set to None).
    """
    halo: Optional[Any] = None  # NFW/Burkert/IsoCore
    disk: Optional[ExponentialDisk] = None
    bulge: Optional[HernquistBulge] = None
    gas: Optional[ExponentialDisk] = None

    def vc_components(self, R: np.ndarray | float) -> Dict[str, np.ndarray]:
        R = _asarray(R)
        out: Dict[str, np.ndarray] = {}
        if self.halo is not None:
            out["halo"] = self.halo.vc(R)
        if self.disk is not None:
            out["disk"] = self.disk.vc(R)
        if self.bulge is not None:
            out["bulge"] = self.bulge.vc(R)
        if self.gas is not None:
            out["gas"] = self.gas.vc(R)
        return out

    def vc_total(self, R: np.ndarray | float) -> np.ndarray:
        comps = self.vc_components(R)
        if not comps:
            return np.zeros_like(_asarray(R))
        V2 = np.zeros_like(next(iter(comps.values()))) ** 2
        for v in comps.values():
            V2 = V2 + v * v
        return np.sqrt(V2)

    def to_dict(self, R: np.ndarray | float) -> Dict[str, np.ndarray]:
        R = _asarray(R)
        d = self.vc_components(R)
        d["total"] = self.vc_total(R)
        d["R"] = R
        return d

# ---------------------------------------------
# Convenience factory from parameter dicts
# ---------------------------------------------

def build_model(params: Dict[str, Any]) -> CompositeModel:
    """Build a CompositeModel from a nested parameter dict.

    Expected structure (all keys optional):
    params = {
        "halo": {"type": "nfw"|"burkert"|"iso", ...},
        "disk": {"M": ..., "Rd": ...},
        "bulge": {"M": ..., "a": ...},
        "gas": {"M": ..., "Rd": ...},
    }

    Notes for halo options:
    - NFW: either {"rho_s": ..., "r_s": ...} or {"M200": ..., "c200": ..., "H0": 70.0}
    - Burkert: {"rho0": ..., "r0": ...}
    - Iso: {"rho0": ..., "rc": ...}
    """
    halo = None
    if "halo" in params and params["halo"]:
        hpars = params["halo"]
        htype = (hpars.get("type", "nfw")).lower()
        if htype == "nfw":
            if ("rho_s" in hpars) and ("r_s" in hpars):
                halo = NFW(rho_s=float(hpars["rho_s"]), r_s=float(hpars["r_s"]))
            elif ("M200" in hpars) and ("c200" in hpars):
                halo = NFW.from_Mc(M200=float(hpars["M200"]), c200=float(hpars["c200"]), H0=float(hpars.get("H0", 70.0)))
            else:
                raise ValueError("NFW requires (rho_s, r_s) or (M200, c200[, H0]).")
        elif htype == "burkert":
            halo = Burkert(rho0=float(hpars["rho0"]), r0=float(hpars["r0"]))
        elif htype in ("iso", "isocore", "piso", "pseudoisothermal"):
            halo = IsoCore(rho0=float(hpars["rho0"]), rc=float(hpars["rc"]))
        else:
            raise ValueError(f"Unknown halo type: {htype}")

    disk = None
    if "disk" in params and params["disk"]:
        dpars = params["disk"]
        disk = ExponentialDisk(M=float(dpars["M"]), Rd=float(dpars["Rd"]))

    bulge = None
    if "bulge" in params and params["bulge"]:
        bpars = params["bulge"]
        bulge = HernquistBulge(M=float(bpars["M"]), a=float(bpars["a"]))

    gas = None
    if "gas" in params and params["gas"]:
        gpars = params["gas"]
        gas = ExponentialDisk(M=float(gpars["M"]), Rd=float(gpars["Rd"]))

    return CompositeModel(halo=halo, disk=disk, bulge=bulge, gas=gas)

__all__ = [
    "G", "rho_crit", "NFW", "Burkert", "IsoCore",
    "ExponentialDisk", "HernquistBulge", "CompositeModel", "build_model",
]

# ---------------------------------------------
# Example execution to display output
# ---------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set up the model using the example from the docstring
    R = np.linspace(0.1, 30, 200)  # Radii from 0.1 to 30 kpc
    halo = NFW.from_Mc(M200=1.2e12, c200=10.0)  # Msun, dimensionless
    disk = ExponentialDisk(M=6e10, Rd=3.0)      # Msun, kpc
    bulge = HernquistBulge(M=8e9, a=0.6)        # Msun, kpc
    model = CompositeModel(halo=halo, disk=disk, bulge=bulge)

    # Compute rotation curve components and total
    result = model.to_dict(R)
    
    # Plot the rotation curves
    plt.figure(figsize=(8, 6))
    plt.plot(result["R"], result["total"], label="Total", color="black", linewidth=2)
    plt.plot(result["R"], result["halo"], label="Halo (NFW)", linestyle="--")
    plt.plot(result["R"], result["disk"], label="Disk", linestyle="-.")
    plt.plot(result["R"], result["bulge"], label="Bulge", linestyle=":")
    
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Circular Velocity (km/s)")
    plt.title("Galactic Rotation Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optionally, print some values
    print("Sample rotation curve values:")
    for i in range(0, len(result["R"]), 40):  # Print every 40th point
        print(f"R = {result['R'][i]:.2f} kpc, V_total = {result['total'][i]:.2f} km/s")