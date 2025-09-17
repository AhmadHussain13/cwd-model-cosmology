# appendix_f/class_cwd/class_cwd.py
import numpy as np

class ClassCWDError(Exception):
    pass

class ClassCWD:
    """
    Thin Python interface to the patched CLASS (CWD) backend.
    If the compiled CLASS wrapper isn't available, we fall back
    to a self-contained phenomenological stub so the rest of the
    pipeline (plots/likelihood/MCMC) runs everywhere.
    """
    def __init__(self, params, backend="auto"):
        """
        params: dict with keys:
          cosmology: {Omega_m, h, A_s, n_s}
          cwd: {alpha0, L_mpc, gamma}
        backend: "auto" | "class" | "stub"
        """
        self.params = self._normalize_params(params)
        self.backend = backend
        self._have_class = False

        if backend in ("auto", "class"):
            try:
                # If you have a compiled Python wrapper to CLASS, import it here:
                # from pylibclass_cwd import ClassCWD as _CBackend
                # self._cobj = _CBackend(self.params)
                # self._have_class = True
                # For now, we simulate failure (replace with your actual wrapper).
                raise ImportError("CLASS CWD backend not found in this environment.")
            except Exception:
                self._have_class = False
                if backend == "class":
                    raise ClassCWDError("Requested CLASS backend but it is not available.")
        # else backend == "stub": we use the stub below

    @staticmethod
    def _normalize_params(p):
        # Defaults + user overrides
        cosmo = dict(Omega_m=0.315, h=0.674, A_s=2.1e-9, n_s=0.965)
        cwd   = dict(alpha0=1.05, L_mpc=10.0, gamma=0.48)
        cosmo.update(p.get("cosmology", {}))
        cwd.update(p.get("cwd", {}))
        return {"cosmology": cosmo, "cwd": cwd}

    # -----------------------------
    # Public API used by scripts
    # -----------------------------
    def get_linear_matter_pk(self, k_hmpc, z):
        """
        Return P_m(k,z) (Mpc/h)^3.
        Uses CLASS if available, else a smooth analytic fit (stub).
        """
        if self._have_class:
            # Example call if you had a wrapper:
            # return self._cobj.get_linear_matter_pk(k_hmpc, z)
            raise NotImplementedError
        else:
            return self._stub_linear_pk(k_hmpc, z)

    def get_pquasar(self, k_hmpc, z, b=3.5, beta=0.3, sigma_v_kms=200.0, alpha0=None, L_mpc=None, gamma=None):
        """
        Compute quasar P_obs(k, mu) integrated over mu (monopole approx),
        using Kaiser+FoG with matter P_m from backend.
        """
        # Allow runtime overrides
        p = self.params["cwd"].copy()
        if alpha0 is not None: p["alpha0"] = alpha0
        if L_mpc  is not None: p["L_mpc"]  = L_mpc
        if gamma  is not None: p["gamma"]  = gamma

        Pm = self.get_linear_matter_pk(k_hmpc, z)

        # phenomenological CWD suppression/enhancement term:
        # scale-dependent factor ~ exp(-k*L/a)^gamma with amplitude alpha0
        a = 1.0 / (1.0 + z)
        L = max(p["L_mpc"], 1e-6)
        kL = np.clip(k_hmpc * L / a, 0, 1e3)
        f5d = p["alpha0"] * np.exp(-kL)  # simple, matches your appendix text
        # Apply to Pm multiplicatively as (1 + f5d) or a modest deviation:
        Pm_cwd = Pm * (1.0 + 0.05 * f5d**p["gamma"])  # ~ few % level around k~0.1

        # Kaiser + FoG monopole (mu-averaged):
        # Pq(k,mu) = b^2 * Pm_cwd * (1 + beta mu^2)^2 * exp[-(k mu sigma_v / H(z))^2 / 2]
        # We approximate mu integral by Gauss-Legendre with 12 nodes
        mu_nodes, mu_wts = np.polynomial.legendre.leggauss(12)
        # Convert sigma_v (km/s) to a k-damping in h/Mpc units; approximate with factor S:
        # Simple dimensionless damping: F = exp(-(k*mu*sigma)^2/2), treat sigma in Mpc/h units:
        # Use sigma_eff ~ (sigma_v / (a H0/h)) * 1e-3 * (Mpc/h)
        H0 = 100.0 * self.params["cosmology"]["h"]   # km/s/Mpc
        sigma_eff = (sigma_v_kms / (a * H0))         # Mpc
        # Convert to (Mpc/h) using h:
        sigma_eff *= self.params["cosmology"]["h"]   # Mpc/h

        Pq = np.zeros_like(k_hmpc)
        for mu, w in zip(mu_nodes, mu_wts):
            FoG = np.exp(-0.5 * (k_hmpc * mu * sigma_eff)**2)
            Pq += w * (b**2) * Pm_cwd * (1.0 + beta * mu**2)**2 * FoG
        # Average over mu in [-1,1] with weights summing to 2:
        Pq *= 0.5
        return Pq

    # -----------------------------
    # Stub backend (no CLASS)
    # -----------------------------
    def _stub_linear_pk(self, k_hmpc, z):
        """
        Smooth ΛCDM-like shape based on a BBKS/E&H-style transfer approximation,
        scaled with a simple z-evolution. This is *not* for precision — just to
        let the rest of the pipeline run anywhere.
        """
        k = np.asarray(k_hmpc)
        Om = self.params["cosmology"]["Omega_m"]
        h  = self.params["cosmology"]["h"]
        ns = self.params["cosmology"]["n_s"]
        As = self.params["cosmology"]["A_s"]

        # Very rough shape: turnover near k~0.02 h/Mpc; power-law tails
        k0 = 0.02
        T  = 1.0 / (1.0 + (k/k0)**2)**1.2
        D  = 1.0 / (1.0 + z)  # crude scaling for growth
        Pk = As * (k/0.05)**(ns-1.0) * T**2 * (D**2) * (1e9)  # scale to Mpc^3/h^3-ish
        return Pk