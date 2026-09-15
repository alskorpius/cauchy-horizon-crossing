"""T34: cost of crossing the seam (inner horizon) in the two-flux Ori model.

Geometry (single outgoing null shell Sigma + ingoing Price tail, CR 2021 arXiv:2101.05006 eqs (10), (32)):
  region R- (before the shell):  ds^2 = -f_-(v,r) dv^2 + 2 dv dr + r^2 dOmega^2,  f_- = f(m_-(v), r),  m_-(v) = m0 - beta/v^p
  region R+ (behind the shell):  same form in its own advanced time v_+,  f_+ = f(m_+(v_+), r);  dv_+/dv = rho(v) = f_-/f_+ on Sigma.
  Mass law: f(m, r) = f_base(r) + (m - m_ref) D(r)  [RN: D = -2/r exactly; scenario: linearised law of T2, D = -2 G(r)/r, G = dM/dm].
  In the single chart (v, r) the R+ metric is -rho^2 f_+ dv^2 + 2 rho dv dr (conformal factor psi = ln rho(v), a function of v only).
  Scenario profile near the inner root: the numerical tuning residue (f, f', f'' ~ 2e-7, 8e-6, 1e-2 at r0) is removed with a
  local window so that the base profile has an exact triple root; detuning eps0 (T1 type a) sets kappa_- = -(3/2)|a|^{1/3} eps0^{2/3}.

Observer: radial timelike geodesic, E = 1 at the outer horizon, entering at advanced time v_e.  Falls through R- (static up to the
slowly varying tail), crosses Sigma, continues in R+ until the Planck cut K = 1/l_P^4 (largest mass considered) or r -> r_floor.
Tidal tensor E_ab = R_acbd u^c u^d in the parallel-propagated frame (symbolic derivation: T34_derive.py):
  E_rad = f_rr / 2,      E_perp = f_r/(2r) + e^psi vdot^2 dM/dv / r^2      (Schwarzschild check: -2M/r^3, +M/r^3).
Null-shell crossing (outgoing Vaidya layer with mass jump [M] = M_+ - M_- = (R/2)(f_- - f_+) > 0 inside the trapped region):
  transverse impulse per direction  I_sh = [M] |udot| / R^2,  |udot| = 1/(|f_-| vdot)  (u = retarded time of R-);  radial impulse 0.
  kinematics across the shell: udot continuous  =>  vdot_+ = rho vdot,  rdot_+ = f_- vdot/2 - 1/(2 rho vdot).
Deformation bookkeeping along tau: J = int E dtau (impulse = relative velocity per unit separation), Delta = int J dtau (Ori's
twice-integrated curvature = strain of a free body), and the linear deviation ODE xi'' = -E xi with xi = 1, xi' = 0 at the outer horizon.
Units G = c = M = 1 (M = mass of the unperturbed profile).  Run: .venv/Scripts/python.exe experiments/cycle2/T34_tidal.py
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "inhomogeneous_collapse"))
import ltb_bounce as lb  # noqa: E402

LOG = []
T0 = time.time()


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------- physical constants
MSUN_S = 4.925e-6          # G M_sun / c^3 [s]
MSUN_M = 1476.6            # G M_sun / c^2 [m]
LPL_M = 1.616e-35          # Planck length [m]
MASSES = {"10 Msun": 10.0, "1e6 Msun": 1e6, "1e9 Msun": 1e9}


def K_planck(Msun):
    """Planck curvature 1/l_P^4 in units M^-4 for a hole of Msun solar masses."""
    return (Msun * MSUN_M / LPL_M) ** 4


K_PL_MAX = K_planck(1e9)

# ----------------------------------------------------------------------------- scenario profile (base, lambda = 0.271)
scen = lb.ScenarioMass()
ELL = scen.ell


def base_fields(r):
    """M, M_r, M_rr and G = dM/dm, G_r, G_rr of the base profile at m = 1 (G_rr by central difference of m_RM)."""
    r = np.asarray(r, float)
    d = scen.all(1.0, r)
    h = 1e-5 * np.maximum(r, 1e-3)
    dp = scen.all(1.0, r + h)
    dm = scen.all(1.0, r - h)
    Grr = (dp["m_RM"] - dm["m_RM"]) / (2 * h)
    return d["m"], d["m_R"], d["m_RR"], d["m_M"], d["m_RM"], Grr


class ScenGeom:
    """Scenario family, linearised mass law of T2: f(m, r) = f_ideal(r) + eps0 (-2 M1/r) + (m - 1)(-2 G/r)."""

    m_ref = 1.0
    Y_SW = 0.10        # |y| below which the polynomial near-root model is used for the shell
    Y_WIN = 0.40       # window of the residue removal

    def __init__(self, eps0, label):
        self.eps0 = eps0
        self.label = label
        rr = np.linspace(0.6, 0.76, 400001)
        m, *_ = base_fields(rr)
        f = 1 - 2 * m / rr
        sg = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        self.r0 = float(rr[sg[len(sg) // 2]]) if len(sg) else float(rr[np.argmin(np.abs(f))])
        y = np.linspace(-0.12, 0.12, 4001)
        r = self.r0 + y
        m, *_ = base_fields(r)
        c = P.polyfit(y, 1 - 2 * m / r, 5)
        self.res = np.array(c[:3])                                # tuning residue c0 + c1 y + c2 y^2
        self.P0 = np.array([0.0, 0.0, 0.0, c[3], c[4], c[5]])      # ideal triple root
        self.a3 = float(c[3])
        y2 = np.linspace(-0.15, 0.15, 4001)
        r2 = self.r0 + y2
        m2, mr2, mrr2, G2, Gr2, Grr2 = base_fields(r2)
        self.Pa = P.polyfit(y2, -2 * m2 / r2, 9)
        self.Plin = P.polyfit(y2, -2 * G2 / r2, 9)
        f_id = self.exact(0.0, r2)[0] - eps0 * P.polyval(y2, self.Pa)
        self.mismatch_sw = float(np.max(np.abs(P.polyval(y2, self.P0) - f_id)[np.abs(y2) < self.Y_SW + 0.01]))
        say(f"  [{label}] r0 = {self.r0:.6f}; residue removed (c0, c1, c2) = {self.res}; a = f'''/6 = {self.a3:+.4f} (T1: -2.158); "
            f"eps0 = {eps0:.3e}; |P0 - f_ideal| on |y| < {self.Y_SW}: {self.mismatch_sw:.1e}")
        self._cache = (None, None)

    # window and residue
    def _corr(self, r):
        y = r - self.r0
        w = np.exp(-(y / self.Y_WIN) ** 2)
        c0, c1, c2 = self.res
        q = c0 + c1 * y + c2 * y**2
        qy = c1 + 2 * c2 * y
        qyy = 2 * c2
        wy = w * (-2 * y / self.Y_WIN**2)
        wyy = w * (4 * y**2 / self.Y_WIN**4 - 2 / self.Y_WIN**2)
        return w * q, wy * q + w * qy, wyy * q + 2 * wy * qy + w * qyy

    def exact(self, q, r):
        """f, f_r, f_rr, D = df/dm, Misner-Sharp M for mass parameter m = 1 + q."""
        r = np.maximum(np.asarray(r, float), 1e-6)
        m, mr, mrr, G, Gr, Grr = base_fields(r)
        e = 1 + self.eps0
        M = e * m + q * G
        Mr = e * mr + q * Gr
        Mrr = e * mrr + q * Grr
        f = 1 - 2 * M / r
        fr = 2 * M / r**2 - 2 * Mr / r
        frr = -4 * M / r**3 + 4 * Mr / r**2 - 2 * Mrr / r
        c, cy, cyy = self._corr(r)
        f, fr, frr = f - c, fr - cy, frr - cyy
        D = -2 * G / r
        return f, fr, frr, D, r * (1 - f) / 2

    def outer_horizon(self, q):
        return brentq(lambda r: float(self.exact(q, r)[0]), 1.5, 2.6, xtol=1e-13)

    # near-root polynomial model
    def poly_fminus(self, qm):
        return P.polyadd(P.polyadd(self.P0, self.eps0 * self.Pa), qm * self.Plin)

    def local(self, qm, y_guess=None):
        """(y_root, Taylor coefficients of f_-(y_root + x), dy_root/dq_-) for tail parameter q_- = m_- - 1.
        Root by Newton iteration from the cubic estimate y = cbrt(-c0/a) (the perturbed cubic is monotonic: one real root);
        exact for q_- = 0 (triple root at y = 0).  Cached on q_-."""
        if self._cache[0] == qm:
            return self._cache[1]
        c = self.poly_fminus(qm)
        c = np.pad(c, (0, len(self.Plin) - len(c)))
        if qm == 0.0 and self.eps0 == 0.0:
            yr = 0.0
        else:
            c0 = float(c[0])
            yr = float(np.cbrt(-c0 / self.a3)) if c0 != 0.0 else 0.0
            dc = P.polyder(c)
            for _ in range(60):
                fv = float(P.polyval(yr, c))
                dv = float(P.polyval(yr, dc))
                if dv == 0.0:
                    break
                step = fv / dv
                yr -= step
                if abs(step) <= 1e-15 * max(abs(yr), 1e-300):
                    break
        tay = []
        d = c.copy()
        for k in range(len(c)):
            tay.append(float(P.polyval(yr, d)) / math.factorial(k))
            d = P.polyder(d)
        tay = np.array(tay)
        tay[0] = 0.0
        dyr = (-float(P.polyval(yr, self.Plin)) / tay[1]) if tay[1] != 0.0 else 0.0
        out = (yr, tay, dyr)
        self._cache = (qm, out)
        return out

    def shell_f(self, qm, x):
        """f_- at R = r0 + y_root + x (Taylor near the root, exact ideal profile farther out)."""
        yr, tay, _ = self.local(qm)
        if abs(yr + x) < self.Y_SW:
            return float(P.polyval(x, tay))
        return float(self.exact(qm, self.r0 + yr + x)[0])

    def shell_D(self, qm, x):
        yr, _, _ = self.local(qm)
        return float(P.polyval(yr + x, self.Plin)) if abs(yr + x) < 0.15 else float(self.exact(qm, self.r0 + yr + x)[3])

    def shell_root(self, qm):
        return self.r0 + self.local(qm)[0]

    def shell_droot_dq(self, qm):
        return self.local(qm)[2]

    def kappa_root(self, qm=0.0):
        yr, tay, _ = self.local(qm)
        return 0.5 * tay[1]


class RNGeom:
    """Reissner-Nordstrom control: f(m, r) = 1 - 2m/r + e^2/r^2, D = -2/r (exact)."""

    m_ref = 0.0

    def __init__(self, e=0.9, label="RN e=0.9"):
        self.e = e
        self.label = label
        self.eps0 = 0.0
        self.r0 = self.rminus(1.0)

    def rminus(self, m):
        return m - np.sqrt(m**2 - self.e**2)

    def rplus(self, m):
        return m + np.sqrt(m**2 - self.e**2)

    def exact(self, q, r):
        m = q
        f = 1 - 2 * m / r + self.e**2 / r**2
        fr = 2 * m / r**2 - 2 * self.e**2 / r**3
        frr = -4 * m / r**3 + 6 * self.e**2 / r**4
        return f, fr, frr, -2 / r, m - self.e**2 / (2 * r)

    def outer_horizon(self, q):
        return self.rplus(q)

    def shell_f(self, m, x):
        rm, rp = self.rminus(m), self.rplus(m)
        r = rm + x
        return x * (x + rm - rp) / r**2

    def shell_D(self, m, x):
        return -2 / (self.rminus(m) + x)

    def shell_root(self, m):
        return self.rminus(m)

    def shell_droot_dq(self, m):
        return 1 - m / np.sqrt(m**2 - self.e**2)

    def kappa_root(self, m=1.0):
        rm, rp = self.rminus(m), self.rplus(m)
        return (rm - rp) / (2 * rm**2)


# ----------------------------------------------------------------------------- shell (Sigma) dynamics
def solve_shell(geom, p=11, eps0=1e-3, v0=50.0, dm_plus=1e-4, v_end=1e12, R0=None, npts=4000, L_max=600.0):
    """Outgoing null shell in R- time v: x = R - r_root(m_-(v)), Lambda = ln(m_+ - m_ref), W = int (f_-/f_+) dv."""
    m0 = 1.0
    beta = eps0 * m0 * v0**p
    m_minus = lambda v: m0 - beta / v**p
    dm_minus = lambda v: p * beta / v ** (p + 1)
    qmf = (lambda v: m_minus(v)) if geom.m_ref == 0.0 else (lambda v: -beta / v**p)   # q_- = m_- - m_ref without cancellation
    R0 = R0 or (0.5 * (geom.rminus(m0) + geom.rplus(m0)) if isinstance(geom, RNGeom) else 1.33)

    def rhs(lv, y):
        v = np.exp(lv)
        lx, LQ, W = y
        x = math.exp(lx)
        Q = math.exp(min(LQ, 700.0))                # Q = m_+ - m_-(v) > 0 (mass-function jump across the shell in R- normalisation)
        qm = qmf(v)
        fm = geom.shell_f(qm, x)
        D = geom.shell_D(qm, x)
        fp = fm + Q * D
        dlx = (fm / 2 - geom.shell_droot_dq(qm) * dm_minus(v)) / x
        dLQ = dm_minus(v) * D / fm                  # d ln Q/dv from m_+' = m_-' f_+/f_-
        dW = fm / fp
        return [v * dlx, v * dLQ, v * dW]

    def ev_L(lv, y):
        return L_max - y[1]
    ev_L.terminal = True

    x0 = R0 - geom.shell_root(qmf(v0))
    Q0 = m_minus(v0) * dm_plus
    lv = np.linspace(np.log(v0), np.log(v_end), npts)
    sol = solve_ivp(rhs, (lv[0], lv[-1]), [math.log(x0), math.log(Q0), 0.0], method="LSODA", rtol=1e-9, atol=1e-12, t_eval=lv, events=[ev_L])
    lvs = sol.t
    lx, LQ, W = sol.y
    ok = np.isfinite(lx) & np.isfinite(LQ) & np.isfinite(W) & (LQ < 690)
    lvs, lx, LQ, W = lvs[ok], lx[ok], LQ[ok], W[ok]
    x = np.exp(lx)
    Q = np.exp(LQ)
    v = np.exp(lvs)
    qm = qmf(v)
    q = Q + qm
    fm = np.array([geom.shell_f(a, b) for a, b in zip(qm, x)])
    D = np.array([geom.shell_D(a, b) for a, b in zip(qm, x)])
    fp = fm + Q * D
    rho = fm / fp
    ok = np.isfinite(rho) & (rho > 0) & np.isfinite(fm) & (fm < 0)
    if not ok.all():
        bad = np.nonzero(~ok)[0]
        say(f"  shell: dropped {len(bad)} non-finite rows (first: v = {np.exp(lvs[bad[0]]):.4g}, x = {x[bad[0]]:.3e}, f_- = {fm[bad[0]]:.3e}, f_+ = {fp[bad[0]]:.3e}, Q = {Q[bad[0]]:.3e})")
        lvs, x, Q, LQ, qm, q, fm, D, fp, rho, W = lvs[ok], x[ok], Q[ok], LQ[ok], qm[ok], q[ok], fm[ok], D[ok], fp[ok], rho[ok], W[ok]
        v = np.exp(lvs)
    Rs = np.array([geom.shell_root(a) for a in qm]) + x
    tab = dict(v=v, lnv=lvs, x=x, R=Rs, fm=fm, fp=fp, rho=rho, q=q, Q=Q, qm=qm, W=W, status=int(sol.status), m_minus=m_minus, dm_minus=dm_minus,
               p=p, beta=beta, v0=v0, dm_plus=dm_plus, eps0_tail=eps0, m_ref=geom.m_ref, qmf=qmf)
    tab["lnrho"] = CubicSpline(lvs, np.log(rho))
    tab["lnQ"] = CubicSpline(lvs, LQ)
    tab["Rsp"] = CubicSpline(lvs, Rs)
    tab["lnfm"] = CubicSpline(lvs, np.log(-fm))
    tab["lnfp"] = CubicSpline(lvs, np.log(-fp))
    return tab


def shell_summary(geom, tab, kappa0):
    v, q, rho, fm, fp = tab["v"], tab["Q"], tab["rho"], tab["fm"], tab["fp"]
    dlnq = np.gradient(np.log(q), v)
    say(f"  shell: status {tab['status']}, v in [{v[0]:.3g}, {v[-1]:.4g}], R_Sigma {tab['R'][0]:.4f} -> {tab['R'][-1]:.6f}; kappa_-(background) = {kappa0:+.3e}")
    for vv in (60, 90, 200, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11):
        if vv > v[-1]:
            break
        i = int(np.argmin(np.abs(v - vv)))
        say(f"    v = {v[i]:9.4g}: x = {tab['x'][i]:.3e}, f_- = {fm[i]:+.3e}, f_+ = {fp[i]:+.3e}, 1/rho = {1/rho[i]:.3e}, "
            f"Q = m_+ - m_- = {q[i]:.4e}, d ln Q/dv = {dlnq[i]:+.4e} (Ori: |kappa| - (p+1)/v = {abs(kappa0) - (tab['p']+1)/v[i]:+.4e})")


# ----------------------------------------------------------------------------- observer
def make_observer(geom, tab):
    """Geodesic in first-order Hamiltonian form: state (v, r, u_v, u_r) with u_r = e^psi vdot, u_v = -e^{2psi} f vdot + e^psi rdot.
    d u_v/dtau = 1/2 d_v(g_ab) u^a u^b,  d u_r/dtau = 1/2 d_r(g_ab) u^a u^b  (turning points handled without square roots)."""
    m_minus, dm_minus = tab["m_minus"], tab["dm_minus"]

    def geom_minus(v, r):
        qm = tab["qmf"](v)
        f, fr, frr, D, M = geom.exact(qm, r)
        return float(f), float(fr), float(frr), float(dm_minus(v) * D), float(M), 1.0, 0.0

    def geom_plus(v, r):
        lv = np.log(v)
        Q = np.exp(tab["lnQ"](lv))
        q = Q + tab["qmf"](v)
        dq = Q * tab["lnQ"](lv, 1) / v + dm_minus(v)
        f, fr, frr, D, M = geom.exact(q, r)
        rho = np.exp(tab["lnrho"](lv))
        drho = rho * tab["lnrho"](lv, 1) / v
        return float(f), float(fr), float(frr), float(dq * D), float(M), float(rho), float(drho)

    def velocities(f, rho, uv, ur):
        vd = ur / rho
        rd = uv / rho + f * ur
        return vd, rd

    def rhs_gen(gfun, y):
        v, r, uv, ur, xr, xrd, xp, xpd, Jr, Dr, Jp, Dp = y
        f, fr, frr, fv, M, rho, drho = gfun(v, r)
        vd, rd = velocities(f, rho, uv, ur)
        Erad = frr / 2
        Eperp = fr / (2 * r) + rho * vd**2 * (-fv / (2 * r))
        duv = 0.5 * (-(2 * rho * drho * f + rho**2 * fv) * vd**2 + 2 * drho * vd * rd)
        dur = -0.5 * rho**2 * fr * vd**2
        return [vd, rd, duv, dur, xrd, -Erad * xr, xpd, -Eperp * xp, Erad, Jr, Eperp, Jp]

    return geom_minus, geom_plus, velocities, (lambda t, y: rhs_gen(geom_minus, y)), (lambda t, y: rhs_gen(geom_plus, y))


def kretsch(f, fr, frr, r):
    with np.errstate(over='ignore'):
        return float(np.float64(frr)**2 + 4 * np.float64(fr)**2 / r**2 + 4 * (1 - np.float64(f))**2 / r**4)


def run_observer(geom, tab, v_e, K_cut=K_PL_MAX, r_floor=2e-3, tau_max=60.0):
    geom_minus, geom_plus, velocities, rhs_minus, rhs_plus = make_observer(geom, tab)
    m_minus = tab["m_minus"]
    q_e = tab["qmf"](v_e)
    Rp = geom.outer_horizon(q_e)
    # E = 1 at the outer horizon (f = 0): u_v = -E = -1, vdot = 1/(2E) = 1/2 -> u_r = 1/2
    y0 = [v_e, Rp, -1.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lnv_min, lnv_max = tab["lnv"][0], tab["lnv"][-1]

    def ev_shell(tau, y):
        lv = math.log(y[0])
        if lv < lnv_min:
            return y[1] - 10.0
        return y[1] - float(tab["Rsp"](min(lv, lnv_max)))
    ev_shell.terminal = True
    ev_shell.direction = -1

    def ev_floor(tau, y):
        return y[1] - r_floor
    ev_floor.terminal = True

    def ev_turn(tau, y):
        f = geom_minus(y[0], y[1])[0]
        return velocities(f, 1.0, y[2], y[3])[1]
    ev_turn.terminal = True
    ev_turn.direction = 1

    sol = solve_ivp(rhs_minus, (0, tau_max), y0, method="LSODA", rtol=1e-10, atol=1e-13, events=[ev_shell, ev_floor, ev_turn], dense_output=True, max_step=0.02)
    yA = sol.y[:, -1]
    tauA = sol.t[-1]
    Ks = []
    for tt in np.linspace(0, tauA, 600):
        yy = sol.sol(tt)
        f, fr, frr, fv, M, _, _ = geom_minus(yy[0], yy[1])
        Ks.append(kretsch(f, fr, frr, yy[1]))
    fA = geom_minus(yA[0], yA[1])[0]
    normA = -fA * yA[3]**2 + 2 * yA[3] * velocities(fA, 1.0, yA[2], yA[3])[1]     # should be -1
    out = dict(v_e=v_e, R_plus=float(Rp), crossed=bool(sol.t_events[0].size), status_minus=int(sol.status), norm_check_minus=float(normA),
               tau_minus=float(tauA), r_end_minus=float(yA[1]), v_end_minus=float(yA[0]), K_max_minus=float(np.max(Ks)),
               Jr_minus=float(yA[8]), Dr_minus=float(yA[9]), Jp_minus=float(yA[10]), Dp_minus=float(yA[11]), xr_minus=float(yA[4]), xp_minus=float(yA[6]))
    if not out["crossed"]:
        why = "turning point" if sol.t_events[2].size else ("r floor" if sol.t_events[1].size else f"tau_max/status {sol.status}")
        out.update(tau_total=float(tauA), Jr=float(yA[8]), Dr=float(yA[9]), Jp=float(yA[10]), Dp=float(yA[11]), xr=float(yA[4]), xp=float(yA[6]),
                   r_end=float(yA[1]), v_end=float(yA[0]), K_max=float(np.max(Ks)), end=f"R- only, no shell crossing ({why})", per_mass={})
        return out
    v, r, uv, ur = yA[:4]
    f, fr, frr, fv, M, _, _ = geom_minus(v, r)
    lv = math.log(v)
    rho = float(np.exp(tab["lnrho"](lv)))
    fm = -float(np.exp(tab["lnfm"](lv)))
    fp = -float(np.exp(tab["lnfp"](lv)))
    vd, rd = velocities(f, 1.0, uv, ur)
    E = -uv
    udot = 1.0 / (abs(fm) * vd)
    dM = 0.5 * r * (fm - fp)
    I_sh = dM * udot / r**2
    gp = geom_plus(v, r)
    Kp = kretsch(gp[0], gp[1], gp[2], r)
    out.update(v_cross=float(v), r_cross=float(r), f_minus_cross=float(fm), f_plus_cross=float(fp), inv_rho=float(1 / rho), dM_jump=float(dM),
               I_shell=float(I_sh), E_cross=float(E), vdot_cross=float(vd), K_plus_at_shell=float(Kp), tau_cross=float(tauA), K_minus_at_shell=float(kretsch(f, fr, frr, r)))
    # kinematics behind the shell: vdot continuous (same v), u_r^+ = rho vdot, rdot^+ = f_- vdot/2 - 1/(2 rho vdot), u_v^+ = -rho^2 f_+ vdot + rho rdot^+
    rd_p = fm * vd / 2 - 1.0 / (2 * rho * vd)
    ur_p = rho * vd
    uv_p = -rho**2 * fp * vd + rho * rd_p
    yB = list(map(float, yA))
    yB[2], yB[3] = uv_p, ur_p
    yB[7] = yA[7] - I_sh * yA[6]
    yB[10] = yA[10] + I_sh
    scale = max(rho, 1e-250)

    def rhs_s(s, y):
        return [scale * d for d in rhs_plus(s * scale, y)]

    def ev_planck(s, y):
        g = geom_plus(y[0], y[1])
        return math.log(K_cut) - math.log(max(kretsch(g[0], g[1], g[2], y[1]), 1e-300))
    ev_planck.terminal = True

    def ev_floor2(s, y):
        return y[1] - r_floor
    ev_floor2.terminal = True

    def ev_vmax(s, y):
        return lnv_max - 1e-3 - math.log(y[0])
    ev_vmax.terminal = True

    def ev_turn2(s, y):
        g = geom_plus(y[0], y[1])
        return velocities(g[0], g[5], y[2], y[3])[1]
    ev_turn2.terminal = True
    ev_turn2.direction = 1

    sol2 = solve_ivp(rhs_s, (0, tau_max / scale), yB, method="LSODA", rtol=1e-10, atol=1e-14, events=[ev_planck, ev_floor2, ev_vmax, ev_turn2], dense_output=True, max_step=0.01)
    yC = sol2.y[:, -1]
    tauB = sol2.t[-1] * scale
    ends = ["Planck cut (1e9 Msun)", "r floor", "table end", "turning point (bounce)"]
    which = [k for k in range(4) if sol2.t_events[k].size]
    end = ends[which[0]] if which else f"tau_max (status {sol2.status})"
    ss = np.linspace(0, sol2.t[-1], 1500)
    Ks2 = []
    for s_ in ss:
        yy = sol2.sol(s_)
        g = geom_plus(yy[0], yy[1])
        Ks2.append(kretsch(g[0], g[1], g[2], yy[1]))
    Ks2 = np.array(Ks2)
    per_mass = {}
    for name, Ms in MASSES.items():
        Kp_ = K_planck(Ms)
        if Kp >= Kp_:
            per_mass[name] = dict(where="at the seam (R+ side of the shell already trans-Planckian)", tau_plus=0.0, r=float(r), Jr=float(yA[8]), Dr=float(yA[9]),
                                  Jp=float(yA[10]), Dp=float(yA[11]), xr=float(yA[4]), xp=float(yA[6]), Jp_with_shell=float(yA[10] + I_sh))
            continue
        idx = np.nonzero(Ks2 >= Kp_)[0]
        if idx.size and idx[0] > 0:
            sfun = lambda s_: math.log(kretsch(*geom_plus(sol2.sol(s_)[0], sol2.sol(s_)[1])[:3], sol2.sol(s_)[1])) - math.log(Kp_)
            s_c = brentq(sfun, ss[idx[0] - 1], ss[idx[0]])
            yy = sol2.sol(s_c)
            per_mass[name] = dict(where="inside R+", tau_plus=float(s_c * scale), r=float(yy[1]), Jr=float(yy[8]), Dr=float(yy[9]), Jp=float(yy[10]), Dp=float(yy[11]), xr=float(yy[4]), xp=float(yy[6]))
        else:
            per_mass[name] = dict(where=f"not reached ({end})", tau_plus=float(tauB), r=float(yC[1]), Jr=float(yC[8]), Dr=float(yC[9]), Jp=float(yC[10]), Dp=float(yC[11]), xr=float(yC[4]), xp=float(yC[6]))
    fC = geom_plus(yC[0], yC[1])
    vdC, rdC = velocities(fC[0], fC[5], yC[2], yC[3])
    normC = -fC[5]**2 * fC[0] * vdC**2 + 2 * fC[5] * vdC * rdC
    out.update(end=end, tau_plus=float(tauB), tau_total=float(tauA + tauB), r_end=float(yC[1]), v_end=float(yC[0]), K_max=float(max(np.max(Ks), np.max(Ks2))),
               Jr=float(yC[8]), Dr=float(yC[9]), Jp=float(yC[10]), Dp=float(yC[11]), xr=float(yC[4]), xp=float(yC[6]), per_mass=per_mass, norm_check_plus=float(normC),
               Jp_plus=float(yC[10] - yB[10]), Jr_plus=float(yC[8] - yA[8]), Dp_plus=float(yC[11] - yA[11]), Dr_plus=float(yC[9] - yA[9]), status_plus=int(sol2.status))
    return out


def fmt(o):
    if not o.get("crossed"):
        return (f"    v_e = {o['v_e']:9.4g}: no shell crossing; tau = {o['tau_total']:.3f}, r_end = {o['r_end']:.3f}, "
                f"J_perp = {o['Jp']:+.3e}, Delta_perp = {o['Dp']:+.3e}, Delta_rad = {o['Dr']:+.3e}, xi_perp = {o['xp']:.3f}, xi_rad = {o['xr']:.3f}, K_max = {o['K_max']:.3g}")
    return (f"    v_e = {o['v_e']:9.4g}: cross v = {o['v_cross']:.5g}, r = {o['r_cross']:.5f}, tau {o['tau_cross']:.3f}; 1/rho = {o['inv_rho']:.3e}, [M] = {o['dM_jump']:.3e}, "
            f"I_sh = {o['I_shell']:.3e} | R-: J_perp {o['Jp_minus']:+.3e}, D_perp {o['Dp_minus']:+.3e}, D_rad {o['Dr_minus']:+.3e}, K_max {o['K_max_minus']:.3g} | "
            f"R+: {o['end']} r = {o['r_end']:.3e}, tau_+ = {o['tau_plus']:.3e}, J_perp {o['Jp_plus']:+.3e}, J_rad {o['Jr_plus']:+.3e}, D_perp {o['Dp_plus']:+.3e}, D_rad {o['Dr_plus']:+.3e} | "
            f"total D_perp {o['Dp']:+.3e}, D_rad {o['Dr']:+.3e}, xi_perp {o['xp']:+.3e}, xi_rad {o['xr']:+.3e}, K_max {o['K_max']:.3g}, K_+(shell) {o['K_plus_at_shell']:.3g}, v_end {o['v_end']:.6g}, norm {o['norm_check_minus']:+.6f}/{o['norm_check_plus']:+.6f}")


def eps0_for_kappa(kappa_target):
    """detuning eps0 (type a) such that the ideal near-root model has kappa_- = kappa_target."""
    if kappa_target == 0.0:
        return 0.0
    g = lambda le: math.log(abs(ScenGeomQuiet(math.exp(le)).kappa_root(0.0))) - math.log(abs(kappa_target))
    le = brentq(g, math.log(1e-16), math.log(1e-2), xtol=1e-10)
    return math.exp(le)


class ScenGeomQuiet(ScenGeom):
    def __init__(self, eps0):
        global say
        s0 = say
        say = lambda *_a, **_k: None
        try:
            super().__init__(eps0, "tmp")
        finally:
            say = s0


# ----------------------------------------------------------------------------- driver
def run_case(name, geom, kappa0, entrants, tail_eps=1e-3, v0=50.0, v_end=1e12, R0=None, dm_plus=1e-4):
    say(f"\n=== {name}: tail amplitude {tail_eps}, p = 11, v0 = {v0}, m_+(v0) = m_-(v0)(1 + {dm_plus}) ===")
    tab = solve_shell(geom, eps0=tail_eps, v0=v0, v_end=v_end, R0=R0, dm_plus=dm_plus)
    shell_summary(geom, tab, kappa0)
    rows = []
    for ve in entrants:
        if ve > tab["v"][-1] * 0.999 or np.exp(tab["lnQ"](np.log(ve))) > 1e120:
            say(f"    v_e = {ve:9.4g}: skipped (beyond the table or q > 1e120 -> curvature overflow; the classical description ended long before)")
            continue
        o = run_observer(geom, tab, ve)
        rows.append(o)
        say(fmt(o))
    say(f"  [{time.time() - T0:.0f} s]")
    return dict(name=name, kappa0=kappa0, tail_eps=tail_eps, v0=v0, dm_plus=dm_plus, rows=rows, shell=dict(v=tab["v"].tolist(), inv_rho=(1 / tab["rho"]).tolist(), q=tab["q"].tolist(), fm=tab["fm"].tolist(), R=tab["R"].tolist()))


def main():
    results = {}
    say("T34: tidal cost of crossing the inner horizon in the two-flux Ori model (linearised law; E = 1 radial observer)")
    # ---- RN control
    rn = RNGeom(0.9)
    say(f"RN e = 0.9: r_- = {rn.rminus(1):.5f}, r_+ = {rn.rplus(1):.5f}, kappa_- = {rn.kappa_root():+.4f}")
    results["RN"] = run_case("RN e=0.9 (Ori control)", rn, rn.kappa_root(), [20, 22, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400], v0=20.0, v_end=1e4)
    # ---- scenario: kappa_- = -1e-3, -1e-8, 0
    for kt in (-1e-3, -1e-8, 0.0):
        e0 = eps0_for_kappa(kt)
        geom = ScenGeom(e0, f"scenario kappa={kt:+.0e}")
        k0 = geom.kappa_root(0.0)
        say(f"  ideal model: kappa_-(eps0) = {k0:+.4e} (target {kt:+.0e}), root r_- = {geom.shell_root(0.0):.6f}")
        if kt == -1e-3:
            ent = [50, 55, 60, 80, 100, 200, 500, 1e3, 3e3, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5, 5e5]
        elif kt == -1e-8:
            ent = [50, 55, 60, 80, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11]
        else:
            ent = [50, 55, 60, 80, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
        results[f"scen_{kt:+.0e}"] = run_case(f"scenario kappa_- = {kt:+.0e} (eps0 = {e0:.3e})", geom, k0, ent)
    # ---- starting cost: amplitude and v0 dependence (kappa = -1e-3 and 0)
    say("\n=== starting cost: entrant at v_e = v0 (first crossing) for tail amplitudes 1e-2, 1e-3, 1e-4 and v0 = 20, 50, 100 ===")
    start = {}
    for kt in (-1e-3, 0.0):
        e0 = eps0_for_kappa(kt)
        geom = ScenGeomQuiet(e0)
        for amp in (1e-2, 1e-3, 1e-4):
            for v0 in (20.0, 50.0, 100.0):
                tab = solve_shell(geom, eps0=amp, v0=v0, v_end=1e6)
                o = run_observer(geom, tab, v0)
                o2 = run_observer(geom, tab, v0 * 2)
                key = f"kappa={kt:+.0e}, amp={amp:.0e}, v0={v0:.0f}"
                start[key] = dict(first=o, double=o2)
                say(f"  {key}: first crossing: 1/rho = {o.get('inv_rho', float('nan')):.4e}, I_sh = {o.get('I_shell', float('nan')):.4e}, [M] = {o.get('dM_jump', float('nan')):.3e}, "
                    f"D_perp = {o['Dp']:+.4e}, D_rad = {o['Dr']:+.4e}, xi_perp = {o['xp']:.4f}, xi_rad = {o['xr']:.4f}, end {o['end']} at r = {o['r_end']:.3e}; "
                    f"v_e = 2 v0: 1/rho = {o2.get('inv_rho', float('nan')):.4e}, I_sh = {o2.get('I_shell', float('nan')):.4e}, D_perp = {o2['Dp']:+.4e}")
    results["start"] = start
    json.dump(results, open(OUT / "T34_delta.json", "w", encoding="utf-8"), indent=1, default=float)
    (LOGS / "T34_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time() - T0:.0f} s)")


if __name__ == "__main__":
    main()
