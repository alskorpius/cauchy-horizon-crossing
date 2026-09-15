"""T35 part A: does the finite, epoch-independent tidal deformation of T34 survive a continuous outflux?

The single outgoing null shell of T34 (all of the backscattered energy dm_+ = 1e-4 on the ray u = v0) is replaced by N outgoing
null shells Sigma_k, k = 1..N: a discretisation of the Poisson-Israel outflux L_out(u) ~ u^{-(p+1)}, p = 11, on u in [v0, 4 v0]
(geometric grid in u; cell weights from the exact CDF; the 4^{-11} = 2.4e-7 of the tail beyond 4 v0 is dropped).  Ray u is launched
(backscattered) at the radius R0 = 1.33 M (RN: mid-way between the horizons) at advanced time v = u with mass-parameter jump
Q_k(u) = w_k dm_+ m_-(u); N = 1 is exactly the T34 configuration.  At each launch the region inside the new shell acquires the
extra mass parameter Q_k (an infinitesimal ingoing shell, <= 1e-4 at launch).
Chain of regions (outer to inner): R_- (pure ingoing tail, mass m_-(v)) | Sigma_N (latest ray) | ... | Sigma_1 (ray u = v0) | inside.
Each region is generalised Vaidya in its own advanced time; the linear mass law f(m, r) = f(m_-, r) + (m - m_-) D(r) (RN exact,
scenario: T2/T34 linearised law) makes the region geometry outside shell k  f_out,k = f_-(v, r) + S_k(v) D(r), S_k = sum of the
jumps Q_i of the shells outside k.  Junction on each shell (Barrabes-Israel, CR 2021 eq. (32)): dm_in/dv_out = dm_out/dv_out f_in/f_out
and dv_in/dv_out = rho_k = f_out/f_in; in the common label v of R_-:
    dQ_k/dv = (Q_k D/f_out,k) (dm_-/dv + dS_k/dv) / P_out,k,     dR_k/dv = P_out,k f_out,k / 2,     P_out,k = prod_{i outside k} rho_i
(telescoping: coincident shells reproduce the single shell exactly).  Shells are integrated sequentially from the outermost inward
in the variables (ln x, ln Q) with x = R - r_root of the region *outside* the shell (root of f_out,k; vectorised Newton on the
Taylor model of T34) while S_k <= S_SW = 1e-8, and in the absolute radius R with exact evaluation afterwards (no cancellation once
|S D| >> rounding; the region root then runs inward as S inflates and the frozen shell is left far from it).  The motion of the
reference root is the analytic tail-driven part dr/dq * dm_-/dv plus a finite difference of (r_root,region - r_root,tail).
RN control: the launch window is u in [v0, 1.15 v0] because the earlier shells reach f ~ 1e-8 within ~10 M there (|kappa| = 1.37)
and the launch bookkeeping (an ingoing shell of energy Q_k at v = u_k, unavoidable for null dust: a shell cannot be created at a
vertex, DTR) would otherwise inflate them by exp(Q_k D/f_out) — negligible only while the earlier shells are still at |f| >> Q_k.
Observer (T34): radial timelike geodesic, E = 1 at the outer horizon, entering at advanced time v_e after all launches; it crosses
the shells in sequence.  Behind a shell of blueshift 1/rho the observer is nearly an ingoing null ray (dv_common/dtau ~ P), so the
common label v is frozen and every region is a static snapshot at the crossing label; the crossing kinematics and the transverse
impulse I_k = [M]_k |udot| / R_k^2 are those of T34.  The local energy is carried as ln(eps); the deviation variables are scaled by
eps (zeta = xi'/eps, Jhat = J/eps, ds = eps dtau).  Bookkeeping: Delta = int int E dtau^2, xi from xi'' = -E xi.
Units G = c = M = 1.  Run: .venv/Scripts/python.exe experiments/cycle2/T35A_multishell.py [Nmax]
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import T34_tidal as t34  # noqa: E402  (scenario/RN geometries, constants)

LOG = []
LAST = []
T0 = time.time()
S_SW = 1e-8        # switch of the shell variables (region-root x -> absolute R with exact evaluation); the region root moves by (S/|a|)^{1/3} ~ 2e-3 at the switch


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


K_PL = {name: t34.K_planck(M) for name, M in t34.MASSES.items()}
LNK_CUT = math.log(K_PL["1e9 Msun"])


# ----------------------------------------------------------------------------- tail region R_-
class Tail:
    """m_-(v) = m0 - beta/v^p; roots and accurate f for mass parameters q (tail) or q + S (region outside a shell)."""

    def __init__(self, geom, p=11, eps0=1e-3, v0=50.0, v_end=1e12):
        self.geom, self.p, self.v0 = geom, p, v0
        self.beta = eps0 * v0**p
        self.isRN = isinstance(geom, t34.RNGeom)
        self.qmf = (lambda v: 1.0 - self.beta / v**p) if self.isRN else (lambda v: -self.beta / v**p)
        self.dq = lambda v: p * self.beta / v ** (p + 1)
        if not self.isRN:
            g = geom
            base = P.polyadd(g.P0, g.eps0 * g.Pa)
            self.cbase = np.pad(base, (0, len(g.Plin) - len(base)))
            self.Plin = g.Plin
            self.dPlin = P.polyder(g.Plin)
            self.ddPlin = P.polyder(g.Plin, 2)

    # --- vectorised root + Taylor model of f(q_eff, r) around its root near r0 (scenario)
    def roots_taylor(self, q_eff):
        q_eff = np.asarray(q_eff, float)
        c = self.cbase[None, :] + q_eff[:, None] * self.Plin[None, :]
        y = np.cbrt(-c[:, 0] / self.geom.a3)
        cT = c.T
        dcT = P.polyder(cT)
        for _ in range(80):
            fv = P.polyval(y, cT, tensor=False)
            dv = P.polyval(y, dcT, tensor=False)
            step = np.where(dv != 0, fv / dv, 0.0)
            y = y - step
            if np.max(np.abs(step)) < 1e-17:
                break
        tay = np.zeros_like(c)
        d = cT
        for k in range(c.shape[1]):
            tay[:, k] = P.polyval(y, d, tensor=False) / math.factorial(k)
            d = P.polyder(d) if k < c.shape[1] - 1 else d
        tay[:, 0] = 0.0
        dyr = -P.polyval(y, self.Plin) / tay[:, 1]
        return y, tay, dyr

    def root_tail(self, v):
        return self.geom.rminus(self.qmf(v)) if self.isRN else self.geom.r0 + float(self.roots_taylor([self.qmf(v)])[0][0])

    def fields(self, v, r, S=0.0):
        """f, f', f'' of the region with mass parameter q_-(v) + S at radii r (array), and D, D', D''."""
        r = np.atleast_1d(np.asarray(r, float))
        q = self.qmf(v) + S
        if self.isRN:
            f, fr, frr, D, _ = self.geom.exact(q, r)
            return f, fr, frr, D, 2 / r**2, -4 / r**3
        g = self.geom
        y = r - g.r0
        f, fr, frr, D, _ = g.exact(q, r)
        m, mr, mrr, G, Gr, Grr = t34.base_fields(r)
        Dr = 2 * G / r**2 - 2 * Gr / r
        Drr = -4 * G / r**3 + 4 * Gr / r**2 - 2 * Grr / r
        near = (np.abs(y) < g.Y_SW) & (abs(S) <= S_SW)
        if near.any():
            yr, tay, _ = self.roots_taylor([q])
            c = tay[0]
            x = y[near] - yr[0]
            f = f.copy(); fr = fr.copy(); frr = frr.copy(); D = D.copy(); Dr = Dr.copy(); Drr = Drr.copy()
            f[near] = P.polyval(x, c)
            fr[near] = P.polyval(x, P.polyder(c))
            frr[near] = P.polyval(x, P.polyder(c, 2))
            D[near] = P.polyval(y[near], self.Plin)
            Dr[near] = P.polyval(y[near], self.dPlin)
            Drr[near] = P.polyval(y[near], self.ddPlin)
        return f, fr, frr, D, Dr, Drr

    def outer_horizon(self, v):
        return self.geom.outer_horizon(self.qmf(v))


def u_grid(v0, N, p, umax_fac=4.0):
    """geometric u-grid on [v0, umax] and cell weights from the CDF of u^{-(p+1)} (normalised on the grid range)."""
    if N == 1:
        return np.array([v0]), np.array([1.0])
    u = v0 * umax_fac ** (np.arange(N) / (N - 1))
    edges = np.concatenate([[u[0]], np.sqrt(u[1:] * u[:-1]), [u[-1] * math.sqrt(u[-1] / u[-2])]])
    F = edges ** (-p)
    w = F[:-1] - F[1:]
    return u, w / w.sum()


# ----------------------------------------------------------------------------- shell chain
def safe_solve(rhs, l0, l1, y0, t_eval, events, lv):
    """solve_ivp with a retry: if the rhs raises (non-finite state), the span is cut back to the last good time and rerun;
    a shell whose integration cannot proceed keeps a truncated table (the chain is then valid only up to that v)."""
    end = l1
    for _ in range(6):
        try:
            return solve_ivp(rhs, (l0, end), y0, method="LSODA", rtol=1e-9, atol=1e-12, t_eval=t_eval[t_eval <= end], events=events)
        except FloatingPointError:
            bad = LAST[0] if LAST and np.isfinite(LAST[0]) else end
            end = min(end, bad) - 3 * (lv[1] - lv[0])
            if end <= l0 + (lv[1] - lv[0]):
                break
    class _Empty:  # noqa: D401
        t = np.array([l0]); y = np.array([[y0[0]], [y0[1]]]); status = 1
    return _Empty()


def solve_chain(tail, us, ws, R0, dm_plus=1e-4, v_end=1e12, npts=3000, L_max=600.0):
    """Sequential integration of the shells from the outermost (latest launch) inward.  Per-shell tables on a common ln v grid:
    R (radius), lnQ, lnfout (ln|f_out| at the shell), lnrho; nan before launch / after the table end."""
    v0 = tail.v0
    N = len(us)
    lv = np.linspace(math.log(v0), math.log(v_end), npts)
    v = np.exp(lv)
    q_tail = tail.qmf(v)
    dq_tail = tail.dq(v)
    if tail.isRN:
        root_tail = tail.geom.rminus(q_tail)
        droot_tail = tail.geom.shell_droot_dq(q_tail)
    else:
        yr, tay_t, dyr = tail.roots_taylor(q_tail)
        root_tail = tail.geom.r0 + yr
        droot_tail = dyr
    order = np.argsort(-us)
    R_tab = np.full((N, npts), np.nan)
    lnQ_tab = np.full((N, npts), np.nan)
    lnf_tab = np.full((N, npts), np.nan)
    lnrho_tab = np.zeros((N, npts))
    i_launch = np.zeros(N, int)
    S_out = np.zeros(npts)
    lnP_out = np.zeros(npts)
    valid_to = npts - 1
    g = tail.geom
    for n, j in enumerate(order):
        u, w = us[j], ws[j]
        i0 = int(np.argmin(np.abs(lv - math.log(u))))
        i_launch[j] = i0
        if i0 >= valid_to:
            continue
        Q0 = w * dm_plus * (q_tail[i0] if tail.isRN else 1.0 + q_tail[i0])
        dS_tab = np.gradient(S_out, v)          # derivative of the (linearly interpolated) outer mass; consistent to O(h^2)
        q_eff = q_tail + S_out
        # phase-1 reference: root of the region outside the shell (S <= S_SW)
        if tail.isRN:
            root_reg = tail.geom.rminus(q_eff); droot_reg = tail.geom.shell_droot_dq(q_eff); tay_reg = None
        else:
            yr_r, tay_reg, dyr_r = tail.roots_taylor(q_eff)
            root_reg = g.r0 + yr_r; droot_reg = dyr_r
        # motion of the region root: analytic tail-driven part (resolvable below rounding) + finite-difference part driven by the
        # outer shells' mass S (exactly zero before launches; the linearised dr/dq * dS would overshoot at kappa ~ 0 where r_root ~ q^{1/3})
        droot_dv = droot_tail * dq_tail + np.gradient(root_reg - root_tail, v)
        above = np.nonzero(S_out[i0:] > S_SW)[0]
        i_sw = i0 + int(above[0]) if above.size else None      # first index in phase 2

        def interp(l, arr):
            return float(np.interp(l, lv, arr))

        def f_D_phase1(l, x):
            vv = math.exp(l)
            if tail.isRN:
                q = interp(l, q_eff)
                return g.shell_f(q, x), g.shell_D(q, x)
            yr = interp(l, root_reg) - g.r0
            y = yr + x
            if abs(y) < g.Y_SW:
                c = np.array([interp(l, tay_reg[:, k]) for k in range(tay_reg.shape[1])]); c[0] = 0.0
                return float(P.polyval(x, c)), float(P.polyval(y, tail.Plin))
            f, fr, frr, D, _ = g.exact(interp(l, q_eff), g.r0 + y)
            return float(f), float(D)

        def f_D_phase2(l, R):
            # phase 2 (S > S_SW): absolute radius, exact evaluation (|S D| >> rounding, no cancellation)
            S = interp(l, S_out)
            q = interp(l, q_tail)
            if tail.isRN:
                f, fr, frr, D, _ = g.exact(q, R)
                return float(f) + S * float(D), float(D)
            f, fr, frr, D, _ = g.exact(q, R)
            return float(f) + S * float(D), float(D)

        def make_rhs(phase):
            def rhs(l, y):
                LAST[:] = [l, list(y)]
                if not (np.isfinite(l) and np.isfinite(y).all()) or l > lv[-1] + 1.0 or (phase == 1 and y[0] > 3.0):
                    raise FloatingPointError("integrator excursion")   # caught by safe_solve: the table is truncated before it
                vv = math.exp(min(l, lv[-1] + 1.0))
                Q = math.exp(min(y[1], 700.0))
                dS = interp(l, dS_tab); lnP = interp(l, lnP_out)
                dq = interp(l, dq_tail) + dS
                if phase == 1:
                    x = math.exp(max(y[0], -700.0))
                    fout, D = f_D_phase1(l, x)
                    fout = min(fout, -1e-300)
                    dro = interp(l, droot_dv)
                    dpos = (math.exp(lnP) * fout / 2 - dro) / x          # d ln x / dv
                else:
                    fout, D = f_D_phase2(l, max(y[0], 1e-2))
                    fout = min(fout, -1e-300)
                    dpos = max(math.exp(lnP + math.log(-fout)) / 2 * -1.0, -1e3) if lnP + math.log(-fout) < 700 else -1e3   # dR / dv (guarded)
                dlQ = min(dq * D / fout, 1e3)
                return [vv * dpos, vv * dlQ]
            return rhs

        def ev_L(l, y):
            return L_max - min(y[1], 1e6)
        ev_L.terminal = True

        def ev_x(l, y):
            return max(y[0], -1e6) + 600.0
        ev_x.terminal = True

        def ev_none(l, y):
            return y[0] - 5e-2                       # phase 2: stop if the shell (frozen in practice) is driven below r = 0.05 M by rounding
        ev_none.terminal = True

        x0 = R0 - root_reg[i0]
        y0 = [math.log(x0), math.log(Q0)]
        i_end1 = valid_to if i_sw is None else min(i_sw, valid_to)
        segs = []
        if i_end1 > i0:
            sol = safe_solve(make_rhs(1), lv[i0], lv[i_end1], y0, lv[i0:i_end1 + 1], [ev_L, ev_x], lv)
            segs.append((1, i0, sol))
            done = sol.status == 1
            if not done and i_sw is not None and i_sw < valid_to:
                R_end = root_reg[i_end1] + math.exp(max(sol.y[0][-1], -700.0))
                sol2 = safe_solve(make_rhs(2), lv[i_end1], lv[valid_to], [R_end, sol.y[1][-1]], lv[i_end1:valid_to + 1], [ev_L, ev_none], lv)
                segs.append((2, i_end1, sol2))
        else:
            sol2 = safe_solve(make_rhs(2), lv[i0], lv[valid_to], [R0, math.log(Q0)], lv[i0:valid_to + 1], [ev_L, ev_none], lv)
            segs.append((2, i0, sol2))
        last = i0
        for phase, ia, sol in segs:
            good = np.isfinite(sol.y).all(axis=0)
            k = int(np.argmin(good)) if not good.all() else sol.t.size
            for i in range(k):
                ii = ia + i
                if phase == 1:
                    x = math.exp(max(sol.y[0][i], -700.0))
                    fout, D = f_D_phase1(lv[ii], x)
                    R_tab[j, ii] = root_reg[ii] + x
                else:
                    fout, D = f_D_phase2(lv[ii], sol.y[0][i])
                    R_tab[j, ii] = sol.y[0][i]
                fin = fout + math.exp(min(sol.y[1][i], 700.0)) * D
                lnQ_tab[j, ii] = sol.y[1][i]
                lnf_tab[j, ii] = math.log(max(-fout, 1e-300))
                lnrho_tab[j, ii] = math.log(fout / fin) if fin < 0 and fout < 0 else 0.0
                last = ii
            if sol.status == 1 or not good.all():
                break
        S_out[i0:last + 1] += np.exp(np.minimum(lnQ_tab[j, i0:last + 1], 700.0))
        lnP_out[i0:last + 1] += lnrho_tab[j, i0:last + 1]
        if last < valid_to:
            valid_to = last
    idx = np.argsort(us)
    d = np.diff(R_tab[idx, : valid_to + 1], axis=0)
    cross = float(np.nanmin(d)) if N > 1 and np.isfinite(d).any() else 1.0
    return dict(lv=lv[: valid_to + 1], v=v[: valid_to + 1], R=R_tab[:, : valid_to + 1], lnQ=lnQ_tab[:, : valid_to + 1], lnf=lnf_tab[:, : valid_to + 1],
                lnrho=lnrho_tab[:, : valid_to + 1], us=us, ws=ws, i_launch=i_launch, order=order, min_dR=cross, v_last=float(v[valid_to]), R0=R0,
                S_out=S_out[: valid_to + 1], lnP_out=lnP_out[: valid_to + 1])


# ----------------------------------------------------------------------------- observer through the chain
def lnK_of(f, fr, frr, r):
    t = np.array([abs(frr), 2 * abs(fr) / r, 2 * abs(1 - f) / r**2])
    m = t.max()
    if m == 0.0:
        return -np.inf
    return 2 * math.log(m) + math.log(float(np.sum((t / m) ** 2)))


def run_observer(tail, ch, v_e, r_floor=2e-3, lnK_cut=LNK_CUT):
    lv_tab, v_tab = ch["lv"], ch["v"]
    N = len(ch["us"])
    order = ch["order"]
    tabs = []
    for j in range(N):
        ok = np.isfinite(ch["R"][j])
        li = lv_tab[ok]; lq = ch["lnQ"][j][ok]
        dlq = np.gradient(lq, v_tab[ok]) if lq.size > 2 else np.zeros_like(lq)
        tabs.append((li, ch["R"][j][ok], lq, dlq, ch["lnf"][j][ok]))

    def R_shell(j, v):
        l = math.log(v)
        li, Rt = tabs[j][0], tabs[j][1]
        if li.size == 0 or l < li[0] or l > li[-1]:
            return -1.0
        return float(np.interp(l, li, Rt))

    y = [v_e, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]        # v, xi_r, zeta_r, xi_p, zeta_p, Jr^, Dr, Jp^, Dp
    ln_eps, S, dS, lnP = 0.0, 0.0, 0.0, 0.0
    r = tail.outer_horizon(v_e)
    rec = dict(v_e=v_e, R_plus=r, crossings=[], end="")
    first_hit = {k: None for k in K_PL}

    def region_fields(v, rr, S):
        f, fr, frr, D, Dr, Drr = tail.fields(v, rr, S)
        return float(f[0]), float(fr[0]), float(frr[0]), float(D[0])

    for step in range(N + 1):
        nxt = order[step] if step < N else None
        if nxt is not None and R_shell(nxt, y[0]) < 0:
            rec["crossings"].append(dict(k=int(nxt), u=float(ch["us"][nxt]), skipped=True, r=r, v=y[0]))
            continue
        Se, lnPe, lne, dSe = S, lnP, ln_eps, dS

        def rhs(rr, yy):
            v = yy[0]
            f, fr, frr, D = region_fields(v, rr, Se)
            e2 = math.exp(-2 * lne)
            w = math.sqrt(max(1.0 - f * e2, 1e-300))
            Erad = frr / 2
            flux = e2 * math.exp(-lnPe) * (-(tail.dq(v) + dSe) * D) / (2 * rr * (1 + w) ** 2)
            Eperp = fr / (2 * rr) + flux
            dv = -e2 * math.exp(-lnPe) / ((1 + w) * w)
            return [dv, -yy[2] / w, Erad * e2 * yy[1] / w, -yy[4] / w, Eperp * e2 * yy[3] / w,
                    -Erad * e2 / w, -yy[5] / w, -Eperp * e2 / w, -yy[7] / w]

        def ev_shell(rr, yy):
            return rr - R_shell(nxt, yy[0])
        ev_shell.terminal = True
        ev_shell.direction = -1

        def ev_turn(rr, yy):
            return 1.0 - region_fields(yy[0], rr, Se)[0] * math.exp(-2 * lne) - 1e-12
        ev_turn.terminal = True

        def ev_planck(rr, yy):
            f, fr, frr, D = region_fields(yy[0], rr, Se)
            return lnK_cut - lnK_of(f, fr, frr, rr)
        ev_planck.terminal = True

        events = ([ev_shell] if nxt is not None else []) + [ev_turn, ev_planck]
        r_stop = max(R_shell(nxt, y[0]) * 0.999, r_floor) if nxt is not None else r_floor
        sol = None
        if nxt is not None and R_shell(nxt, y[0]) >= r:
            pass                                    # the next shell is already at/above the observer (frozen shells within rounding): immediate crossing
        elif r - r_stop > 0:
            sol = solve_ivp(rhs, (r, r_stop), y, method="LSODA", rtol=1e-9, atol=1e-13, events=events, max_step=0.02)
            y = list(sol.y[:, -1]); r = float(sol.t[-1])
        v = y[0]
        f, fr, frr, D = region_fields(v, r, Se)
        lnK_here = lnK_of(f, fr, frr, r)
        for name, Kp in K_PL.items():
            if first_hit[name] is None and lnK_here >= math.log(Kp):
                first_hit[name] = dict(where=f"region {step} (inside {step} shells)", r=r, Dp=y[8], Dr=y[6], xp=y[3], xr=y[1])
        if sol is not None and sol.t_events[-1].size:
            rec["end"] = f"Planck cut (1e9 Msun) in region {step} at r = {r:.4e}"; break
        if sol is not None and sol.t_events[-2].size:
            rec["end"] = f"turning point in region {step} at r = {r:.4e}"; break
        if nxt is None:
            rec["end"] = f"r floor {r_floor} reached (inside all {N} shells)"; break
        if sol is not None and not sol.t_events[0].size:
            rec["crossings"].append(dict(k=int(nxt), u=float(ch["us"][nxt]), skipped=True, r=r, v=v)); continue
        # ---- crossing of shell nxt at (v, r); f_out from the shell's own table (accurate near the root)
        l = math.log(v)
        li, Rt, lq, dlq, lf = tabs[nxt]
        Q = math.exp(min(float(np.interp(l, li, lq)), 700.0))
        dQ = Q * float(np.interp(l, li, dlq))
        fout = -math.exp(float(np.interp(l, li, lf)))
        fin = fout + Q * D
        e2 = math.exp(-2 * ln_eps)
        w = math.sqrt(max(1.0 - fout * e2, 1e-300))
        A_eps = (1 + w) * abs(fin / fout)
        ratio = (A_eps**2 + fin * e2) / (2 * A_eps)
        dM = -0.5 * r * Q * D
        I_eps = dM * (1 + w) / (abs(fout) * r**2)
        y[4] = (y[4] - I_eps * y[3]) / ratio
        y[7] = (y[7] + I_eps) / ratio
        y[2] = y[2] / ratio
        y[5] = y[5] / ratio
        ln_eps_new = ln_eps + math.log(ratio)
        lnrho = math.log(fout / fin)
        fin_fields = region_fields(v, r, S + Q)
        lnK_in = lnK_of(*fin_fields[:3], r)
        rec["crossings"].append(dict(k=int(nxt), u=float(ch["us"][nxt]), r=r, v=v, Q=Q, dM=dM, ln_inv_rho=-lnrho,
                                     ln_I=ln_eps + math.log(max(I_eps, 1e-300)), ln_eps_before=ln_eps, ln_eps_after=ln_eps_new,
                                     lnK_out=lnK_here, lnK_in=lnK_in, Dp=y[8], Dr=y[6], xp=y[3], xr=y[1]))
        ln_eps = ln_eps_new
        S += Q; dS += dQ; lnP += lnrho
        for name, Kp in K_PL.items():
            if first_hit[name] is None and lnK_in >= math.log(Kp):
                first_hit[name] = dict(where=f"at shell u = {ch['us'][nxt]:.4g} (inside face)", r=r, Dp=y[8], Dr=y[6], xp=y[3], xr=y[1])
        if lnK_in >= lnK_cut:
            rec["end"] = f"Planck cut (1e9 Msun) at the inside face of shell u = {ch['us'][nxt]:.4g}, r = {r:.4e}"; break
    rec.update(Dp=y[8], Dr=y[6], xp=y[3], xr=y[1], ln_eps_end=ln_eps, ln_inv_P=-lnP, S_end=S, r_end=r, v_end=y[0],
               n_crossed=sum(1 for c in rec["crossings"] if not c.get("skipped")), per_mass=first_hit)
    return rec


# ----------------------------------------------------------------------------- driver
def run_case(label, geom, kappa0, entrants, Ns, v0, v_end, R0=None, dm_plus=1e-4, tail_eps=1e-3, umax_fac=4.0, detail=False):
    say(f"\n=== {label}: kappa_- = {kappa0:+.3e}, v0 = {v0}, tail 1e-3 p = 11, outflux dm_+ = {dm_plus} on u in [v0, {umax_fac:.4g} v0] ===")
    tail = Tail(geom, eps0=tail_eps, v0=v0, v_end=v_end)
    R0 = R0 or (0.5 * (geom.rminus(1.0) + geom.rplus(1.0)) if tail.isRN else 1.33)
    res = {}
    for N in Ns:
        t1 = time.time()
        us, ws = u_grid(v0, N, 11, umax_fac)
        ch = solve_chain(tail, us, ws, R0, dm_plus=dm_plus, v_end=v_end)
        say(f"  N = {N}: u in [{us[0]:.4g}, {us[-1]:.4g}], heaviest weight {ws.max():.4g} (u = {us[np.argmax(ws)]:.4g}), launch radius R0 = {R0:.3f}; "
            f"table to v = {ch['v_last']:.4g}; min dR between adjacent rays = {ch['min_dR']:+.2e} ({'no crossing' if ch['min_dR'] > 0 else 'CROSSING'}); chain {time.time() - t1:.1f} s")
        j_in = int(np.argmin(us))
        lnQ = ch["lnQ"][j_in]; v = ch["v"]
        ok = np.isfinite(lnQ)
        rate = np.gradient(lnQ[ok], v[ok]) if ok.sum() > 2 else np.zeros(1)
        Ssum = np.log(np.maximum(ch["S_out"], 1e-300))
        rate_S = np.gradient(Ssum, v)
        rows = []
        for ve in entrants:
            if ve > ch["v_last"] * 0.99:
                say(f"    v_e = {ve:9.4g}: beyond the chain table (v_last = {ch['v_last']:.4g}) -> skipped"); continue
            i = int(np.argmin(np.abs(v - ve)))
            t2 = time.time()
            o = run_observer(tail, ch, ve)
            o["rate_innermost_at_ve"] = float(rate[np.argmin(np.abs(v[ok] - ve))]) if ok.sum() > 2 else float("nan")
            o["rate_total_S_at_ve"] = float(rate_S[i]); o["ln_S_at_ve"] = float(Ssum[i])
            rows.append(o)
            cr = [c for c in o["crossings"] if not c.get("skipped")]
            lnI = [c["ln_I"] for c in cr]
            say(f"    v_e = {ve:9.4g}: crossed {o['n_crossed']}/{N} (skipped {len(o['crossings']) - o['n_crossed']}); "
                f"ln(1/P) = {o['ln_inv_P']:.4g}; ln eps_end = {o['ln_eps_end']:.4g}; ln S_end = {math.log(max(o['S_end'], 1e-300)):.4g}; max ln I = {max(lnI) if lnI else float('nan'):.4g}; "
                f"Delta_perp = {o['Dp']:+.6f}, Delta_rad = {o['Dr']:+.6f}, xi_perp = {o['xp']:+.4e}, xi_rad = {o['xr']:+.4f}; "
                f"d ln Q_in/dv = {o['rate_innermost_at_ve']:+.4e}, d ln S/dv = {o['rate_total_S_at_ve']:+.4e} (Ori: {abs(kappa0) - 12 / ve:+.4e}); end: {o['end']} [{time.time() - t2:.0f} s]")
            if detail and cr:
                for c in cr[:3] + ([dict(u="...")] if len(cr) > 6 else []) + cr[-3:]:
                    if c.get("u") == "...":
                        say("        ..."); continue
                    say(f"        u = {c['u']:8.4g}: r = {c['r']:.6f}, ln(1/rho) = {c['ln_inv_rho']:+.3e}, ln I = {c['ln_I']:+.4g}, ln eps -> {c['ln_eps_after']:+.4g}, "
                        f"Delta_perp so far = {c['Dp']:+.5f}, xi_perp = {c['xp']:+.3e}, ln K_in = {c['lnK_in']:+.4g}")
        res[N] = dict(rows=rows, us=us.tolist(), ws=ws.tolist(), v_last=ch["v_last"], min_dR=ch["min_dR"])
    return res


def convergence_table(res, entrants, label):
    say(f"\n  convergence in N ({label}):")
    say("    v_e      | N    | Delta_perp   | Delta_rad    | xi_perp     | xi_rad    | ln(1/P)    | crossed")
    conv = {}
    for ve in entrants:
        vals = []
        for N, d in res.items():
            o = next((o for o in d["rows"] if o["v_e"] == ve), None)
            if o is None:
                continue
            say(f"    {ve:9.4g} | {N:4d} | {o['Dp']:+.6f}    | {o['Dr']:+.6f}    | {o['xp']:+.4e} | {o['xr']:+.4f}   | {o['ln_inv_P']:10.4g} | {o['n_crossed']}")
            vals.append((N, o["Dp"], o["Dr"]))
        if len(vals) >= 3:
            (N1, d1, _), (N2, d2, _), (N3, d3, _) = vals[-3:]
            try:
                a = math.log(abs((d2 - d1) / (d3 - d2))) / math.log(N3 / N2) if (d3 - d2) != 0 and (d2 - d1) != 0 else float("nan")
                dinf = d3 + (d3 - d2) * (N3 / N2) ** (-a) / (1 - (N3 / N2) ** (-a)) if np.isfinite(a) and a > 0 else d3
            except (ValueError, ZeroDivisionError):
                a, dinf = float("nan"), d3
            say(f"      -> Delta_perp: last change {d3 - d2:+.2e} (N {N2} -> {N3}); fitted order a = {a:.2f}; extrapolated Delta_perp(N -> inf) = {dinf:+.6f}")
            conv[str(ve)] = dict(order=a, Dp_inf=dinf, last_change=d3 - d2)
    return conv


def main():
    import os
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    Ns = [n for n in (1, 3, 10, 100, 1000) if n <= Nmax]
    cases = os.environ.get("T35A_CASES", "RN,m3,z,wide").split(",")
    tag = os.environ.get("T35A_TAG", "")
    say(f"T35A: multi-shell (continuous outflux) test of the T34 deformation; N in {Ns}; cases {cases}")
    results = {}
    rn = t34.RNGeom(0.9)
    # RN: the shells approach the root exponentially (|kappa| = 1.37), so the launch window must be short (u in [v0, 1.15 v0] = 3 M)
    # for the launch bookkeeping (an ingoing shell of energy Q_k at v = u_k) to stay negligible for the earlier shells (see report)
    if "RN" in cases:
        results["RN"] = run_case("RN e=0.9 (Ori control)", rn, rn.kappa_root(), [100.0, 150.0], [n for n in Ns if n <= 10], v0=20.0, v_end=1e4, umax_fac=1.15, detail=True)
        results["RN"]["conv"] = convergence_table(results["RN"], [100.0, 150.0], "RN")
    for kt, ent, vend, ck in ((-1e-3, [1e3, 1e4, 1e5], 1e6, "m3"), (0.0, [1e5, 1e8], 1e9, "z")):
        if ck not in cases:
            continue
        e0 = t34.eps0_for_kappa(kt)
        geom = t34.ScenGeom(e0, f"scenario kappa={kt:+.0e}")
        k0 = geom.kappa_root(0.0)
        key = f"scen_{kt:+.0e}"
        try:
            results[key] = run_case(f"scenario kappa_- = {kt:+.0e}", geom, k0, ent, Ns, v0=50.0, v_end=vend, detail=True)
            results[key]["conv"] = convergence_table(results[key], ent, key)
        except Exception as exc:  # noqa: BLE001
            say(f"  !! case {key} failed: {exc!r}")
    say("\n=== wide u-range check (u in [v0, 0.9 v_e], N = 100): contribution of the late, light rays ===")
    for key, kt, ve, vend in (("scen_-1e-03", -1e-3, 1e4, 1e6), ("scen_+0e+00", 0.0, 1e5, 1e9)):
        e0 = t34.eps0_for_kappa(kt)
        geom = t34.ScenGeom(e0, "tmp")
        try:
            results[key + "_wide"] = run_case(f"scenario kappa_- = {kt:+.0e}, wide u", geom, geom.kappa_root(0.0), [ve], [100], v0=50.0, v_end=vend, umax_fac=0.9 * ve / 50.0)
        except Exception as exc:  # noqa: BLE001
            say(f"  !! wide case {key} failed: {exc!r}")
    json.dump(results, open(OUT / "T35A_multishell.json", "w", encoding="utf-8"), indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    (LOGS / "T35A_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"-> {OUT} ({time.time() - T0:.0f} s)")


if __name__ == "__main__":
    main()
