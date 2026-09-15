"""T34 post-processing: passable windows from results/T34_delta.json (produced by T34_tidal.py).

Impulse criterion (the seam acts impulsively: proper time behind the shell ~ rho R/E is tiny):
  a body of size L receives a relative transverse velocity dv = I_sh * L; it is disrupted when (1/2) rho_body dv^2 exceeds the binding
  energy density sigma_b, i.e. dv >= v_b = sqrt(2 sigma_b / rho_body):
    molecular bonds:  sigma_b = 1e9 Pa (reviewer's number), rho_body = 1e3 kg/m^3  -> v_b = 1.4e3 m/s  (bodies of 1 m and 10 km)
    nuclei:           sigma_b ~ 8 MeV per nucleon * n_nuc = 2e32 Pa, rho_nuc = 2.3e17 kg/m^3 -> v_b = 0.14 c   (L = 1 fm)
    Planck:           K_+ on the R+ side of the shell >= 1/l_P^4  (classical description ends at the seam)
Strain criterion (reviewer's phrasing, quasi-static): Delta = int int E dtau^2 >= sigma_b / Y = 1e9 Pa / 1e11 Pa = 1e-2 (solid, Y ~ 1e11 Pa).
Times: v in units of G M/c^3; the seam forms at v0 (model start); external entry time after formation t = (v_e - v0) G M / c^3.
Run: .venv/Scripts/python.exe experiments/cycle2/T34_windows.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / HERE.name
LOGS = HERE.parents[1] / "logs" / HERE.name
LOGS.mkdir(parents=True, exist_ok=True)
R = json.load(open(OUT / "T34_delta.json", encoding="utf-8"))
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


MSUN_S = 4.925e-6
MSUN_M = 1476.6
LPL_M = 1.616e-35
C = 2.998e8
MASSES = {"10 Msun": 10.0, "1e6 Msun": 1e6, "1e9 Msun": 1e9}
V_B = {"molecular, 1 m body": (math.sqrt(2 * 1e9 / 1e3), 1.0), "molecular, 10 km body": (math.sqrt(2 * 1e9 / 1e3), 1e4),
       "nuclear, 1 fm": (0.14 * C, 1e-15)}


def fmt_time(t):
    if t < 1e-3:
        return f"{t:.2e} s"
    if t < 60:
        return f"{t:.3g} s"
    if t < 3600:
        return f"{t/60:.3g} min"
    if t < 86400 * 3:
        return f"{t/3600:.3g} h"
    if t < 3.156e7 * 2:
        return f"{t/86400:.3g} d"
    return f"{t/3.156e7:.3g} yr"


def kappa_label(name):
    return name


def window(rows, v0, key, thr):
    """first v_e at which rows[key] >= thr (log-linear interpolation in v_e); None if never within the table."""
    vs = np.array([r["v_e"] for r in rows if r.get("crossed")])
    ys = np.array([r[key] for r in rows if r.get("crossed")])
    if len(vs) == 0 or ys.max() < thr:
        return None
    if ys[0] >= thr:
        return vs[0]
    i = int(np.nonzero(ys >= thr)[0][0])
    # interpolate ln y linearly in v between i-1 and i
    v1, v2, y1, y2 = vs[i - 1], vs[i], ys[i - 1], ys[i]
    if y1 <= 0:
        return v2
    w = (math.log(thr) - math.log(y1)) / (math.log(y2) - math.log(y1))
    return v1 + w * (v2 - v1)


say("T34 windows (from T34_delta.json)")
summary = {}
for case in ("RN", "scen_-1e-03", "scen_-1e-08", "scen_+0e+00"):
    if case not in R:
        continue
    c = R[case]
    rows = c["rows"]
    v0 = c["v0"]
    say(f"\n=== {c['name']} (kappa_- = {c['kappa0']:+.3e}, v0 = {v0}) ===")
    say("  v_e      | 1/rho      | I_sh [1/M]  | [M] jump   | K_+(shell) | R+ end                     | tau_+ [M]  | J_perp tot | Delta_perp | Delta_rad | xi_perp   | xi_rad   | Planck: 10 / 1e6 / 1e9 Msun")
    for r in rows:
        if not r.get("crossed"):
            say(f"  {r['v_e']:8.3g} | no crossing: {r['end']}; Delta_perp {r['Dp']:+.3e}, Delta_rad {r['Dr']:+.3e}, xi_perp {r['xp']:.3f}, xi_rad {r['xr']:.3f}")
            continue
        pm = r["per_mass"]
        pms = " / ".join(("seam" if v["where"].startswith("at the seam") else ("R+ r=%.2e" % v["r"] if v["where"] == "inside R+" else "no")) for k, v in pm.items())
        say(f"  {r['v_e']:8.3g} | {r['inv_rho']:.3e} | {r['I_shell']:.3e} | {r['dM_jump']:.3e} | {r['K_plus_at_shell']:.2e} | {r['end'][:26]:26s} | {r['tau_plus']:.3e} | {r['Jp']:+.3e} | {r['Dp']:+.3e} | {r['Dr']:+.3e} | {r['xp']:+.3e} | {r['xr']:+.3e} | {pms}")
    # exponent of 1/rho growth
    vs = np.array([r["v_e"] for r in rows if r.get("crossed")])
    ir = np.array([r["inv_rho"] for r in rows if r.get("crossed")])
    if len(vs) > 3:
        k = ir > 1e3
        if k.sum() >= 2:
            if c["kappa0"] != 0:
                sl = np.polyfit(vs[k], np.log(ir[k]), 1)[0]
                say(f"  growth of 1/rho: d ln(1/rho)/dv_e = {sl:.4e} (|kappa_-| = {abs(c['kappa0']):.3e}) -> doubling time {math.log(2)/sl:.4g} M")
            else:
                sl = np.polyfit(np.log(vs[k]), np.log(ir[k]), 1)[0]
                say(f"  growth of 1/rho: 1/rho ~ v_e^{sl:.3f} (analytic 3/2 for the triple root)")
    # windows per mass
    res_case = {}
    for mname, Ms in MASSES.items():
        Msec = MSUN_S * Ms
        KPl = (Ms * MSUN_M / LPL_M) ** 4
        lines = []
        for bname, (vb, L) in V_B.items():
            I_thr = (vb / L) * Msec               # threshold impulse in units 1/M
            # background (R- passage) impulse magnitude
            Jbg = max(abs(r["Jp_minus"]) for r in rows if r.get("crossed"))
            if Jbg >= I_thr:
                lines.append(f"    {bname:22s}: exceeded already by the background tidal field between the horizons (|J_perp| = {Jbg:.2e} >= {I_thr:.2e} 1/M) -> no window")
                res_case[(mname, bname)] = ("background", None)
                continue
            vthr = window(rows, v0, "I_shell", I_thr)
            if vthr is None:
                lines.append(f"    {bname:22s}: not reached within the table (I_sh max {max(r['I_shell'] for r in rows if r.get('crossed')):.2e} < {I_thr:.2e} 1/M)")
                res_case[(mname, bname)] = ("never", None)
            else:
                t = (vthr - v0) * Msec
                lines.append(f"    {bname:22s}: I_sh >= {I_thr:.2e} 1/M at v_e = {vthr:.4g} M -> entries within {fmt_time(t)} after seam formation keep this structure")
                res_case[(mname, bname)] = ("window", t)
        vP = window(rows, v0, "K_plus_at_shell", KPl)
        if vP is None:
            lines.append(f"    Planck at the seam    : not reached within the table")
            res_case[(mname, "Planck")] = ("never", None)
        else:
            lines.append(f"    Planck at the seam    : K_+(shell) >= {KPl:.2e} at v_e = {vP:.4g} M -> classical description of the crossing ends {fmt_time((vP - v0) * Msec)} after formation")
            res_case[(mname, "Planck")] = ("window", (vP - v0) * Msec)
        say(f"  {mname} (G M/c^3 = {Msec:.3e} s): " )
        for l in lines:
            say(l)
    summary[case] = {f"{k[0]} | {k[1]}": v for k, v in res_case.items()}

# starting cost
if "start" in R:
    say("\n=== starting cost (entrant at v_e = v0, the first crossing) ===")
    for key, d in R["start"].items():
        o, o2 = d["first"], d["double"]
        say(f"  {key}: 1/rho = {o.get('inv_rho', float('nan')):.4e}, I_sh = {o.get('I_shell', float('nan')):.4e} 1/M, [M] = {o.get('dM_jump', float('nan')):.3e}, "
            f"Delta_perp = {o['Dp']:+.4e}, Delta_rad = {o['Dr']:+.4e}, xi_perp = {o['xp']:.4f}, xi_rad = {o['xr']:.4f}, end: {o['end']} (r = {o['r_end']:.2e}); "
            f"at v_e = 2 v0: I_sh = {o2.get('I_shell', float('nan')):.3e}, Delta_perp = {o2['Dp']:+.3e}")

json.dump(summary, open(OUT / "T34_windows.json", "w", encoding="utf-8"), indent=1, default=str)
(LOGS / "T34_windows_log.txt").write_text("\n".join(LOG), encoding="utf-8")

# ---- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for case, lab, col in (("RN", "RN e=0.9 (Ori control), v0=20", "k"), ("scen_-1e-03", "scenario kappa_- = -1e-3", "C0"), ("scen_-1e-08", "scenario kappa_- = -1e-8", "C1"), ("scen_+0e+00", "scenario kappa_- = 0 (triple root)", "C2")):
        if case not in R:
            continue
        rows = [r for r in R[case]["rows"] if r.get("crossed")]
        v0 = R[case]["v0"]
        x = np.array([r["v_e"] - v0 + 1 for r in rows])
        ax[0].loglog(x, [r["I_shell"] for r in rows], "o-", color=col, label=lab, ms=3)
        ax[1].semilogx(x, [r["Dp"] for r in rows], "o-", color=col, label=lab, ms=3)
        ax[1].semilogx(x, [r["Dr"] for r in rows], "s--", color=col, ms=3)
        ax[2].loglog(x, [max(r["xp"], 1e-12) for r in rows], "o-", color=col, label=lab, ms=3)
    for Ms, ls in ((10, ":"), (1e6, "-."), (1e9, "--")):
        Msec = MSUN_S * Ms
        ax[0].axhline((1.4e3 / 1.0) * Msec, color="gray", ls=ls, lw=0.8)
        ax[0].axhline((0.14 * C / 1e-15) * Msec, color="red", ls=ls, lw=0.8)
    ax[0].set_xlabel("v_e - v0 + 1  [M]  (external entry time after seam formation)")
    ax[0].set_ylabel("I_sh = transverse impulse at the shell [1/M]")
    ax[0].set_title("shell impulse; grey = molecular (1 m), red = nuclear (1 fm); "
                    "dotted / dash-dot / dashed = 10, 1e6, 1e9 Msun", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("v_e - v0 + 1 [M]")
    ax[1].set_ylabel("Delta = int int E dtau^2 (circles: transverse, squares: radial)")
    ax[1].set_title("integrated deformation up to the classical end")
    ax[1].legend(fontsize=7)
    ax[2].set_xlabel("v_e - v0 + 1 [M]")
    ax[2].set_ylabel("xi_perp / xi_perp(outer horizon)")
    ax[2].set_title("transverse size of a free body at the classical end")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE.parents[1] / "figures" / "T34_delta_vs_v.png", dpi=110)
    say(f"figure -> {OUT / 'T34_delta_vs_v.png'}")
except Exception as e:  # noqa: BLE001
    say(f"figure not produced: {e}")
