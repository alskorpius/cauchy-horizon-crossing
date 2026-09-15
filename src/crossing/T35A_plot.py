"""T35A figure: convergence of the integrated deformation with the number of outgoing shells N (continuous-outflux limit), and
the impulse sequence for one entrant.  Parses the run logs results/T35A_run_m31000.txt (kappa = -1e-3, N <= 1000) and
T35A_run_rnz.txt (RN, kappa = 0, N <= 100), and the JSON results/T35A_multishell.json for the crossing sequence.
Run: .venv/Scripts/python.exe experiments/cycle2/T35A_plot.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[2] / "data" / Path(__file__).resolve().parent.name
ROW = re.compile(r"^\s*([0-9.e+]+)\s*\|\s*(\d+)\s*\|\s*([+-][0-9.]+)\s*\|\s*([+-][0-9.]+)\s*\|")


def parse(fname, label_re):
    """rows of the 'convergence in N (<label>)' tables: {label: {v_e: [(N, Dp, Dr), ...]}}."""
    tabs, cur = {}, None
    for line in open(OUT / fname, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*convergence in N \((.*)\):", line)
        if m:
            cur = m.group(1); tabs.setdefault(cur, {}); continue
        m = ROW.match(line)
        if m and cur is not None:
            ve = float(m.group(1)); N = int(m.group(2))
            tabs[cur].setdefault(ve, []).append((N, float(m.group(3)), float(m.group(4))))
    return {k: v for k, v in tabs.items() if re.search(label_re, k)}


series = {}
for fname, lab, title in (("T35A_run_m31000.txt", "scen_-1e-03", "scenario, kappa_- = -1e-3/M"), ("T35A_run_rnz.txt", "scen_\\+0e\\+00", "scenario, kappa_- = 0 (triple root)"),
                          ("T35A_run_rnz.txt", "^RN", "RN e = 0.9 (Ori control)")):
    if (OUT / fname).exists():
        t = parse(fname, lab)
        if t:
            series[title] = next(iter(t.values()))

fig, axes = plt.subplots(1, len(series) + 1, figsize=(4.3 * (len(series) + 1), 4),
                         squeeze=False)
axes = axes[0]     # always a 1-D array, even for a single panel
for ax, (title, tab) in zip(axes, series.items()):
    for ve, rows in sorted(tab.items()):
        rows = sorted(set(rows))
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label=f"v_e = {ve:g}")
    ax.set_xscale("log"); ax.set_xlabel("N (outgoing shells)"); ax.set_ylabel("Delta_perp at the classical end"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[-1]
jf = OUT / "T35A_multishell.json"
if jf.exists():
    d = json.load(open(jf, encoding="utf-8"))
    key = "scen_+0e+00"
    if key in d:
        Ns = [int(n) for n in d[key] if n != "conv"]
        N = max(Ns)
        o = next((o for o in d[key][str(N)]["rows"] if o["v_e"] == 1e5), None)
        if o:
            cr = [c for c in o["crossings"] if not c.get("skipped")]
            u = np.array([c["u"] for c in cr]); lnI = np.array([c["ln_I"] for c in cr]); Dp = np.array([c["Dp"] for c in cr])
            ax.plot(u, lnI / np.log(10), "b.-", ms=3, label="log10 impulse I_k")
            ax2 = ax.twinx(); ax2.plot(u, Dp, "r-", label="Delta_perp before shell k"); ax2.set_ylabel("Delta_perp accumulated", color="r")
            ax.set_xlabel("u of the crossed shell (outer -> inner = right -> left)"); ax.set_ylabel("log10 I_k")
            ax.set_title(f"kappa_- = 0, v_e = 1e5, N = {N}: impulses vs accumulated Delta", fontsize=9); ax.invert_xaxis(); ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT.parents[1] / "figures" / "T35A_convergence.png", dpi=130)
print("->", OUT / "T35A_convergence.png", {k: sorted(v) for k, v in series.items()})
