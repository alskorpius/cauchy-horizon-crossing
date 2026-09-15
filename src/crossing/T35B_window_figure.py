"""T35 part B: one figure of the passable window of the seam (inner horizon) from T34.

Every number is taken from experiments/cycle2/results/T34_windows.json (computed threshold times,
seconds after seam formation) and, for the cells that T34 could not compute directly, from
T34_REPORT.md section 5: (i) the untuned row kappa_- = -1/ell is the RN control (|kappa_-| = 1.37/M)
rescaled by 1.37/3.7 in time; (ii) the Planck cells marked "*" in T34 are extrapolated with the fitted
exponential law K_+ ~ e^{2|kappa_-| v_e}; (iii) the nuclear cells of the exact triple root come from the
v^{3/2} impulse law. No physics is recomputed here.

Run: .venv/Scripts/python.exe experiments/cycle2/T35B_window_figure.py
Output: results/T35B_window.png, results/T35B_window.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "data" / HERE.name
FIG = HERE.parents[1] / "figures"
WIN = json.loads((RES / "T34_windows.json").read_text(encoding="utf-8"))

YR = 3.15576e7
DAY = 86400.0
HOUR = 3600.0

MASSES = ["10 Msun", "1e6 Msun", "1e9 Msun"]
MASS_LABEL = {"10 Msun": r"$M = 10\,M_\odot$", "1e6 Msun": r"$M = 10^{6}\,M_\odot$", "1e9 Msun": r"$M = 10^{9}\,M_\odot$"}
ROWS = [  # (label, json key or None for the rescaled RN proxy)
    (r"$\kappa_- = -1/\ell$ (untuned)", "RN"),
    (r"$\kappa_- = -10^{-3}/M$", "scen_-1e-03"),
    (r"$\kappa_- = -10^{-8}/M$", "scen_-1e-08"),
    (r"$\kappa_- = 0$ (triple root)", "scen_+0e+00"),
]
RN_RESCALE = 1.37 / 3.7  # T34_REPORT.md section 5: RN control |kappa_-| = 1.37/M rescaled to 3.7/M

# Cells that T34 gives only by extrapolation (T34_REPORT.md, section 5, marked "*"), in seconds.
EXTRAP_PLANCK = {
    ("scen_-1e-03", "1e6 Msun"): 18 * DAY,
    ("scen_-1e-03", "1e9 Msun"): 51 * YR,
    ("scen_-1e-08", "10 Msun"): 23 * DAY,
    ("scen_-1e-08", "1e6 Msun"): 6.6e3 * YR,
    ("scen_-1e-08", "1e9 Msun"): 6.8e6 * YR,
}
# Exact triple root: nuclear threshold from the v^{3/2} impulse law (T34_REPORT.md section 5/6).
EXTRAP_NUCLEAR_TRIPLE = {"10 Msun": 900 * YR, "1e6 Msun": 2e11 * YR, "1e9 Msun": 2e16 * YR}

THRESH = ["molecular, 1 m body", "nuclear, 1 fm", "Planck"]


def cell(key, mass, thr):
    """Return (status, time_s, extrapolated) for one cell."""
    st, t = WIN[key][f"{mass} | {thr}"]
    extrap = False
    if key == "RN" and t is not None:
        t = t * RN_RESCALE
    if st == "never":
        if thr == "Planck" and (key, mass) in EXTRAP_PLANCK:
            return "window", EXTRAP_PLANCK[(key, mass)], True
        if thr == "nuclear, 1 fm" and key == "scen_+0e+00":
            return "window", EXTRAP_NUCLEAR_TRIPLE[mass], True
        return "never", None, False
    return st, t, extrap


# Sequential single-hue ramp (light -> dark) for the structure levels that survive; grayscale-safe.
C_MOL = "#c6dbef"   # molecules intact
C_NUC = "#6baed6"   # nuclei intact, molecules broken
C_CLS = "#2171b5"   # classical crossing, nuclei broken
C_TP = "#e6e6e6"    # trans-Planckian (no classical description)
INK = "#222222"

T_MIN, T_MAX = 1e-4, 1e20

fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.4), sharex=True)
fig.subplots_adjust(left=0.26, right=0.97, top=0.84, bottom=0.08, hspace=0.34)

for ax, mass in zip(axes, MASSES):
    ax.set_title(MASS_LABEL[mass], loc="left", fontsize=11, color=INK, pad=6)
    for i, (label, key) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        s_mol, t_mol, e_mol = cell(key, mass, THRESH[0])
        s_nuc, t_nuc, e_nuc = cell(key, mass, THRESH[1])
        s_pl, t_pl, e_pl = cell(key, mass, THRESH[2])
        # segment boundaries
        x0 = T_MIN
        edges = []
        if s_mol == "background":
            t_mol_eff = T_MIN
        else:
            t_mol_eff = t_mol
        beyond = s_nuc == "window" and t_nuc > T_MAX / 30   # drawn as text, not as a marker (would sit on the arrow)
        t_nuc_eff = t_nuc if (s_nuc == "window" and not beyond) else T_MAX
        t_pl_eff = t_pl if s_pl == "window" else T_MAX
        h = 0.55
        ax.barh(y, t_mol_eff - x0, left=x0, height=h, color=C_MOL, edgecolor="white", linewidth=1.5)
        ax.barh(y, t_nuc_eff - t_mol_eff, left=t_mol_eff, height=h, color=C_NUC, edgecolor="white", linewidth=1.5)
        ax.barh(y, t_pl_eff - t_nuc_eff, left=t_nuc_eff, height=h, color=C_CLS, edgecolor="white", linewidth=1.5)
        if s_pl == "window":
            ax.barh(y, T_MAX - t_pl_eff, left=t_pl_eff, height=h, color=C_TP, edgecolor="white", linewidth=1.5, hatch="///")
        # markers at thresholds
        for t, ex, s in ((t_mol, e_mol, s_mol), (t_nuc, e_nuc, s_nuc), (t_pl, e_pl, s_pl)):
            if s != "window" or (t is t_nuc and beyond):
                continue
            ax.plot(t, y, marker="o", ms=7, mfc="white" if ex else INK, mec=INK, mew=1.4, ls="none", zorder=5)
        # annotations for special cells
        if s_mol == "background":
            xa = (t_pl if s_pl == "window" else T_MIN * 10) * 4
            ax.text(xa, y, "no molecular window: the 1 m threshold is already exceeded\nby the background tidal field between the horizons",
                    va="center", ha="left", fontsize=7.0, color=INK, zorder=6)
        if s_pl == "never":
            ax.annotate("", xy=(T_MAX, y), xytext=(T_MAX / 30, y),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), zorder=6)
            note = "never trans-Planckian"
            if s_nuc == "window" and t_nuc > T_MAX / 30:
                sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
                m, e = f"{t_nuc/YR:.0e}".split("e+")
                note += f"; nuclei intact for {m}·10{e.lstrip('0').translate(sup)} yr (beyond axis)"
            ax.text(T_MAX / 35, y + 0.36, note, ha="right", va="bottom", fontsize=7.2, color=INK)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=9)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(T_MIN, T_MAX)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

axes[-1].set_xlabel("external entry time after formation of the seam  [s]", fontsize=10, color=INK)
# secondary time labels (top of first panel)
top = axes[0].secondary_xaxis("top")
ticks = [(HOUR, "1 h"), (DAY, "1 d"), (YR, "1 yr"), (1e3 * YR, "10³ yr"), (1e6 * YR, "10⁶ yr"),
         (1.4e10 * YR, "Hubble time")]
top.set_xticks([t for t, _ in ticks])
top.set_xticklabels([l for _, l in ticks], fontsize=8, color=INK)
top.tick_params(length=3, color="#999999")
top.spines["top"].set_color("#999999")

legend = [
    Patch(facecolor=C_MOL, label="molecules intact (1 m body)"),
    Patch(facecolor=C_NUC, label="nuclei intact, molecules broken"),
    Patch(facecolor=C_CLS, label="classical crossing, nuclei broken"),
    Patch(facecolor=C_TP, hatch="///", edgecolor="white", label="seam trans-Planckian (no classical description)"),
    Line2D([], [], marker="o", ms=7, mfc=INK, mec=INK, ls="none", label="threshold computed (T34)"),
    Line2D([], [], marker="o", ms=7, mfc="white", mec=INK, ls="none", label="threshold extrapolated (T34, *)"),
]
fig.legend(handles=legend, loc="upper left", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.26, 0.975))
fig.text(0.26, 0.985, "Structure that a body entering at a given time keeps when it crosses the seam",
         fontsize=10.5, color=INK, ha="left", va="bottom")

RES.mkdir(exist_ok=True)
fig.savefig(FIG / "T35B_window.png", dpi=220)
fig.savefig(FIG / "T35B_window.pdf")
print("written:", RES / "T35B_window.png", RES / "T35B_window.pdf")

# Print the compact table used in the note (seconds -> human units), for cross-checking with T34_REPORT.md.
def human(t):
    if t is None:
        return "-"
    if t < 1:
        return f"{t*1e3:.3g} ms"
    if t < 60:
        return f"{t:.3g} s"
    if t < HOUR:
        return f"{t/60:.3g} min"
    if t < DAY:
        return f"{t/HOUR:.3g} h"
    if t < YR:
        return f"{t/DAY:.3g} d"
    return f"{t/YR:.3g} yr"

for label, key in ROWS:
    for mass in MASSES:
        cells = [cell(key, mass, thr) for thr in THRESH]
        print(f"{label:28s} {mass:8s} | " + " | ".join(
            (s if s != "window" else human(t) + ("*" if ex else "")) for s, t, ex in cells))
