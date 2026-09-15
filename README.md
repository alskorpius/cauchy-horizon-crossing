# cauchy-horizon-crossing

Code and data for the note **"Crossing the inner horizon of a regular black hole: the cost of the Cauchy horizon and the passable window"**.

A body that falls into a regular black hole and crosses the inner (Cauchy) horizon accumulates a *finite, epoch-independent* deformation — Ori's weak singularity, reproduced here for an inner-extremal regular profile — but receives a transverse velocity kick at the outgoing shell that grows without bound with the entry time, so the crossing is survivable only inside a window measured in milliseconds for stellar-mass holes and in years to millennia for supermassive ones.

The note itself is in [`docs/window_note.md`](docs/window_note.md).

## What is and is not established

The convergence of the result in the number of shells *N* — the check that the single outgoing shell is not an artefact of the idealisation — is established **for detuned profiles (κ₋ ≠ 0) and for Reissner–Nordström**: replacing one shell by N = 3…1000 shells discretising a u⁻¹² outflux changes the deformation at the classical end by at most 2·10⁻⁴ (κ₋ = −10⁻³), 1.4·10⁻⁴ (RN).

**For the exact triple root (κ₋ = 0) convergence in N is not formally established.** The observed variation there is 2.5·10⁻³ (v_e = 10⁵) to 7·10⁻³ (v_e = 10⁸) and it is *not monotone* in N: the N = 10 value overshoots the single shell, N = 100 relaxes back to +4·10⁻⁴, and a wide integration window (u up to 0.9 v_e, 100 shells) gives +1.0·10⁻³. The runs that would settle this were aborted — the N = 100 table ends at v = 1.3·10⁶ because an outer shell's integration was truncated by the NaN guard, and the N = 1000 run of that case was stopped after 20 minutes without output. The triple-root evidence therefore rests on N ≤ 100 at v_e = 10⁵ and N ≤ 10 at v_e = 10⁸. The likely cause is bookkeeping rather than physics: at κ₋ = 0 earlier shells hover at f ~ 10⁻⁴…10⁻³ when later ones launch, and each launch multiplies their jump by exp(Q_k D/f_out) = 1 + O(10⁻²), a perturbation of the chain that shrinks as 1/N. This is stated here, in the main text, and not only in `docs/limitations.md`.

## Install and run

```
git clone https://github.com/alskorpius/inner-extremal-rbh   # optional: paper 1, for the family
git clone https://github.com/alskorpius/cauchy-horizon-crossing
cd cauchy-horizon-crossing
python -m venv .venv && .venv/Scripts/activate      # POSIX: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13.6. Every script is run from the repository root, is single-threaded, and uses only relative paths:

```
python src/crossing/T34_derive.py          # symbolic tidal tensor (sympy), standalone
python src/crossing/T34_tidal.py           # main run: shell, observer, deformation   (~25 min)
python src/crossing/T34_windows.py         # thresholds and the window table          (~1 min)
python src/crossing/T34_shell_energy.py    # starting-cost sweep in the shell energy  (~5 min)
python src/crossing/T35A_multishell.py     # convergence in the number of shells      (long, see below)
python src/crossing/T35A_plot.py           # convergence figure
python src/crossing/T35B_window_figure.py  # Figure 1 of the note
```

Order matters: `T34_windows.py` and `T34_shell_energy.py` read `data/crossing/T34_delta.json` written by `T34_tidal.py`; `T35B_window_figure.py` reads `data/crossing/T34_windows.json`; `T35A_multishell.py` imports `T34_tidal` for the scenario and RN geometries.

`T35A_multishell.py` is the expensive one — the N = 1000 case took about 15 minutes of wall clock, and the triple-root case at large N does not finish (see above). Its raw run logs from the original runs are kept in `logs/crossing/`.

## Figure and table map

| Item in the note | Script | Output |
|---|---|---|
| Figure 1 (passable window) | `src/crossing/T35B_window_figure.py` | `figures/T35B_window.pdf`, `figures/T35B_window.png` |
| Table 1 (entry-time thresholds) | `src/crossing/T34_windows.py` | `data/crossing/T34_windows.json`, `logs/crossing/T34_windows_log.txt` |
| §2, universal deformation Δ⊥ → 1.01, ξ⊥ → 3·10⁻³, Δ_rad → 0.26 | `src/crossing/T34_tidal.py` | `data/crossing/T34_delta.json` |
| §2, starting impulse I_sh(v₀) = 4.2·10⁻⁴ M⁻¹ | `src/crossing/T34_shell_energy.py` | `data/crossing/T34_shell_energy.json` |
| §2, RN control d ln Q/dv = 1.1668 / 1.2352 / 1.3096 at v = 60 / 90 / 200 | `src/crossing/T34_tidal.py`, `src/crossing/T35A_multishell.py` | `data/crossing/T34_delta.json`, `logs/crossing/T35A_run_rnz.txt` |
| §4, convergence in N | `src/crossing/T35A_multishell.py` | `data/crossing/T35A_multishell.json`, `figures/T35A_convergence.png` |
| §2, deformation vs entry time | `src/crossing/T34_windows.py` | `figures/T34_delta_vs_v.png` |

## Layout

```
src/crossing/                the T34 / T35 computations
src/inhomogeneous_collapse/  ltb_bounce.py — the scenario mass function
src/stability/               triple_root_monotone.py — the inner-extremal family
data/                        results as JSON
figures/                     figure scripts' output (PDF/PNG)
logs/                        raw stdout logs of the original runs
docs/window_note.md          the note
docs/reproducibility.md      number in the note -> script -> result file
docs/limitations.md          scope and limits
```

The background profile is *not* read from a file: `ltb_bounce.py` regenerates the inner-extremal family analytically through `triple_root_monotone.py`, so this repository is numerically self-contained. It shares the physical profile with paper 1 by construction, not by data dependency.

## Limitations

Test body without back-reaction; classical treatment throughout, with the Planck column marking only where that treatment ends. Single outgoing shell in the main computation, with the continuous-outflux check as described above. The mass law behind the shell is the linearised family law. Model parameters are fixed at ingoing amplitude 10⁻³, outgoing-shell energy 10⁻⁴, p = 11, v₀ = 50 M, and a radial E = 1 observer. Full discussion in [`docs/limitations.md`](docs/limitations.md).

## Citation

Oleh Popenkov, *Crossing the inner horizon of a regular black hole: the cost of the Cauchy horizon and the passable window* (2026).
Code and data: https://github.com/alskorpius/cauchy-horizon-crossing — see [`CITATION.cff`](CITATION.cff).

ORCID: [0009-0008-9894-2982](https://orcid.org/0009-0008-9894-2982)

## License

Code in `src/` and the figure scripts: MIT. Data in `data/`, raw logs in `logs/`, and the generated figures: CC-BY-4.0. See [`LICENSE`](LICENSE).

## How this work was produced

The calculations in this repository were carried out with automated agents under the author's direction, with an independent cross-audit performed between two separate research projects. The author is responsible for the results.
