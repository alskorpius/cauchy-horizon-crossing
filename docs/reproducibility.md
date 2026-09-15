# Reproducibility map

Every number in the note maps to one command and one result file. Commands are run from the repository root with the interpreter of the virtual environment created from `requirements.txt`. Units: G = c = 1, M = 1 unless stated otherwise.

## Environment

Python 3.13.6; NumPy 2.3.5; SciPy 1.17.0; SymPy 1.14.0; Matplotlib 3.10.8. All scripts are single-threaded and deterministic — there is no random number generation anywhere in this repository, so repeated runs differ only through integrator tolerances and the NumPy/SciPy versions pinned above.

## Number → script → result file

| Number in the note | Command | Result file | Section |
|---|---|---|---|
| Tidal tensor in the parallel-propagated frame; Schwarzschild check E_rad = −2M/r³, E_⊥ = +M/r³ | `python src/crossing/T34_derive.py` | stdout (symbolic identities; residual 0) | §1 |
| Late-entrant asymptotes Δ⊥ → 1.012, ξ⊥ → 3.0·10⁻³, Δ_rad → 0.263…0.267 (scenario, all κ₋) | `python src/crossing/T34_tidal.py` | `data/crossing/T34_delta.json`, `logs/crossing/T34_log.txt` | §2 |
| Reissner–Nordström control: Δ⊥ → 1.038, ξ⊥ → 3.4·10⁻³ | `python src/crossing/T34_tidal.py` | `data/crossing/T34_delta.json` (RN rows) | §2 |
| RN shell growth d ln Q/dv = 1.1668 / 1.2352 / 1.3096 at v = 60 / 90 / 200, against Ori's \|κ\| − (p+1)/v = 1.1698 / 1.2365 / 1.3098 | `python src/crossing/T34_tidal.py` | `data/crossing/T34_delta.json`, `logs/crossing/T35A_run_rnz.txt` | §2 |
| Early-entrant core passage: r_cross = 1.31 M at v_e = v₀; ξ⊥ = 23, Δ⊥ = −7.2 | `python src/crossing/T34_tidal.py` | `data/crossing/T34_delta.json` | §2 |
| Blueshift laws: 1/ρ ∝ e^{\|κ₋\|v_e} with doubling time ln2/\|κ₋\| (693 M at κ₋ = −10⁻³); 1/ρ ∝ v_e^{3/2} at the exact triple root | `python src/crossing/T34_tidal.py` | `data/crossing/T34_delta.json` | §2 |
| Starting impulse I_sh(v₀) = 4.2·10⁻⁴ M⁻¹ per 10⁻⁴ of outgoing-shell energy (8.6 s⁻¹ for 10 M☉); 4 % variation over ingoing amplitude 10⁻⁴…10⁻²; < 0.1 % over v₀ = 20, 50, 100 M | `python src/crossing/T34_shell_energy.py` | `data/crossing/T34_shell_energy.json`, `logs/crossing/T34_shell_energy_log.txt` | §2 |
| Table 1 in full (molecular 1 m, molecular 10 km, nuclear 1 fm, Planck columns; e.g. 10⁹ M☉ at κ₋ = −10⁻³: 3.1 yr / 610 d / 10.1 yr / 51 yr\*) | `python src/crossing/T34_windows.py` | `data/crossing/T34_windows.json`, `logs/crossing/T34_windows_log.txt` | §3 |
| Threshold definitions: molecular v_b = 1.4 km s⁻¹ across 1 m; nuclear v_b = 0.14 c across 1 fm; Planck K ≥ 1/l_P⁴ | `python src/crossing/T34_windows.py` | `data/crossing/T34_windows.json` | §1, §3 |
| Deformation versus entry time (figure) | `python src/crossing/T34_windows.py` | `figures/T34_delta_vs_v.png` | §2 |
| Convergence in N: max change 2·10⁻⁴ (κ₋ = −10⁻³), 1.4·10⁻⁴ (RN), 2.5·10⁻³ non-monotone (triple root) | `python src/crossing/T35A_multishell.py` | `data/crossing/T35A_multishell.json`, `logs/crossing/T35A_log.txt` | §4 |
| Richardson limits Δ⊥ = 0.657044 (v_e = 10³), 1.008091 (10⁴), 1.012490 (10⁵) at κ₋ = −10⁻³ | `python src/crossing/T35A_multishell.py` | `data/crossing/T35A_multishell.json` | §4 |
| Wide-window and N = 1000 numbers (runs that crashed at the end) | — | `logs/crossing/T35A_run_m31000.txt`, `T35A_run_wide2.txt`, `T35A_run_z1000.txt`, `T35A_run_100.txt` | §4 |
| Convergence figure | `python src/crossing/T35A_plot.py` | `figures/T35A_convergence.png` | §4 |
| Figure 1 of the note (passable window) | `python src/crossing/T35B_window_figure.py` | `figures/T35B_window.pdf`, `figures/T35B_window.png` | §3 |

Cells marked \* in Table 1 are extrapolations, not computed points: the fitted exponential law K₊ ∝ e^{2\|κ₋\|v_e} for the Planck column of the small-\|κ₋\| rows, and the v^{3/2} law for the nuclear column of the triple root. The note's table marks them, and `T34_windows.json` carries the flag per cell.

## Numbers that come from logs rather than JSON

Three `T35A_multishell.py` runs crashed after printing their results but before writing JSON: the N = 1000 case at κ₋ = −10⁻³, the wide-window N = 100 case, and one RN/triple-root run. Their numbers are quoted from the raw stdout logs kept in `logs/crossing/` (`T35A_run_m31000.txt`, `T35A_run_wide2.txt`, `T35A_run_z1000.txt`, `T35A_run_100.txt`). The JSON in `data/crossing/T35A_multishell.json` covers the runs that completed. Re-running `T35A_multishell.py` reproduces the completed runs; the aborted ones are documented in `docs/limitations.md` rather than claimed as results.

## The convergence run does not complete

`T35A_multishell.py` was re-run for this release on a 2024 desktop. It completed the
Reissner-Nordstrom sweep (N = 3, 10, 100, 1000; the N = 1000 chain took 1359 s) and the
detuned case kappa_- = -1e-3 up to N = 100, whose table ends at v = 1.304e6 exactly as
recorded in the original notes. It then entered the N = 1000 chain for the detuned case
and was still running after five hours, having grown to 5.8 GB of resident memory, at
which point it was stopped.

`data/crossing/T35A_multishell.json` is therefore the file written by the original run of the cases that did complete (RN, and the triple root in the narrow and wide windows), not by a re-run for this release; it was produced by the same code that is published here.

This reproduces the documented behaviour rather than contradicting it: the original run
of that same case was abandoned after twenty minutes without output. The numbers quoted
for the wide-window and N = 1000 cases therefore come from the raw stdout logs in
`logs/crossing/`, which are the only record of them, and `data/crossing/` carries the
results of the runs that did finish. The convergence claim in the note's main text is
limited accordingly: established for the detuned profiles and for Reissner-Nordstrom,
not established for the exact triple root.

## Library modules

Three directories under `src/` are libraries, not entry points, and produce no results of their own:

- `src/inhomogeneous_collapse/ltb_bounce.py` supplies the `ScenarioMass` background to `T34_tidal.py`. Running it standalone performs a full Lemaitre-Tolman-Bondi collapse study that belongs to the companion paper, not to this note.
- `src/stability/triple_root_monotone.py` builds the inner-extremal family analytically; `T34_tidal.py` reaches it through `ltb_bounce.py`.
- `src/approach_map/` and `src/baseline/` are imported by `triple_root_monotone.py` for its Einstein-tensor helpers.

Every number in the note comes from the `src/crossing/` scripts listed above.

## Order of execution

```
python src/crossing/T34_derive.py
python src/crossing/T34_tidal.py
python src/crossing/T34_windows.py
python src/crossing/T34_shell_energy.py
python src/crossing/T35A_multishell.py
python src/crossing/T35A_plot.py
python src/crossing/T35B_window_figure.py
```

`T34_windows.py` and `T34_shell_energy.py` require `data/crossing/T34_delta.json`; `T35B_window_figure.py` requires `data/crossing/T34_windows.json`; `T35A_plot.py` reads both `data/crossing/T35A_multishell.json` and the raw logs in `logs/crossing/`.
