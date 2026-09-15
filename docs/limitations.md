# Limitations

## Scope of the claim

Two statements are asserted. First, the deformation accumulated by a freely falling test body that crosses the inner horizon of an inner-extremal regular black hole is finite and, for late entrants, independent of the entry time: Δ⊥ → 1.01, ξ⊥ → 3·10⁻³, Δ_rad → 0.26 for every surface gravity examined, including the exact triple root, and 1.04 / 3.4·10⁻³ / 0.16 for the Reissner–Nordström control. This reproduces Ori's weak-singularity result on a regular background. Second, the transverse impulse delivered at the outgoing shell is not bounded: it grows exponentially in the entry time when κ₋ ≠ 0 and as v_e^{3/2} at the exact triple root, which converts the qualitative statement "the tides are weak" into a finite window of entry times inside which a body of a given size retains molecular or nuclear structure.

Nothing is asserted about what happens after the classical description ends, about back-reaction of the infalling body, or about whether such a crossing is realised in any astrophysical setting.

## Convergence in the number of shells

The main computation replaces the continuous outgoing flux by a single null shell. The check that this is not an artefact discretises a u⁻¹² outflux into N = 3…1000 shells and compares the deformation at the classical end.

For the detuned profiles and for Reissner–Nordström the check succeeds: the deformation changes by at most 2·10⁻⁴ at κ₋ = −10⁻³ and 1.4·10⁻⁴ for RN, and the Richardson-extrapolated limits are stable to six digits (Δ⊥ = 0.657044, 1.008091, 1.012490 at v_e = 10³, 10⁴, 10⁵).

For the exact triple root the check does not succeed. The variation is 2.5·10⁻³ at v_e = 10⁵ and 7·10⁻³ at v_e = 10⁸, and it is not monotone in N: N = 10 overshoots the single shell, N = 100 relaxes to +4·10⁻⁴, and a wide integration window with 100 shells gives +1.0·10⁻³. The runs that would have settled the question did not finish. The N = 100 table ends at v = 1.3·10⁶ because an outer shell's integration was truncated by the NaN guard, so v_e = 10⁸ is covered only up to N = 10; the N = 1000 run of that case was stopped after twenty minutes without output. The available evidence therefore rests on N ≤ 100 at v_e = 10⁵ and N ≤ 10 at v_e = 10⁸.

A re-run for this release confirmed the obstruction directly: the N = 1000 chain for the
detuned case did not finish in five hours of single-core time on a 2024 desktop, growing
to 5.8 GB of resident memory before being stopped. The Reissner-Nordstrom sweep at the
same N completed in 1359 s, so the cost is specific to the near-degenerate profiles, where
the shells hover for long stretches of advanced time.

The most likely explanation is an artefact of the shell bookkeeping rather than a physical effect. At κ₋ = 0 the earlier shells hover at f ~ 10⁻⁴…10⁻³ while later shells launch, and each launch multiplies their jump by exp(Q_k D/f_out) = 1 + O(10⁻²). That is a few-percent perturbation of the chain, and it shrinks as 1/N, which is consistent with the observed non-monotone approach. It is not established that it shrinks to zero.

## Model assumptions

The interior is the two-flux model of Ori in the formulation of Carballo-Rubio et al.: an ingoing Price tail m₋(v) = M − β/v^p with p = 11, β/v₀^p = 10⁻³ M, v₀ = 50 M, and one outgoing null shell of energy δm₊ = 10⁻⁴ M. The background is the base member of the inner-extremal family (ℓ = 0.271 M, R₋ = 0.677 M, R₊ = 1.985 M), detuned to κ₋ = −10⁻³/M and −10⁻⁸/M, taken at the exact triple root, and compared against the untuned case |κ₋| ~ 1/ℓ. The residual tuning error of the base profile was removed with a local window so that κ₋ = 0 is an exact triple root (a = f‴/6 = −2.169).

Behind the shell the mass function follows the linearised family law. The exact ℓ-fixed family would move the Planck-threshold cells earlier by at most a few e-folds and would leave both the impulse and every pre-inflation number unchanged.

The observer is radial with E = 1. The deviation equation is linear, so ξ → 0 means "compressed beyond the linear regime" rather than a computed final size. "Crossing the seam" is identified with entering the inflated region behind the shell; the entry time enters only through 1/ρ and the Misner–Sharp jump [M] at the moment the observer meets the shell.

## Parameter sensitivity

The starting impulse is linear in the outgoing-shell energy over 10⁻⁵…10⁻³ and changes by 4 % when the ingoing amplitude runs over 10⁻⁴…10⁻². Changing v₀ over 20, 50, 100 M changes it by less than 0.1 %. These are the only model parameters the window table is sensitive to at leading order.

## Thresholds

The three thresholds are physical criteria, not computed material properties. Molecular: 10⁹ Pa at 10³ kg m⁻³, giving a breaking velocity of 1.4 km s⁻¹ across one metre. Nuclear: 8 MeV per nucleon, giving 0.14 c across one femtometre; a strain criterion has no meaning at that scale, which is why the impulse rather than Δ is used. Planck: the Kretschmann scalar on the outer side of the shell reaching 1/l_P⁴, which marks the end of the classical description rather than a material failure. A different choice of material constants moves the entries in Table 1 by factors of order unity, not by orders of magnitude, because the underlying growth in the impulse is exponential or power-law in the entry time.

The strain criterion Δ ≥ 10⁻² is exceeded by every crossing through the background field alone, with |Δ| = 0.7…7 for a free body. The quasi-static core stress reproduces the ordinary spaghettification hierarchy and is independent of the seam. Neither is a property of the Cauchy horizon.

## Extrapolated cells

Table 1 contains cells that are extrapolations rather than computed points, marked with an asterisk there and flagged per cell in `data/crossing/T34_windows.json`: the Planck column of the small-|κ₋| rows uses the fitted exponential law K₊ ∝ e^{2|κ₋|v_e}, and the nuclear column of the triple-root rows uses the v^{3/2} law. The untuned row is the Reissner–Nordström control with times rescaled by 1.37/3.7 rather than an independent computation of a regular untuned profile.

## Numerical limitations

All integrations use SciPy's adaptive solvers at their default tolerances with an explicit NaN guard; where that guard fires, the run is truncated and the truncation is recorded in the log rather than extrapolated over. Three of the multishell runs crashed after printing results but before writing JSON, so the wide-window and N = 1000 numbers are quoted from raw stdout logs kept in `logs/crossing/`. No random number generator is used anywhere in this repository, so the only run-to-run variation comes from library versions, which are pinned in `requirements.txt`.

## What would change the conclusion

A multishell computation at the exact triple root that reaches N ≥ 10³ at v_e = 10⁸ without truncation would either confirm the 1/N scaling of the launch artefact, which would close the convergence question, or show a residual that does not shrink, which would mean the single-shell idealisation fails precisely in the degenerate case and the triple-root row of Table 1 would have to be withdrawn. A treatment that includes back-reaction of the infalling body, or a non-radial observer, would change the impulse but is not expected to change the finiteness of Δ, since that follows from the ratio of the divergent blueshift to the shrinking proper time behind the shell.
