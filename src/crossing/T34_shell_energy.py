"""T34 supplement: dependence of the starting impulse on the outgoing-shell energy dm_plus (m_+(v0) = m_-(v0)(1 + dm_plus))
and on the ingoing tail amplitude, scenario kappa_- = -1e-3, entrants v_e = v0 and 2 v0.  Run: .venv/Scripts/python.exe experiments/cycle2/T34_shell_energy.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import T34_tidal as T
geom = T.ScenGeomQuiet(T.eps0_for_kappa(-1e-3))
rows = []
for dmp in (1e-5, 1e-4, 1e-3):
    for amp in (1e-4, 1e-3, 1e-2):
        tab = T.solve_shell(geom, eps0=amp, v0=50.0, v_end=1e6, dm_plus=dmp)
        o1 = T.run_observer(geom, tab, 50.0); o2 = T.run_observer(geom, tab, 100.0); o3 = T.run_observer(geom, tab, 1e4)
        rows.append(dict(dm_plus=dmp, amp=amp, I_v0=o1["I_shell"], M_v0=o1["dM_jump"], I_2v0=o2["I_shell"], I_1e4=o3["I_shell"], inv_rho_1e4=o3["inv_rho"], Dp_v0=o1["Dp"], xp_v0=o1["xp"]))
        print(f"dm_plus = {dmp:.0e}, tail amp = {amp:.0e}: v_e = v0: [M] = {o1['dM_jump']:.3e}, I_sh = {o1['I_shell']:.3e}, Delta_perp = {o1['Dp']:+.3f}, xi_perp = {o1['xp']:.2f}; "
              f"v_e = 2 v0: I_sh = {o2['I_shell']:.3e}; v_e = 1e4: I_sh = {o3['I_shell']:.3e}, 1/rho = {o3['inv_rho']:.3e}", flush=True)
json.dump(rows, open(Path(__file__).resolve().parents[2] / "data" / "crossing" / "T34_shell_energy.json", "w"), indent=1)
