"""T34 helper: symbolic tidal tensor for ds^2 = -f(v,r) dv^2 + 2 dv dr + r^2 dOmega^2 along a radial timelike worldline.
Convention: E_ab = R_acbd u^c u^d, geodesic deviation xi''^a = -E^a_b xi^b (E_rad < 0 = stretching).
Run: .venv/Scripts/python.exe experiments/cycle2/T34_derive.py
"""
import sympy as sp

v, r, th, ph = sp.symbols("v r theta phi")
f = sp.Function("f")(v, r)
x = [v, r, th, ph]
g = sp.Matrix([[-f, 1, 0, 0], [1, 0, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2 * sp.sin(th) ** 2]])
gi = g.inv()
n = 4
Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], x[c]) + sp.diff(g[d, c], x[b]) - sp.diff(g[b, c], x[d])) for d in range(n)) / 2
         for c in range(n)] for b in range(n)] for a in range(n)]
def Riem(a, b, c, d):  # R^a_{bcd}
    e = sp.diff(Gam[a][b][d], x[c]) - sp.diff(Gam[a][b][c], x[d])
    e += sum(Gam[a][c][k] * Gam[k][b][d] - Gam[a][d][k] * Gam[k][b][c] for k in range(n))
    return sp.simplify(e)
R = {}
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                R[(a, b, c, d)] = sum(g[a, k] * Riem(k, b, c, d) for k in range(n))  # R_abcd
vd, rd = sp.symbols("vdot rdot")
u = [vd, rd, 0, 0]
# radial unit spacelike vector orthogonal to u: e = (a, b): g(e,u)=0, g(e,e)=1
a_, b_ = sp.symbols("a b")
e = [a_, b_, 0, 0]
sol = sp.solve([sum(g[i, j] * e[i] * u[j] for i in range(n) for j in range(n)),
                sum(g[i, j] * e[i] * e[j] for i in range(n) for j in range(n)) - 1], [a_, b_], dict=True)
norm = -f * vd**2 + 2 * vd * rd + 1  # = 0 on shell
E_perp = sp.simplify(sum(R[(2, i, 2, j)] * u[i] * u[j] for i in range(n) for j in range(n)) / r**2)
E_rad_list = []
for s in sol:
    ee = [s[a_], s[b_], 0, 0]
    E_rad_list.append(sp.simplify(sum(R[(i, k, j, l)] * ee[i] * u[k] * ee[j] * u[l] for i in range(n) for k in range(n) for j in range(n) for l in range(n))))
print("E_perp =", E_perp)
print("E_rad  =", E_rad_list[0])
# Ricci u u
Ric = sp.Matrix(n, n, lambda b, d: sum(Riem(a, b, a, d) for a in range(n)))
Ruu = sp.simplify(sum(Ric[i, j] * u[i] * u[j] for i in range(n) for j in range(n)))
print("R_uu   =", Ruu)
print("trace check E_rad + 2 E_perp - R_uu =", sp.simplify(E_rad_list[0] + 2 * E_perp - Ruu))
# Schwarzschild check: f = 1 - 2M/r, vdot=1/(E+sqrt(E^2-f)), rdot = f vdot - E
M, En = sp.symbols("M E", positive=True)
fs = 1 - 2 * M / r
vds = 1 / (En + sp.sqrt(En**2 - fs)); rds = fs * vds - En
sub = {f: fs}
Ep = E_perp.subs(f, fs).doit().subs({vd: vds, rd: rds})
Er = E_rad_list[0].subs(f, fs).doit().subs({vd: vds, rd: rds})
print("Schwarzschild E_perp =", sp.simplify(Ep), " E_rad =", sp.simplify(Er))
# Kretschmann
K = 0
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                K += R[(a, b, c, d)] * sum(gi[a, p] * gi[b, q] * gi[c, s] * gi[d, t] * R[(p, q, s, t)] for p in range(n) for q in range(n) for s in range(n) for t in range(n))
print("Kretschmann =", sp.simplify(K))
