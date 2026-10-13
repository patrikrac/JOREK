"""GVEC equilibrium for the reversed-shear double-tearing intear case -> JOREK ffprime_file.

usage: python gvec_ffprime.py OUTDIR T_0 [F_edge] [PREV_PROFILES PHI_EDGE]

Reduced MHD (model199) uses B_phi = F0/R, so its q is q_GVEC*F0/F(psi). With PREV_PROFILES
(gvec_profiles.dat of an earlier run) GVEC is given q_target*F_prev(psi)/F0 instead, so that
the JOREK q matches q_target.

Target q(psi_N) = natural cubic spline through the anchors below (psi_N = normalised
poloidal flux). GVEC prescribes iota(s) with s = normalised toroidal flux, so q(psi_N)
is mapped via s(psi_N) = int_0^psi_N q / int_0^1 q (exact: dPhi/dpsi = q).
Pressure = JOREK p = rho*T (mu0 = 1 units), rho = 1 flat, T from the intear T_coef.
PhiEdge is iterated until F = R*B_phi at the boundary equals F_edge (JOREK F0).
Writes OUTDIR/ffprime_dtm.dat (psi_N, -F dF/dpsi, psi in Wb/rad) and a summary.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, LSQUnivariateSpline, make_interp_spline
from scipy.integrate import cumulative_trapezoid
import gvec

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
T0 = float(sys.argv[2]); F_edge = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
R0, a = 3.0, 1.0
mu0 = 4e-7 * np.pi

# --- target q(psi_N) and pressure p(psi_N) (JOREK units) -----------------------------
Q_ANCH = ([0.0, 0.1, 0.175, 0.25, 0.5, 0.75, 1.0], [2.45, 2.0, 1.9, 2.0, 2.7, 3.4, 4.5])
q_target = CubicSpline(*Q_ANCH, bc_type="natural")
if len(sys.argv) > 4:
    prev = np.loadtxt(sys.argv[4])
    q_of_psi = lambda x: q_target(x) * np.interp(x, prev[:, 1], prev[:, 3]) / F_edge
else:
    q_of_psi = q_target
T1, Tc1, Tc4, Tc5 = 1e-2 * T0, -0.8, 1.0, 5.0     # intear: T_1 = T_0/100, T_coef(1,4,5)
def p_of_psi(x):
    return (T0 - T1) * (1 + Tc1 * x) * (0.5 - 0.5 * np.tanh((x - Tc5) / Tc4)) + T1

psi = np.linspace(0, 1, 2001)
Phi_n = cumulative_trapezoid(q_of_psi(psi), psi, initial=0); Phi_n /= Phi_n[-1]
s_grid = np.linspace(0, 1, 81)
psi_of_s = np.interp(s_grid, Phi_n, psi)
iota_spl = make_interp_spline(s_grid, 1 / q_of_psi(psi_of_s), k=3)
pres_spl = make_interp_spline(s_grid, p_of_psi(psi_of_s) / mu0, k=3)

def run(phi_edge, tag):
    params = {
        "ProjectName": "dtm", "which_hmap": 1, "nfp": 1, "PhiEdge": phi_edge,
        "sgrid_nElems": 24, "X1X2_deg": 5, "LA_deg": 5,
        "X1_mn_max": [8, 0], "X2_mn_max": [8, 0], "LA_mn_max": [8, 0],
        "X1_b_cos": {(0, 0): R0, (1, 0): a}, "X2_b_sin": {(1, 0): a},
        "X1_a_cos": {(0, 0): R0},
        "iota": {"type": "bspline", "knots": iota_spl.t, "coefs": iota_spl.c},
        "pres": {"type": "bspline", "knots": pres_spl.t, "coefs": pres_spl.c},
        "minimize_tol": 1e-9, "totalIter": 200000,
    }
    r = gvec.run(params, runpath=out / tag)
    return r.state

def analyse(state):
    rho = np.linspace(0, 1, 401); th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    ev = state.evaluate("pos", "B", "X1", "X2", "dX1_dr", "dX1_dt", "dX2_dr", "dX2_dt",
                        "iota", "p", "Phi", rho=rho, theta=th, zeta=[0.0])
    pos = np.asarray(ev.pos.transpose("rad", "pol", "tor", "xyz"))[..., 0, :]
    B = np.asarray(ev.B.transpose("rad", "pol", "tor", "xyz"))[..., 0, :]
    R = np.hypot(pos[..., 0], pos[..., 1])
    Bphi = (-pos[..., 1] * B[..., 0] + pos[..., 0] * B[..., 1]) / R
    Fs = R * Bphi
    sgn = np.sign(Fs.mean()); Fs, Bphi = sgn * Fs, sgn * Bphi
    F = Fs.mean(1); Fvar = (Fs.std(1) / np.abs(F)).max()
    g = lambda k: np.asarray(ev[k].transpose("rad", "pol", "tor"))[..., 0]
    jac2 = np.abs(g("dX1_dr") * g("dX2_dt") - g("dX1_dt") * g("dX2_dr"))
    dPhi = (Bphi * jac2).mean(1) * 2 * np.pi            # dPhi_tot/drho
    Phi_tot = cumulative_trapezoid(dPhi, rho, initial=0)
    iota = np.asarray(ev.iota).reshape(-1)[: len(rho)] if ev.iota.size == len(rho) else g("iota")[:, 0]
    psi_pol = cumulative_trapezoid(iota * dPhi, rho, initial=0) / (2 * np.pi)   # Wb/rad
    psiN = psi_pol / psi_pol[-1]
    # FF' = 0.5 dF^2/dpsi. F^2 is smooth in psi (not in rho, where dpsi/drho -> 0 at the
    # axis amplifies GVEC's derivative noise), so fit a least-squares quintic spline to
    # F^2(psi_N) and differentiate that.
    knots = np.linspace(0, 1, 33)[1:-1]
    F2spl = LSQUnivariateSpline(psiN, F**2, knots, k=5)
    ffp = -0.5 * F2spl.derivative()(psiN) / psi_pol[-1]
    phiE = float(np.asarray(ev.Phi).reshape(-1)[-1]) if "Phi" in ev else np.nan
    return dict(rho=rho, psiN=psiN, F=F, ffp=ffp, psi_tot=psi_pol[-1], Phi_tot=Phi_tot[-1],
                Fvar=Fvar, q=1 / iota, R_axis=R[0].mean(), PhiE_gvec=phiE)

phi_edge = float(sys.argv[5]) if len(sys.argv) > 5 else 3.25
for it in range(4):
    if os.environ.get("REANALYSE"):     # re-use the last converged state in OUTDIR
        st = gvec.find_state(sorted(out.glob("it*"))[-1]); res = analyse(st); break
    st = run(phi_edge, f"it{it}"); res = analyse(st)
    Fb = res["F"][-1]
    print(f"it {it}: PhiEdge={phi_edge:.6f} F_edge={Fb:.6f} F_axis={res['F'][0]:.6f} "
          f"Phi_tot/2pi={res['Phi_tot']/2/np.pi:.6f} (gvec {res['PhiE_gvec']:.6f}) "
          f"psi_tot={res['psi_tot']:.6f} R_axis={res['R_axis']:.5f} F-var={res['Fvar']:.1e}", flush=True)
    if abs(Fb / F_edge - 1) < 1e-6: break
    phi_edge *= F_edge / Fb

# ffprime file on uniform psi_N, constant continuation to psi_N = 2
xg = np.linspace(0, 1, 201)
yg = np.interp(xg, res["psiN"], res["ffp"])
xe = np.array([1.25, 1.5, 2.0])
np.savetxt(out / "ffprime_dtm.dat", np.column_stack([np.r_[xg, xe], np.r_[yg, np.full(3, yg[-1])]]), fmt="%.10e")
np.savetxt(out / "gvec_profiles.dat",
           np.column_stack([res["rho"], res["psiN"], res["q"], res["F"], res["ffp"]]),
           header="rho psi_N q F FFprime_jorek", fmt="%.8e")
print("FF' psi_N=0,0.1,0.17,0.25,0.5,0.75,1:", np.round(np.interp([0, .1, .17, .25, .5, .75, 1], xg, yg), 5))
print("psi_bnd-psi_axis (JOREK units):", res["psi_tot"])
