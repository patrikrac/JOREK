#!/usr/bin/env python3
"""
obtain_diff_coefficients.py

Purpose
-------
Given a fixed equilibrium geometry and prescribed source terms, this script computes
radially varying transport coefficients that keep chosen 1D profiles steady in time
under a diffusion model.

It solves the *inverse* 1D diffusion problem on the normalized radial coordinate psi_N:

  ∂u/∂t = (1/alpha(psi)) * ∂/∂psi [ D(psi) / gamma(psi) * ∂u/∂psi ] + S(psi)

For a target profile u to remain stationary (∂u/∂t = 0), the diffusion coefficient must satisfy the integral
constraint (your Eq. 2):

  X(psi) = - (psi_bnd - psi_axis)^2 / ( alpha(psi) * du/dpsi )
           * ∫_0^psi gamma(psi') * Source(psi') dpsi'

where:
  - psi is psi_N (normalized flux coordinate)
  - X is the diffusivity: D for density, and kappa for temperatures
  - alpha(psi), gamma(psi) are geometry/metric coefficients (input columns)
  - Source(psi) is the imposed source term:
        * S for density (particle source)
        * HEATi, HEATe, HEAT for Ti, Te, T respectively
  - (psi_bnd - psi_axis)^2 is an overall scaling coming from the mapping used in the derivation

All these profiles in the required order can be exported from jorek2_postproc (https://jorek.eu/wiki/doku.php?id=jorek2_postproc)
using the command "pol_contour_integrals". It will conveniently export the file that you need to feed to this program

The script computes:
  - Density diffusivity:      D(psi)
  - Ion heat diffusivity:     kappa_i(psi)
  - Electron heat diffusivity:kappa_e(psi)
  - Total heat diffusivity:   kappa_t(psi)

and writes each profile to a separate ASCII file with two columns: (psi_N, coefficient).

Robustness / practical handling of incompatible inputs
------------------------------------------------------
The analytic formula can become undefined or unphysical in common edge cases.
This script handles them as follows:

1) Unconstrained region (0/0):
   If |du/dpsi| is ~0 and the cumulative source integral ∫ gamma*Source dpsi is ~0,
   then the steady-state equation does not constrain X (any X works).
   -> Those points are set to NaN and then filled by linear interpolation from nearby
      constrained regions.

2) Impossible region (requires infinite flux):
   If |du/dpsi| is ~0 but the cumulative source integral is not ~0, then no finite
   diffusivity can keep the profile steady there.
   -> Those points are set to NaN and (by default) filled by interpolation.
      If you want to abort instead, run with --hard-fail-impossible.

3) Physical constraints and basic sanity:
   - alpha <= 0 -> invalid (set to NaN and fill)
   - X < 0 or non-finite -> treated as invalid (set to NaN and fill)
   - Final values are clipped into user-defined [MIN, MAX] ranges for each profile.

Core fix (psi_N < EXTR_CORE_PSIN)
---------------------------------
The computed diffusion profiles can be poorly defined very close to the magnetic axis.
After computing D/kappa, the script enforces a clean core behavior:

  - It fits a straight line using the first two points with psi_N >= EXTR_CORE_PSIN
  - It overwrites all points with psi_N < EXTR_CORE_PSIN using that line
  - It ensures a psi_N = 0 point exists by prepending it if missing

This produces well-behaved coefficients at the axis, even if the input does not contain
psi_N = 0.

SOL extension (psi_N > 1)
-------------------------
Input profiles typically end around psi_N ≈ 1 (separatrix). This script can extend the
final diffusion profiles out to EXTEND_TO_PSIN with N_SOL_POINTS additional points:

  - By default, it extends with a constant value equal to the last computed point.
  - Optionally, you can ramp the SOL value up smoothly using an arctan-based factor:
        X_SOL(psi) = X_sep * [1 + RAMP_AMP * f(psi)]
    where f(psi)=0 for psi<=1 and tends to 1 in the SOL.

Hardcoded parameters you should review
--------------------------------------
This script is intentionally configured via constants near the top of the file (rather than
many command-line flags), because the appropriate thresholds and extensions are case-dependent.
Before using on a new dataset, review and adapt:

1) Column indices (COL_*)
   These define where psi_N, derivatives, geometry coefficients and sources are read from
   the input file. If your input format changes, update these first.

2) Numerical tolerances / validity thresholds
   - EPS_INT: threshold for "cumulative source integral I(psi) is ~0"
   - EPS_ALP: alpha positivity threshold (alpha <= EPS_ALP is treated invalid)
   - EPS_DER_RHO / EPS_DER_Ti / EPS_DER_Te / EPS_DER_T:
       thresholds for "derivative is ~0". These control what is considered:
         * unconstrained (0/0)  -> interpolated
         * impossible (I!=0)    -> interpolated or hard-failed
     If these are too large, you may incorrectly classify regions as flat/unconstrained.
     If too small, numerical noise can create very large coefficients.

3) Physical bounds (MIN/MAX)
   - D_MIN_RHO, D_MAX_RHO
   - KAPPA_*_MIN, KAPPA_*_MAX
   These clip the final coefficients. They are not part of the analytic solution; they are
   practical safeguards. Make them consistent with your transport model and units.

4) Core extrapolation control
   - EXTR_CORE_PSIN:
     All points with psi_N < EXTR_CORE_PSIN are overwritten by a linear extrapolation
     from the first two points above EXTR_CORE_PSIN, and psi_N=0 is inserted if missing.
     Increase it if the axis region is unreliable/noisy; decrease it if you trust the core data.

5) SOL extension control
   - EXTEND_TO_PSIN, N_SOL_POINTS:
     Extend the final profiles beyond the separatrix up to EXTEND_TO_PSIN.
   - RAMP_AMP, RAMP_CENTER, RAMP_WIDTH:
     Optional arctan ramp in the SOL. With RAMP_AMP=0 the SOL extension is constant.
     With RAMP_AMP>0 the coefficient increases smoothly in the SOL, asymptoting to
       (1 + RAMP_AMP) * value_at_separatrix.

Typical usage
-------------
  python3 obtain_diff_coefficients.py input_profiles.txt --skiprows 1
  python3 obtain_diff_coefficients.py input_profiles.txt --hard-fail-impossible

Outputs
-------
  D_profile.dat     : psiN, D
  kappa_i.dat       : psiN, kappa_i
  kappa_e.dat       : psiN, kappa_e
  kappa_t.dat       : psiN, kappa_t
"""

import numpy as np

# -----------------------------
# User-editable column indices
# -----------------------------
COL_PSI   = 0   # psi_N
COL_DRHO  = 5   # d(rho)/d(psi_N)
COL_ALPHA = 2   # alpha(psi_N)
COL_GAMMA = 1   # gamma(psi_N)
COL_S     = 6   # <S>(psi_N)
COL_PSI_AXIS = 16  # psi_axis (constant column)
COL_PSI_BND  = 17  # psi_bnd  (constant column)

# Temperature derivatives and heating sources
COL_DTi      = 8
COL_DTe      = 10
COL_DT       = 12
COL_HEATi    = 13
COL_HEATe    = 14
COL_HEAT     = 15

# -----------------------------
# Numerics / checks parameters
# -----------------------------
EPS_INT = 1e-12   # threshold for |I| ~ 0
EPS_ALP = 1e-14   # threshold for alpha positivity

# Density diffusivity thresholds
EPS_DER_RHO = 1e-2
D_MIN_RHO   = 1e-7
D_MAX_RHO   = 2e-5

# Ion kappa thresholds
EPS_DER_Ti = 1e-3
KAPPA_I_MIN = 1e-7
KAPPA_I_MAX = 5e-4

# Electron kappa thresholds
EPS_DER_Te = 1e-12
KAPPA_E_MIN = 1e-7
KAPPA_E_MAX = 5e-4

# Total kappa thresholds
EPS_DER_T  = 1e-12
KAPPA_T_MIN = 1e-7
KAPPA_T_MAX = 5e-4

# Extrapolate towards the core
EXTR_CORE_PSIN = 0.05

# -----------------------------
# SOL extension
# -----------------------------
EXTEND_TO_PSIN = 3.0     # extend profiles up to this psi_N
N_SOL_POINTS   = 100     # number of added points in [psi_end, EXTEND_TO_PSIN]

# Ramp in SOL: multiplier = 1 + RAMP_AMP * f(psi)
# f(psi)=0 for psi<=1, and tends to 1 for psi>>RAMP_CENTER
RAMP_AMP    = 0.0        # 0 => constant extension only; >0 ramps up in SOL
RAMP_CENTER = 1.01
RAMP_WIDTH  = 0.02        # smaller => sharper rise

FILL_BAD = True
VERBOSE  = True


def cumulative_trapz(y, x):
    """Cumulative trapezoidal integral with I[0]=0."""
    dx = np.diff(x)
    I = np.zeros_like(x, dtype=float)
    I[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return I

def prepend_zero_and_extrapolate_from(psi, X, psi_fit_min=0.1):
    """
    For final profile X(psi):
      1) Find first two points with psi >= psi_fit_min (call them (x1,y1), (x2,y2))
      2) Build a line y(x) through those two points
      3) Overwrite ALL points with psi < psi_fit_min using that line
      4) Ensure psi=0 exists: if not present, prepend (0, y(0))

    This matches: "extrapolate from 0.1 to 0, make first point 0, extrapolate linearly
    from closer points with psin > 0.1".
    """
    psi = np.asarray(psi, dtype=float).ravel()
    X   = np.asarray(X,   dtype=float).ravel()

    if psi.size != X.size:
        raise ValueError(f"core_linear_extrapolate_and_prepend_zero: length mismatch {psi.size} vs {X.size}")

    idx = np.where(psi >= psi_fit_min)[0]
    if idx.size < 2:
        raise ValueError(f"Need at least 2 points with psi >= {psi_fit_min} to extrapolate core.")

    i1, i2 = idx[0], idx[1]
    x1, x2 = psi[i1], psi[i2]
    y1, y2 = X[i1], X[i2]
    if np.abs(x2 - x1) < 1e-14:
        raise ValueError("Duplicate psi values in core extrapolation points.")

    m = (y2 - y1) / (x2 - x1)

    # Overwrite core region psi < psi_fit_min
    core = psi < psi_fit_min
    if np.any(core):
        X = X.copy()
        X[core] = y1 + m * (psi[core] - x1)

    # Prepend psi=0 if missing
    if not np.any(np.isclose(psi, 0.0, atol=1e-12)):
        y0 = y1 + m * (0.0 - x1)
        psi = np.concatenate(([0.0], psi))
        X   = np.concatenate(([y0], X))

    return psi, X

def interp_fill_nans(x, y):
    """
    Fill NaNs in y by linear interpolation in x.
    If fewer than 2 finite points exist, returns y unchanged.
    """
    y2 = y.copy()
    good = np.isfinite(y2)
    if np.count_nonzero(good) < 2:
        return y2
    y2[~good] = np.interp(x[~good], x[good], y2[good])
    return y2

def sol_ramp_arctan(psi, center=1.0, width=0.2):
    f = 0.5 + np.arctan((psi - center) / width) / np.pi
    f = np.clip(f, 0.0, 1.0)
    f = np.where(psi <= 1.0, 0.0, f)  # exactly zero in core
    return f

def extend_profile_with_sol(psi, X):
    psi = np.asarray(psi, dtype=float).ravel()
    X   = np.asarray(X,   dtype=float).ravel()
    psi_end = psi[-1]
    if EXTEND_TO_PSIN <= psi_end + 1e-12:
        return psi, X

    psi_sol = np.linspace(psi_end, EXTEND_TO_PSIN, N_SOL_POINTS + 1)[1:]
    X_sol = np.full_like(psi_sol, X[-1])  # constant tail

    if RAMP_AMP != 0.0:
        f = sol_ramp_arctan(psi_sol, center=RAMP_CENTER, width=RAMP_WIDTH)
        X_sol = X_sol * (1.0 + RAMP_AMP * f)

    return np.concatenate([psi, psi_sol]), np.concatenate([X, X_sol])
  
def compute_diffusivity(
    psi, dprof, alpha, gamma, source, scale,
    eps_der, x_min, x_max,
    hard_fail_impossible=False,
    label="X",
):
    """
    Generic Eq.(2) solver:
      X(psi) = - scale * ∫_0^psi gamma(psi') * source(psi') dpsi' / ( alpha(psi) * dprof(psi) )

    Policy:
      - unconstrained: |dprof|<=eps_der & |I|<=EPS_INT  => set NaN (filled by interpolation)
      - impossible:    |dprof|<=eps_der & |I|> EPS_INT  => NaN (or hard-fail)
      - alpha<=0, X<0, nonfinite                        => NaN (filled by interpolation)
      - then clip to [x_min, x_max]
    """
    I = cumulative_trapz(gamma * source, psi)

    with np.errstate(divide="ignore", invalid="ignore"):
        X = -scale * I / (alpha * dprof)

    alpha_ok = alpha > EPS_ALP
    d_small = np.abs(dprof) <= eps_der
    I_small = np.abs(I) <= EPS_INT

    unconstrained = d_small & I_small
    impossible = d_small & (~I_small)

    if np.any(impossible) and hard_fail_impossible:
        idx = np.where(impossible)[0][0]
        raise RuntimeError(
            f"Impossible region for {label} at psi={psi[idx]:.6g}: |d(profile)/dpsi|~0 but |I|>0. "
            "Cannot keep profile steady with finite diffusivity."
        )

    bad_alpha = ~alpha_ok
    bad_nonfinite = ~np.isfinite(X)
    bad_negative = X < 0

    X_out = X.copy()
    X_out[unconstrained] = np.nan
    X_out[impossible] = np.nan
    X_out[bad_alpha | bad_nonfinite | bad_negative] = np.nan

    if VERBOSE:
        print(f"INFO [{label}]: unconstrained={int(np.sum(unconstrained))} "
              f"impossible={int(np.sum(impossible))} alpha<=0={int(np.sum(bad_alpha))} "
              f"neg={int(np.sum(bad_negative))} nonfinite={int(np.sum(bad_nonfinite))}")

    if FILL_BAD:
        X_out = interp_fill_nans(psi, X_out)
        if np.any(~np.isfinite(X_out)):
            if VERBOSE:
                print(f"WARNING [{label}]: Too few valid points to interpolate everywhere; setting remaining NaNs to 0.")
            X_out[~np.isfinite(X_out)] = 0.0

    X_out = np.clip(X_out, x_min, x_max)
    return X_out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input text file (indices set by COL_*).")
    ap.add_argument("-o", "--output", default="D_profile.dat", help="Output file for density diffusivity D.")
    ap.add_argument("--out-kappa-i", default="kappa_i.dat", help="Output file for ion kappa.")
    ap.add_argument("--out-kappa-e", default="kappa_e.dat", help="Output file for electron kappa.")
    ap.add_argument("--out-kappa-t", default="kappa_t.dat", help="Output file for total kappa.")
    ap.add_argument("--skiprows", type=int, default=0, help="Skip header rows (default: 0)")
    ap.add_argument("--hard-fail-impossible", action="store_true",
                    help="Abort when d(profile)/dpsi~0 but |I|>0 (impossible region).")
    args = ap.parse_args()

    data = np.loadtxt(args.input, skiprows=args.skiprows)
    if data.ndim == 1:
        data = data[None, :]

    psi   = data[:, COL_PSI].astype(float)
    drho  = data[:, COL_DRHO].astype(float)
    alpha = data[:, COL_ALPHA].astype(float)
    gamma = data[:, COL_GAMMA].astype(float)
    S     = data[:, COL_S].astype(float)

    psi_axis = np.nanmedian(data[:, COL_PSI_AXIS].astype(float))
    psi_bnd  = np.nanmedian(data[:, COL_PSI_BND ].astype(float))

    # Temperature derivative + heating columns
    dTi   = data[:, COL_DTi].astype(float)
    dTe   = data[:, COL_DTe].astype(float)
    dT    = data[:, COL_DT].astype(float)
    Qi    = data[:, COL_HEATi].astype(float)
    Qe    = data[:, COL_HEATe].astype(float)
    Qtot  = data[:, COL_HEAT].astype(float)

    # Sort by psi if needed (apply to everything)
    if np.any(np.diff(psi) <= 0):
        order = np.argsort(psi)
        psi   = psi[order]
        drho  = drho[order]
        alpha = alpha[order]
        gamma = gamma[order]
        S     = S[order]
        dTi   = dTi[order]
        dTe   = dTe[order]
        dT    = dT[order]
        Qi    = Qi[order]
        Qe    = Qe[order]
        Qtot  = Qtot[order]
        if VERBOSE:
            print("WARNING: psi_N not strictly increasing; sorted by psi_N.")

    scale = (psi_bnd - psi_axis) ** 2  # no check on sign or ordering, as requested

    if VERBOSE:
        print(f"INFO: points={len(psi)}  psi_axis={psi_axis:.6g}  psi_bnd={psi_bnd:.6g}  scale={(scale):.6g}")

    D_out   = compute_diffusivity(
        psi, drho, alpha, gamma, S, scale,
        eps_der=EPS_DER_RHO, x_min=D_MIN_RHO, x_max=D_MAX_RHO,
        hard_fail_impossible=args.hard_fail_impossible, label="D"
    )

    kappa_i = compute_diffusivity(
        psi, dTi, alpha, gamma, Qi, scale,
        eps_der=EPS_DER_Ti, x_min=KAPPA_I_MIN, x_max=KAPPA_I_MAX,
        hard_fail_impossible=args.hard_fail_impossible, label="kappa_i"
    )

    kappa_e = compute_diffusivity(
        psi, dTe, alpha, gamma, Qe, scale,
        eps_der=EPS_DER_Te, x_min=KAPPA_E_MIN, x_max=KAPPA_E_MAX,
        hard_fail_impossible=args.hard_fail_impossible, label="kappa_e"
    )

    kappa_t = compute_diffusivity(
        psi, dT, alpha, gamma, Qtot, scale,
        eps_der=EPS_DER_T, x_min=KAPPA_T_MIN, x_max=KAPPA_T_MAX,
        hard_fail_impossible=args.hard_fail_impossible, label="kappa_t"
    )
    
    psi0, D_out   = prepend_zero_and_extrapolate_from(psi, D_out,   psi_fit_min=EXTR_CORE_PSIN)
    psi0,kappa_i  = prepend_zero_and_extrapolate_from(psi, kappa_i, psi_fit_min=EXTR_CORE_PSIN)
    psi0,kappa_e  = prepend_zero_and_extrapolate_from(psi, kappa_e, psi_fit_min=EXTR_CORE_PSIN)
    psi0,kappa_t  = prepend_zero_and_extrapolate_from(psi, kappa_t, psi_fit_min=EXTR_CORE_PSIN)

    # Use psi0 from now on
    psi = psi0

    
    psi_out, D_out   = extend_profile_with_sol(psi, D_out)
    psi_out, kappa_i = extend_profile_with_sol(psi, kappa_i)
    psi_out, kappa_e = extend_profile_with_sol(psi, kappa_e)
    psi_out, kappa_t = extend_profile_with_sol(psi, kappa_t)

    np.savetxt(args.output, np.column_stack([psi_out, D_out]))
    np.savetxt(args.out_kappa_i, np.column_stack([psi_out, kappa_i]))
    np.savetxt(args.out_kappa_e, np.column_stack([psi_out, kappa_e]))
    np.savetxt(args.out_kappa_t, np.column_stack([psi_out, kappa_t]))

    if VERBOSE:
        print(f"Wrote {args.output}")
        print(f"Wrote {args.out_kappa_i}")
        print(f"Wrote {args.out_kappa_e}")
        print(f"Wrote {args.out_kappa_t}")

if __name__ == "__main__":
    main()
