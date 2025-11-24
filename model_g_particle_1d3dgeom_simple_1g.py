#!/usr/bin/env python3
"""
Model G Particle – 1D / 1D cylindrical / 1D spherical
Clean, no-checkpoint version that just runs and makes an MP4.

- Written by Brendan Darrer aided by ChatGPT5.1 date: 23rd November 2025 - adapted 24.11.2025 16:07 GMT
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb
- with ChatGPT5.1 writing the code and Brendan guiding it to produce a clean code.

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+

Usage examples:

python3 model_g_particle_1d3dgeom_simple_1g.py.py --geometry 1d
python3 model_g_particle_1d3dgeom_simple_1g.py --geometry 1d_cyl
python3 model_g_particle_1d3dgeom_simple_1g.py --geometry 1d_sph

Requires: pip install numpy scipy matplotlib imageio imageio[ffmpeg]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio

# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--geometry",
    type=str,
    default="1d",
    choices=["1d", "1d_cyl", "1d_sph"],
    help="Geometry: 1d (Cartesian), 1d_cyl (radial 2D), 1d_sph (radial 3D)",
)
parser.add_argument(
    "--L",
    type=float,
    #default=40.0,
    #default=10.0,
    default=7.5,
    help="Domain size (1d: total length; radial: outer radius)",
)
parser.add_argument(
    "--Tfinal",
    type=float,
    default=40.0,
    help="Final simulation time",
)
parser.add_argument(
    "--nx",
    type=int,
    #default=401,
    default=201,
    help="Number of spatial points",
)
parser.add_argument(
    "--nt_anim",
    type=int,
    default=300,
    help="Number of time samples for animation",
)
args = parser.parse_args()

GEOMETRY = args.geometry
L = args.L
Tfinal = args.Tfinal
nx = args.nx
nt_anim = args.nt_anim

run_name = f"{GEOMETRY}__model_g_particle_1d3dgeom_simple_1g" 
out_dir = f"out_{run_name}"
frames_dir = os.path.join(out_dir, "frames")
mp4_path = os.path.join(out_dir, f"{run_name}.mp4")

os.makedirs(frames_dir, exist_ok=True)

# ------------------------------------------------------------
# Parameters (same as your eqs17) :contentReference[oaicite:1]{index=1}
# ------------------------------------------------------------
params = {
    "a": 14.0,
    "b": 29.0,
    "dx": 1.0,
    "dy": 12.0,
    "p": 1.0,
    "q": 1.0,
    "g": 0.1,
    "s": 0.0,
    "u": 0.0,
    "v": 0.0,  # advection magnitude (kept for completeness, normally 0)
    "w": 0.0,
}

# ------------------------------------------------------------
# Seed forcing chi(x,t) – same as in your code :contentReference[oaicite:2]{index=2}
# ------------------------------------------------------------
def bell(s, x):
    return np.exp(- (x / s) ** 2 / 2.0)


nseeds = 1
Tseed = 10.0


def chi(x, t):
    """
    Seed function χ(x,t).
    For 1d (Cartesian): x ∈ [-L/2, L/2], so symmetric seeds ok.
    For radial geometries (1d_cyl, 1d_sph): x = r ∈ [0, L], so seeds must not use negative radii.
    In Option B, mirroring happens only in plotting, not in χ.
    """

    # --- 1D CARTESIAN ---------------------------------------------------------
    if dim == 1:
        if nseeds == 1:
            return -bell(1.0, x) * bell(3.0, t - Tseed)

        elif nseeds == 2:
            return -(bell(1.0, x + 3.303 / 2) +
                     bell(1.0, x - 3.303 / 2)) * bell(3.0, t - Tseed)

        elif nseeds == 3:
            return -(bell(1.0, x + 3.314) +
                     bell(1.0, x) +
                     bell(1.0, x - 3.314)) * bell(3.0, t - Tseed)

    # --- RADIAL GEOMETRIES (1d_cyl or 1d_sph) --------------------------------
    else:
        # In radial coords, r>=0, so only use positive-side seeds.
        if nseeds == 1:
            return -bell(1.0, x) * bell(3.0, t - Tseed)

        elif nseeds == 2:
            # Only place seeds at positive radii.
            r0 = 3.303 / 2
            return -(bell(1.0, x - r0)) * bell(3.0, t - Tseed)

        elif nseeds == 3:
            r0 = 3.314
            return -(bell(1.0, x) +
                     bell(1.0, x - r0)) * bell(3.0, t - Tseed)

    # fallback
    return np.zeros_like(x)

# ------------------------------------------------------------
# Grid & geometry
# ------------------------------------------------------------
if GEOMETRY == "1d":
    dim = 1
    xgrid = np.linspace(-L / 2, L / 2, nx)  # Cartesian line
elif GEOMETRY == "1d_cyl":
    dim = 2
    xgrid = np.linspace(0.0, L, nx)  # radial coordinate r in [0, L]
    #xgrid = np.linspace(-L / 2, L / 2, nx)  # radial coordinate r in [0, L]
elif GEOMETRY == "1d_sph":
    dim = 3
    xgrid = np.linspace(0.0, L, nx)  # radial coordinate r in [0, L]
    #xgrid = np.linspace(-L / 2, L / 2, nx)  # radial coordinate r in [0, L]
dx_space = xgrid[1] - xgrid[0]

# ------------------------------------------------------------
# Homogeneous steady state (G0, X0, Y0) :contentReference[oaicite:3]{index=3}
# ------------------------------------------------------------
a = params["a"]
b = params["b"]
p_par = params["p"]
q_par = params["q"]
g_par = params["g"]
s_par = params["s"]
u_par = params["u"]
w_par = params["w"]

G0 = (a + g_par * w_par) / (q_par - g_par * p_par)
X0 = (p_par * a + q_par * w_par) / (q_par - g_par * p_par)
Y0 = ((s_par * X0**2 + b) * X0 / (X0**2 + u_par)) if (X0**2 + u_par) != 0 else 0.0

print(f"[{GEOMETRY}] Homogeneous state: G0={G0:.6g}, X0={X0:.6g}, Y0={Y0:.6g}")


# ------------------------------------------------------------
# Differential operators (same as your radial logic) :contentReference[oaicite:4]{index=4}
# ------------------------------------------------------------
def laplacian(u):
    """
    Geometry-aware Laplacian:
      dim = 1: standard 1D Cartesian Laplacian u_xx.
      dim = 2: cylindrical symmetry  -> u_rr + (1/r) u_r.
      dim = 3: spherical symmetry    -> u_rr + (2/r) u_r.
    """
    if dim == 1:
        dudxx = np.zeros_like(u)
        dudxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx_space**2
        return dudxx

    # radial case (dim = 2 or 3)
    lap = np.zeros_like(u)
    r = xgrid
    d = dim

    # r=0 regularity: Lap u(0) = d * u''(0), with u''(0) ≈ 2(u1-u0)/dx^2
    lap[0] = 2.0 * d * (u[1] - u[0]) / dx_space**2

    u_ip1 = u[2:]
    u_i = u[1:-1]
    u_im1 = u[:-2]
    r_i = r[1:-1]

    u_xx = (u_ip1 - 2 * u_i + u_im1) / dx_space**2
    u_x = (u_ip1 - u_im1) / (2 * dx_space)

    lap[1:-1] = u_xx + (d - 1.0) * u_x / r_i

    # outer boundary Lap will be overridden by BC via d/dt
    return lap


def grad_1d(u):
    """Simple centered gradient in x or r; used only if v != 0."""
    dudx = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx_space)
    dudx[0] = dudx[-1] = 0.0
    return dudx


# ------------------------------------------------------------
# Packing / unpacking
# ------------------------------------------------------------
def pack(pG, pX, pY):
    return np.concatenate([pG, pX, pY])


def unpack(y):
    return y[:nx], y[nx : 2 * nx], y[2 * nx : 3 * nx]


# ------------------------------------------------------------
# RHS of PDE system (eqs13 with eqs17 params) :contentReference[oaicite:5]{index=5}
# ------------------------------------------------------------
def rhs(t, y_flat):
    pG, pX, pY = unpack(y_flat)

    lapG = laplacian(pG)
    lapX = laplacian(pX)
    lapY = laplacian(pY)

    dGdx = grad_1d(pG)
    dXdx = grad_1d(pX)
    dYdx = grad_1d(pY)

    chi_vec = chi(xgrid, t)

    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s = s_par * (Xtot**3 - X0**3)
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)

    dpGdt = lapG - q_par * pG + g_par * pX - params["v"] * dGdx
    dpXdt = (
        params["dx"] * lapX
        - params["v"] * dXdx
        + p_par * pG
        - (1.0 + b) * pX
        + u_par * pY
        - nonlinear_s
        + nonlinear_xy
        + chi_vec
    )
    dpYdt = (
        params["dy"] * lapY
        - params["v"] * dYdx
        + b * pX
        - u_par * pY
        - nonlinear_xy
        + nonlinear_s
    )

    # Boundary conditions:
    # 1d: Dirichlet at both ends
    # radial: r=0 regular from Laplacian, r=L Dirichlet
    if dim == 1:
        dpGdt[0] = dpGdt[-1] = 0.0
        dpXdt[0] = dpXdt[-1] = 0.0
        dpYdt[0] = dpYdt[-1] = 0.0
    else:
        dpGdt[-1] = dpXdt[-1] = dpYdt[-1] = 0.0

    return pack(dpGdt, dpXdt, dpYdt)

"""
# ------------------------------------------------------------
# Run integration (no checkpoints, just one solve)
# ------------------------------------------------------------
y0 = np.zeros(3 * nx)
t_eval = np.linspace(0.0, Tfinal, nt_anim)
"""
# ------------------------------------------------------------
# Initial condition (must be outside the commented block)
# ------------------------------------------------------------
y0 = np.zeros(3 * nx)   # <-- ADD THIS BACK

# ------------------------------------------------------------
# Segmented integration (writes frames during the simulation)
# ------------------------------------------------------------
segment_dt = Tfinal / nt_anim   # time between frames
t_frames = np.linspace(0, Tfinal, nt_anim)
next_frame = 0

t_curr = 0.0
y_curr = y0.copy()

print(f"[{GEOMETRY}] Running segmented integration...")

while t_curr < Tfinal and next_frame < nt_anim:
    t_next = t_frames[next_frame]

    # Solve a very small time segment
    seg = solve_ivp(
        rhs,
        (t_curr, t_next),
        y_curr,
        method="LSODA",
        atol=1e-6,
        rtol=1e-6,
        dense_output=True
    )

    if not seg.success:
        print("WARNING: segment failed:", seg.message)

    # Update state
    y_curr = seg.y[:, -1]
    t_curr = seg.t[-1]

    # Save frame for this time
    pG, pX, pY = unpack(y_curr)

    #fig, ax = plt.subplots(figsize=(10, 5))
    
    fig, ax = plt.subplots(figsize=(10, 5))

    if dim == 1:
        # Normal 1D line
        ax.plot(xgrid, pY, label="pY (Y)", linewidth=1.5)
        ax.plot(xgrid, pG, label="pG (G)")
        ax.plot(xgrid, pX/10.0, label="pX/10 (X scaled)")
    else:
        # Mirror radial field for visual 1D slice
        xx = np.concatenate((-xgrid[::-1], xgrid))
        ax.plot(xx, np.concatenate((pY[::-1], pY)),      label="pY (mirrored)", linewidth=1.5)
        ax.plot(xx, np.concatenate((pG[::-1], pG)),      label="pG (mirrored)")
        ax.plot(xx, np.concatenate((pX[::-1], pX))/10.0, label="pX/10 (mirrored)")
    
    ax.set_title(f"Model G {GEOMETRY} — t={t_curr:.3f}")
    ax.set_xlabel("Space x")
    ax.set_ylabel("Potential pX/10, pY, pG")
    ax.grid(True)
    ax.legend()
    #ax.set_ylim(-1.0, 1.0)
    ax.set_ylim(-2.0, 2.0)

    fpath = os.path.join(frames_dir, f"frame_{next_frame:04d}.png")
    fig.savefig(fpath, dpi=120)
    plt.close(fig)

    next_frame += 1
    print(f"[{GEOMETRY}] Saved frame {next_frame}/{nt_anim} at t={t_curr:.3f}")

"""
print(f"[{GEOMETRY}] Solving PDE...")
sol = solve_ivp(
    rhs,
    (0.0, Tfinal),
    y0,
    #method="BDF",
    method="LSODA",
    t_eval=t_eval,
    atol=1e-6,
    rtol=1e-6,
)

if not sol.success:
    print("WARNING: solve_ivp reported failure:", sol.message)

Y_all = sol.y  # shape (3*nx, nt_anim)

# ------------------------------------------------------------
# Make frames
# ------------------------------------------------------------
print(f"[{GEOMETRY}] Rendering frames to {frames_dir}...")
for i, t in enumerate(t_eval):
    pG, pX, pY = unpack(Y_all[:, i])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xgrid, pY, label="pY (Y)", linewidth=1.5)
    ax.plot(xgrid, pG, label="pG (G)")
    ax.plot(xgrid, pX / 10.0, label="pX/10 (X scaled)")
    ax.set_title(f"Model G {GEOMETRY} — t={t:.3f}")
    ax.set_xlabel("Space x" if dim == 1 else "Radius r")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-1.0, 1.0)

    fpath = os.path.join(frames_dir, f"frame_{i:04d}.png")
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
"""
# ------------------------------------------------------------
# Assemble MP4
# ------------------------------------------------------------
print(f"[{GEOMETRY}] Assembling MP4: {mp4_path}")
with imageio.get_writer(mp4_path, fps=15) as writer:
    for i in range(nt_anim):
        fpath = os.path.join(frames_dir, f"frame_{i:04d}.png")
        img = imageio.imread(fpath)
        writer.append_data(img)

print(f"[{GEOMETRY}] Done. Video saved to {mp4_path}")

