"""
Config.py - Central configuration for the 5-D CQMOM particle simulation.

All tuneable parameters live here. The notebook and scripts import this file
and must not hard-code any physical, numerical, or aesthetic constant.

Sections
--------
1. simulation      - time horizon, fixed time step, dimensionality
2. initial_conditions - CQMOM quadrature structure and distribution parameters
3. mixture_modes   - per-mode means / stds for the Gaussian copula initialiser
4. physics         - fluid & particle material properties
5. breakage        - fragmentation model constants
6. attrition       - surface attrition model constants
7. restitution     - post-breakage velocity model
8. fluid_forcing   - oscillating background flow
9. time_stepping   - adaptive RK controller bounds and tolerances
10. plotting        - figure dimensions, font sizes, colour palette
11. output          - folder structure and save flags
"""

import json
import os
import numpy as np
from datetime import datetime

# =============================================================================
# 1. SIMULATION
# =============================================================================
_SIMULATION = {
    "dim":             5,       # state-vector length: [d, χ, u, v, w]
    "time_step":       1.0e-2,  # initial / fixed time step [s]
    "simulation_time": 1.0,     # total simulation horizon  [s]
    "num_particles":   10_000,  # Monte Carlo ensemble size
}

# =============================================================================
# 2. INITIAL CONDITIONS - CQMOM quadrature structure
# =============================================================================
_INITIAL_CONDITIONS = {
    # Number of quadrature nodes per dimension [d, χ, u, v, w]
    # The moment tensor has shape (2*N[0], 2*N[1], …)
    "N": [2, 1, 2, 2, 2],

    # Total number of tracked moments per dimension (must be ≥ 2*N[i])
    "moments_idx": [4, 2, 4, 4, 4]
}

# =============================================================================
# 3. MIXTURE MODES - Gaussian copula initialiser
#
# Each mode is a dict with:
#   "means"  : [d, χ, u, v, w]   physical-space target means
#   "stds"   : [d, χ, u, v, w]   physical-space target standard deviations
#
# "mixture_weights" must sum to 1.  Set weight to 0.0 to deactivate a mode.
# "dim_types" selects the marginal family for each dimension.
# =============================================================================
_MIXTURE = {
    "mixture_weights": [0.5, 0.5],   # [mode-1 weight, mode-2 weight]

    "dim_types": ["lognormal", "beta", "normal", "normal", "normal"],

    "mode_1": {
        # Primary population - spherical-ish particles, quiescent
        "means": [130.0e-6,  0.3,  1.0,  1.0,  0.0],
        "stds":  [  2.0e-5,  0.1,  0.5,  0.5,  0.5],
    },

    "mode_2": {
        # Secondary population - same size, drifting flow (inactive by default)
        "means": [130.0e-6,  0.3,  1.0, -1.0,  0.0],
        "stds":  [  2.0e-5,  0.1,  0.5,  0.5,  0.5],
    },
}

# =============================================================================
# 4. PHYSICS - material properties
# =============================================================================
_PHYSICS = {
    "rho_p": 2000.0,   # particle density   [kg/m³]
    "rho_f": 1.2,      # fluid density      [kg/m³]
    "mu_f":  1.8e-5,   # dynamic viscosity  [Pa·s]
    "g":     9.81,     # gravitational acc. [m/s²]
}

# =============================================================================
# 5. BREAKAGE MODEL
#
# Power-law rate:  b(d) = frag_rate_const * d^frag_power   [s⁻¹]
# Binary split:    d' = d / 2^(1/3)    (volume-conserving)
# Aspect ratio relaxation:   χ' = χ + relaxation_factor * (1 − χ)
# =============================================================================
_BREAKAGE = {
    "frag_rate_const":   1.0e8,   # prefactor k_b                [s⁻¹ m⁻frag_power]
    "frag_power":        2.0,     # diameter exponent p_b        [-]
    "min_frag_size":     1.0e-6,  # smallest breakable diameter  [m]
    "relaxation_factor": 0.5,     # aspect-ratio relaxation      [-]
}

_ATTRITION = {
    "tau_relax": 2.0,                # timescale for shape relaxation towards sphericity (chi=1) [s]
    "attrition_rate_const": 1.0e-3,  # attrition rate constant (size reduction proportional to u_rel²) [s⁻¹ m⁻¹]
}

# =============================================================================
# 6. RESTITUTION & KICK MODEL
#
# After breakage each daughter inherits:
#   u_d = restitution_factor * u_parent  ±  kick ~ N(0, sigma_kick²)
#
# Set restitution_factor = 1.0 for fully elastic (momentum-preserving) split.
# Set sigma_kick = 0.0 to disable the velocity dispersion at breakup.
# =============================================================================
_RESTITUTION = {
    "restitution_factor": 0.5,   # fraction of parent velocity kept   [-]
    "sigma_kick":         0.5,   # std-dev of post-breakage kick      [m/s]
}

# =============================================================================
# 7. FLUID FORCING - oscillating background flow
#
# u_f(t) = u_amp * cos(2π * freq * t)
# v_f(t) = v_amp * sin(2π * freq * t)
# w_f(t) = 0
# =============================================================================
_FLUID_FORCING = {
    "freq":  1.0,   # oscillation frequency  [Hz]
    "u_amp": 2.0,   # u-component amplitude  [m/s]
    "v_amp": 2.0,   # v-component amplitude  [m/s]
    "w_amp": 0.0,   # w-component amplitude  [m/s]
}

# =============================================================================
# 8. ADAPTIVE TIME STEPPING (SSP-RK3 controller)
# =============================================================================
_TIME_STEPPING = {
    "min_dt":    5.0e-3,   # minimum allowed time step  [s]
    "max_dt":    1.0,      # maximum allowed time step  [s]
    "error_tol": 1.0e-8,   # local truncation error target
}

# =============================================================================
# 9. CQMOM INVERSION
#
# =============================================================================
_CQMOM = {
    "adaptive": True,
    "rmin":     [1.0e-30] * 5,   # minimum weight threshold per dimension
    "eabs":     [1.0e-30] * 5,   # absolute eigenvalue tolerance per dimension
    "cutoff":   1.0e-6,          # global weight cutoff
    "rcond_mt": 0.25,            # reciprocal conditioning number for the pseudo-inverse applied in MT
}

# =============================================================================
# 10. PLOTTING
# =============================================================================
_PLOTTING = {
    # ── Figure geometry ───────────────────────────────────────────────────────
    "fontـsize":       28,
    "font_size_tick":  26,
    "font_size_legend":26,
    "ncols":           3,
    "panel_width":     8,    # width per column  [inches]
    "panel_height":    4,    # height per row    [inches]
    "dpi":             300,

    # ── Method colour palette (Wong colour-blind safe) ────────────────────────
    "colors": {
        "mc": "#2ca02c",   # green  - Monte Carlo  (ground truth)
        "mt": "#d62728",   # red    - Moment Transport
        "ht": "#000000",   # black  - Hybrid Transport
    },

    # ── Per-dimension histogram colours (initial distribution plot) ───────────
    "dim_colors": {
        "d":  "#E69F00",   # diameter
        "chi": "#56B4E9",  # aspect ratio
        "u":  "#009E73",   # u velocity
        "v":  "#0072B2",   # v velocity
        "w":  "#D55E00",   # w velocity
    },

    # ── rcParams style sheet (applied via plt.rcParams.update) ───────────────
    "style": {
        "font.size":                  28,
        "font.family":                "serif",
        "font.serif":                 ["Times New Roman"],
        "mathtext.fontset":           "cm",
        "text.usetex":                False,
        "axes.grid":                  True,
        "grid.alpha":                 0.4,
        "grid.linestyle":             "--",
        "grid.linewidth":             0.7,
        "lines.linewidth":            3.5,
        "axes.formatter.use_mathtext":True,
        "axes.unicode_minus":         False,
    },
}

# =============================================================================
# 11. OUTPUT
# =============================================================================
_OUTPUT = {
    "base_folder":   "Media",
    "save_figures":  True,
    "save_config":   True,
    "figure_format": "pdf",   # "pdf" | "png" | "svg"
}


# =============================================================================
# ASSEMBLY - single dict exported to the rest of the codebase
# =============================================================================
config = {
    "simulation":         _SIMULATION,
    "initial_conditions": _INITIAL_CONDITIONS,
    "mixture":            _MIXTURE,
    "physics":            _PHYSICS,
    "breakage":           _BREAKAGE,
    "attrition":          _ATTRITION,
    "restitution":        _RESTITUTION,
    "fluid_forcing":      _FLUID_FORCING,
    "time_stepping":      _TIME_STEPPING,
    "cqmom":              _CQMOM,
    "plotting":           _PLOTTING,
    "output":             _OUTPUT,
}


# =============================================================================
# HELPER - create the output folder and persist the run configuration
# =============================================================================
def create_simulation_folder(cfg: dict) -> str:
    """
    Create a timestamped output folder and write a JSON + plain-text summary.

    The folder name encodes the key numerical parameters so that results are
    self-describing on disk:

        Media/5D_CQMOM_<timestamp>_dt=1e-04_tol=1e-06_N=[2,1,2,2,2]/

    Parameters
    ----------
    cfg : dict
        The global ``config`` dict (or any compatible mapping).

    Returns
    -------
    str
        Absolute path to the newly created folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ts   = cfg["time_stepping"]
    ic   = cfg["initial_conditions"]
    sim  = cfg["simulation"]
    brk  = cfg["breakage"]

    folder_name = (
    f"5D_CQMOM_{timestamp}"
    f"_dt={ts['min_dt']:.0e}"
    f"_tol={ts['error_tol']:.0e}"
    f"_N={ic['N']}"
    f"_kb={brk['frag_rate_const']:.0e}"
    f"_ka={cfg['attrition']['attrition_rate_const']:.0e}"
    )
    
    folder_path = os.path.join(cfg["output"]["base_folder"], folder_name)
    os.makedirs(folder_path, exist_ok=True)

    if not cfg["output"]["save_config"]:
        return folder_path

    # ── JSON snapshot (machine-readable) ─────────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    config_path = os.path.join(folder_path, "simulation_config.json")
    with open(config_path, "w") as fh:
        json.dump(cfg, fh, indent=4, default=_serialise)

    # ── Plain-text summary (human-readable) ───────────────────────────────────
    summary_path = os.path.join(folder_path, "simulation_summary.txt")
    mix = cfg["mixture"]
    ret = cfg["restitution"]
    phys = cfg["physics"]
    ff  = cfg["fluid_forcing"]

    with open(summary_path, "w") as fh:
        def w(line=""):
            fh.write(line + "\n")

        w(f"5-D CQMOM Particle Simulation  -  {timestamp}")
        w("=" * 60)

        w("\n[Simulation]")
        w(f"  Total time          : {sim['simulation_time']} s")
        w(f"  Initial time step   : {sim['time_step']} s")
        w(f"  MC ensemble size    : {sim['num_particles']:,}")
        w(f"  State dimensions    : {sim['dim']}  [d, χ, u, v, w]")
        w(f"  Dim types           : {mix['dim_types']}")

        w("\n[CQMOM]")
        w(f"  Nodes per dim (N)   : {ic['N']}")
        w(f"  Moments per dim     : {ic['moments_idx']}")
        w(f"  Adaptive inversion  : {cfg['cqmom']['adaptive']}")
        w(f"  rcond MT            : {cfg['cqmom']['rcond_mt']}")

        w("\n[Adaptive Time Stepping]")
        w(f"  Min dt              : {ts['min_dt']}")
        w(f"  Max dt              : {ts['max_dt']}")
        w(f"  Error tolerance     : {ts['error_tol']}")

        w("\n[Physics]")
        w(f"  Particle density ρ_p: {phys['rho_p']} kg/m³")
        w(f"  Fluid density    ρ_f: {phys['rho_f']} kg/m³")
        w(f"  Fluid viscosity  μ_f: {phys['mu_f']} Pa·s")
        w(f"  Gravity          g  : {phys['g']} m/s²")

        w("\n[Breakage Model]")
        w(f"  Rate constant k_b   : {brk['frag_rate_const']:.2e} s⁻¹ m⁻{brk['frag_power']}")
        w(f"  Diameter exponent   : {brk['frag_power']}")
        w(f"  Min fragment size   : {brk['min_frag_size']:.2e} m")
        w(f"  AR relaxation factor: {brk['relaxation_factor']}")

        w("\n[Attrition Model]")
        w(f"  Tau relax          : {cfg['attrition']['tau_relax']} s")
        w(f"  Attrition rate const: {cfg['attrition']['attrition_rate_const']} s⁻¹ m⁻¹")

        w("\n[Restitution / Kick]")
        w(f"  Restitution factor  : {ret['restitution_factor']}")
        w(f"  Kick sigma          : {ret['sigma_kick']} m/s")

        w("\n[Fluid Forcing]")
        w("  Model               : u=Ucos(2πft), v=Vsin(2πft), w=const")
        w(f"  Frequency           : {ff['freq']} Hz")
        w(f"  u amplitude         : {ff['u_amp']} m/s")
        w(f"  v amplitude         : {ff['v_amp']} m/s")
        w(f"  w amplitude         : {ff['w_amp']} m/s")

        w("\n[Mixture Modes]")
        w(f"  Weights             : {mix['mixture_weights']}")
        for mode_key in ("mode_1", "mode_2"):
            m = mix[mode_key]
            w(f"  {mode_key}  means = {m['means']}")
            w(f"           stds  = {m['stds']}")

        w("\n[Derived]")
        w(f"  Max fluid speed     : {np.sqrt(ff['u_amp']**2 + ff['v_amp']**2):.2f} m/s")

        w("\n" + "=" * 60)

    return folder_path