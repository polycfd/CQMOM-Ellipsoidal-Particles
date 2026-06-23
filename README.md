<p align="left">
  <a href="https://doi.org/10.5281/zenodo.20044545">
    <img src="https://img.shields.io/badge/DOI-10.5281/zenodo.20044545-blue" alt="Latest version">
  </a>
</p>

# CQMOM for non-spherical particles

A macroscopic Eulerian solver for tracking the evolution of non-spherical oblate ellipsoidal particle populations using the **conditional quadrature method of moments (CQMOM)**. Developed as part of the M.Sc. thesis of Mohamed Amine Bouguezzoul with the title *"Conditional Quadrature Method of Moments for Populations of Non-Spherical Particles"* at Polytechnique Montréal (2026), supervised by Bruno Blais, Bianca Viggiano, and Fabian Denner.

For a hands-on introduction to the **quadrature method of moments (QMOM)**, check out the [QMOM-Intro](https://github.com/polycfd/QMOM-Intro) repository.

---

## Overview

Simulating dispersed multiphase flows containing non-spherical particles requires solving a population balance equation (PBE) over a high-dimensional phase space. Conventional bin-based discretization is computationally intractable at 5D. This framework bypasses that bottleneck by tracking a small set of statistical **moments** of the number density function (NDF) instead of the full distribution.

The phase space is 5-dimensional:

$$\boldsymbol{\xi} = [d, \chi, u, v, w]$$

| Symbol | Description |
|--------|-------------|
| $d$ | Volume-equivalent diameter |
| $\chi$ | Aspect ratio (oblate spheroid) |
| $u, v, w$ | Translational velocity components |

---

## Key geatures

- **Generic N-dimensional CQMOM** — recursive conditional moment inversion via the adaptive Wheeler algorithm, supporting arbitrary phase-space dimensionality
- **Hybrid transport (HT)** — symmetric Strang-splitting architecture that decouples continuous aerodynamic advection from discontinuous breakage, preserving Hankel matrix positive-definiteness
- **Moment transport (MT)** — direct moment transport with an explicit SSP-RK3 solver, provided as a comparison baseline
- **Monte Carlo (MC) benchmark** — Lagrangian Monte Carlo reference for validation
- **Shape-dependent drag** — oblate spheroid with aerodynamic drag under the quasi-steady orientation (QSO) assumption
- **Surface attrition** — kinetic-energy-driven $v^2$-attrition and continuous spheroidization
- **Stochastic breakage recoil** — isotropic velocity kick with analytically closed Gaussian moment expectations
- **Gaussian copula mixture initializer** — supports lognormal, beta, and normal marginals with arbitrary correlation structure
- **Adaptive time stepping** — SSP-RK3 with embedded error estimator and time-step adaptation

---

## Repository structure

```
├── CQMOM/                        # CQMOM folder
│   ├── Tools/                    # Inventory of tools
│   │   ├── CQMOM.py              # Wheeler algorithm + N-dimensional CQMOM inversion
│   │   ├── Flux_Calculator.py    # Algorithm to compute the phase-space fluxes
│   │   ├── Initial_NDF.py        # Gaussian copula mixture initializer
│   │   └── SSP_RK_Solver.py      # SSP-RK3 time integrator and adaptive step controller
│   │
│   ├── Config.py                 # Central configuration — all tuneable parameters
│   ├── CQMOM-Generic.ipynb       # Simulation notebook for CQMOM in any dimension
│   ├── CQMOM-Particles.ipynb     # Simulation notebook for non-spherical particles
│   └── Media                     # Where media files are stored
│
└── QSO-Assumption                # QSO folder
    ├── QSO-Notebook.ipynb        # Testing the quasi-steady orientation assumption
    └── Media                     # Where media files are stored

```

---

## Quick start

### Requirements

```bash
pip install numpy scipy matplotlib jupyter
```

### Running the main simulation

Open and run the `CQMOM-Particles.ipynb` notebook. All physical, numerical, and plotting parameters are controlled centrally through `Config.py` — no hard-coded constants exist in the notebooks.

### Configuration (`Config.py`)

The configuration is organized into sections:

| Section | Key parameters |
|---------|---------------|
| `simulation` | `dim`, `time_step`, `simulation_time`, `num_particles` |
| `initial_conditions` | `N` (nodes per dim), `moments_idx` |
| `mixture` | `mixture_weights`, `dim_types`, per-mode means/stds |
| `physics` | `rho_p`, `rho_f`, `mu_f`, `g` |
| `breakage` | `frag_rate_const`, `frag_power`, `min_frag_size` |
| `attrition` | `tau_relax`, `attrition_rate_const` |
| `restitution` | `restitution_factor`, `sigma_kick` |
| `fluid_forcing` | `freq`, `u_amp`, `v_amp`, `w_amp` |
| `cqmom` | `rcond_ht`, `rcond_mt`, `rmin`, `eabs`, `cutoff` |

A time-stamped output folder with a JSON config snapshot and plain-text summary is automatically created on each run.

---

## Numerical methods

### CQMOM inversion (`CQMOM.py`)

The multivariate inversion follows a hierarchical chain:

1. **Primary inversion** — Wheeler algorithm on marginal moments $M_{k,0,\ldots,0}$
2. **Conditional inversion** — Vandermonde system solved via pseudoinverse to extract conditional moments at each primary node
3. **Recursion** — repeated for each subsequent dimension, conditioning on all prior nodes

The adaptive Wheeler algorithm monitors diagonal elements of the sigma table and dynamically reduces quadrature order to prevent singularity when distributions become nearly monodisperse.

### Moment Transport (MT) — fully coupled baseline

The MT method advances the moment tensor $\mathbf{M}_\mathbf{k}$​ by solving the moment transport equations via SSP-RK3 using a single coupled operator: 

$$\mathcal{L}_\text{MT} = \mathcal{L}^{\text{adv}}_{\Delta t} + \mathcal{L}^{\text{break}}_{\Delta t}$$

The corresponding MTE is:

$$\mathbf{M}_\mathbf{k}^{n+1} = \mathbf{M}_\mathbf{k}^{n} + \Delta t \left( \mathcal{F}^{\text{adv}}_\mathbf{k} + \mathcal{S}^{\text{break}}_\mathbf{k} \right)$$

where the advection flux $\mathcal{F}^{\text{adv}}$ and the breakage source $\mathcal{S}^{\text{break}}$ are both closed via CQMOM at every Runge–Kutta sub-stage, requiring three full inversions per time step.

When competing continuous and discontinuous fluxes are processed simultaneously, the Hankel matrices become ill-conditioned, forcing the adaptive Wheeler algorithm into localized low-order fallbacks that corrupt kinematic variance and geometric mass conservation. The MT solver is retained as a reference: its failure modes directly quantify the mathematical stiffness that the HT architecture is designed to circumvent.

### Hybrid Transport (HT) — Strang splitting

The HT method advances the full moment tensor $\mathbf{M}_\mathbf{k}$​ by solving the moment transport equations using a Strang operator splitting: 

$$\mathcal{L}_\text{HT} = \mathcal{L}^{\text{adv}}_{\Delta t/2} \circ \mathcal{L}^{\text{break}}_{\Delta t} \circ \mathcal{L}^{\text{adv}}_{\Delta t/2}$$

1. **Half-step advection** — integrate physical ODEs for $\Delta t/2$, reconstruct moments from advected nodes
2. **Full-step breakage** — update moments via SSP-RK3 under the isolated breakage source term
3. **CQMOM inversion** — reconstruct nodes 
4. **Half-step advection** — integrate remaining $\Delta t/2$, reconstruct final moments

This architecture isolates the expensive matrix inversion to a single evaluation per time step, achieving (in our experience) up to **50× speedup** over the MT approach.

### SSP-RK3 time integration (`SSP_RK_Solver.py`)

Strong-stability-preserving third-order Runge–Kutta scheme with an embedded second-order error estimate for adaptive time-stepping. The SSP property prevents numerical dispersion from pushing the moment set outside the realizable space.

---

## Physical model

### Drag on oblate spheroids

Under the **quasi-steady orientation (QSO)** assumption, justified by a time-scale analysis showing that the rotational timescale is smaller than the translational timescale, each particle is assumed to adopt a drag-maximizing orientation. The effective drag coefficient is modelled using correlations from Ouchene (2020), but other drag models for oblate spheroids are equally applicable.

### Breakage

Power-law rate kernel: $R_{\text{break}}(d) = k \cdot d^\gamma$

Volume-conserving binary split: $d_{\text{daughter}} = d_{\text{parent}} / 2^{1/3}$

Stochastic recoil velocity: $\mathbf{v}_{\text{recoil}} \sim \mathcal{N} (0, \sigma _{\text{recoil}}^2  \mathbf{I})$

The Gaussian moment expectation for the recoil is analytically closed inside the CQMOM fluxes, no random sampling is required.

### Surface attrition

$$\frac{\text{d}d}{\text{d}t} = -k_{\text{shear}} \|\mathbf{u}_\mathrm{f} - \mathbf{u}\|^2, \qquad \frac{\text{d}\chi}{\text{d}t} = \frac{1}{\tau_{\text{shape}}}(1 - \chi)$$

---

## Validation

Three test cases of increasing complexity are provided in the notebook `CQMOM-Particles.ipynb`:

| Test Case | Physics active |
|-----------|----------------|
| Monomodal settling | Drag + binary breakage |
| Multimodal settling | Drag + binary breakage + velocity recoil|
| Multimodal jets / fully coupled swirling flow | Drag + attrition + breakage + recoil |

> For the first and second test case, you might consider editing the discontinuous source term for $\chi$ so that: $\chi _\text{daughter} = \alpha \chi _\text{parent} + (1-\alpha) $.

All CQMOM results are validated against Monte Carlo simulations tracking $N_p = 10{,}000$ particles.

---

## Results summary

| Metric | Monte Carlo | HT | MT |
|--------|-------------|----|----|
| Geometric fidelity | ✅ Reference | ✅ Excellent | ⚠️ Degrades under stiffness |
| Granular temperature | ✅ Reference | ✅ Good | ❌ Severe collapses |
| Realizability | ✅ Always | ✅ Maintained | ❌ Hankel matrix failures |
| Runtime (monomodal) | 25 s | **2.5 s** | 70 s |
| Runtime (fully coupled) | 27 s | **3 s** | 156 s |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{Bouguezzoul2026,
  author  = {Mohamed Amine Bouguezzoul},
  title   = {Conditional Quadrature Method of Moments for Populations
             of Non-Spherical Particles},
  school  = {Polytechnique Montréal},
  year    = {2026},
  month   = {May},
}
```

---

## Acknowledgements

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC, funding reference ALLRP-592877-24), PRIMA Québec, and Nouveau Monde Graphite.

---

## License

This code is under the copyright of its developers and made available as open-source software under the terms of the [MIT License](LICENSE).

---

## Related References

- [Marchisio & Fox (2013)](https://doi.org/10.1017/CBO9781139016599) — *Computational Models for Polydisperse Particulate and Multiphase Systems*
- [Yuan & Fox (2011)](https://www.doi.org/10.1016/j.jcp.2011.07.020) — CQMOM formulation
- [Ouchene (2020)](https://www.doi.org/10.1063/5.0011618) — Oblate spheroid drag and torque correlations
- [Strang (1968)](https://www.doi.org/10.1137/0705041) — Operator splitting
- [Heylmun, Fox & Passalacqua](https://www.doi.org/10.1016/j.cpc.2021.108072) (2021) — Joint size-velocity QBMM