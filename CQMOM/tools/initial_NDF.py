import numpy as np
import itertools
from scipy.stats import norm, beta, lognorm, multivariate_normal

def mc_gaussian_moments(mu, cov, N, num_seeds=int(1e6)):
    """
    Generate multivariate Gaussian raw moments using Monte Carlo sampling.
    """

    nodes = np.random.multivariate_normal(mu, cov, size=num_seeds)

    weights = np.ones(num_seeds) / num_seeds
    moments = np.zeros(N)

    for idx in itertools.product(*[range(n) for n in N]):
        monomial = np.prod([nodes[:, i] ** idx[i] for i in range(len(idx))], axis=0)
        moments[idx] = np.sum(weights * monomial)

    return moments, nodes, weights

def calculate_beta_params(mu, std):
    """
    Converts desired Mean and Std Dev of a Beta distribution [0,1]
    into shape parameters alpha (a) and beta (b).
    """
    
    var = std**2
    
    # Constraint check: variance cannot exceed mu*(1-mu) for a Beta on [0,1]
    max_var = mu * (1.0 - mu)
    if var >= max_var*(1-1e-12):
        raise ValueError(f"Std dev {std} is too large for mean {mu} on [0,1]. "
                         f"Max possible std is {np.sqrt(max_var):.4f}")
    if mu <= 0 or mu >= 1:
         raise ValueError("Mean must be strictly between 0 and 1.")

    # Standard method of moments formulas
    # Common term K = mu(1-mu)/var - 1
    K = (mu * (1.0 - mu) / var) - 1.0
    alpha = mu * K
    beta_val = (1.0 - mu) * K
    
    # Ensure numerical stability
    alpha = max(alpha, 1e-4)
    beta_val = max(beta_val, 1e-4)
    
    return alpha, beta_val

def calculate_lognormal_underlying_params(mu_phys, std_phys):
    """
    Converts physical Mean and Std Dev into the underlying 
    Normal 'mu' and 'sigma' parameters required by scipy.stats.lognorm.
    """
    if mu_phys <= 0: raise ValueError("Physical mean for lognormal must be > 0")
    
    cov_sq = (std_phys / mu_phys)**2 # Coefficient of variation squared
    
    sigma_ln = np.sqrt(np.log(1.0 + cov_sq))
    mu_ln = np.log(mu_phys) - 0.5 * sigma_ln**2
    
    # Note: scipy lognorm uses 's' for sigma_ln and 'scale' for exp(mu_ln)
    return sigma_ln, np.exp(mu_ln)

def generate_copula_mixture_conditions(mixture_weights, mode_configs, dim_types, N_moments_shape, number_density=1.0, num_seeds=int(1e6)):
    """
    Generates initial conditions using a Gaussian Copula Mixture Model.

    The returned moments are absolute raw moments:

        M_alpha = n0 * (1/N) * sum_i prod_d x_i,d^alpha_d

    where n0 is the specified number density.

    Args:
        mixture_weights (list):
            Probability of each mixture mode. Must sum to 1.

        mode_configs (list of dicts):
            Configuration of each mixture mode.
            Each dict requires:
                - 'means': physical target means
                - 'stds': physical target standard deviations
                - 'correlations' (optional): correlation matrix

        dim_types (list of strings):
            Marginal distribution type for each coordinate.
            Supported:
                - 'lognormal'
                - 'beta'
                - 'normal'

        N_moments_shape (tuple):
            Shape of the required raw moment tensor.

        number_density (float):
            Absolute number density n0.
            The returned weights sum to this value.

        num_seeds (int):
            Number of Monte Carlo samples.

    Returns:
        tuple:
            (moments_tensor, nodes_array, weights_array)
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    num_modes = len(mixture_weights)
    D = len(dim_types)

    if len(mode_configs) != num_modes:
        raise ValueError("Number of mode configurations must match number of mixture weights.")

    if not np.isclose(np.sum(mixture_weights), 1.0):
        raise ValueError("Mixture weights must sum to one.")

    if number_density < 0:
        raise ValueError("Number density must be non-negative.")

    if len(N_moments_shape) != D:
        raise ValueError("Moment tensor dimension must match number of physical dimensions.")

    # ------------------------------------------------------------------
    # Generate samples according to mixture distribution
    # ------------------------------------------------------------------

    samples_per_mode = np.random.multinomial(num_seeds, mixture_weights)

    all_samples_list = []

    print(f"Generating Gaussian Copula Mixture ({num_modes} modes):")

    for m_idx in range(num_modes):

        count = samples_per_mode[m_idx]

        if count == 0:
            continue

        config = mode_configs[m_idx]

        phys_means = config["means"]
        phys_stds = config["stds"]

        corr_matrix = config.get("correlations", np.eye(D))

        if np.shape(corr_matrix) != (D, D):
            raise ValueError(f"Correlation matrix for mode {m_idx} has incorrect shape.")

        print(f"  - Mode {m_idx+1}: Generating {count} samples...")

        # --------------------------------------------------------------
        # Gaussian copula base
        # --------------------------------------------------------------

        Z_samples = np.random.multivariate_normal(np.zeros(D), corr_matrix, size=count)
        U_samples = norm.cdf(Z_samples)
        X_samples = np.zeros_like(U_samples)

        # --------------------------------------------------------------
        # Marginal transformations
        # --------------------------------------------------------------

        for d_idx in range(D):

            dtype = dim_types[d_idx]

            mu_target = phys_means[d_idx]
            std_target = phys_stds[d_idx]

            if dtype == "beta":
                alpha, beta_val = calculate_beta_params(mu_target, std_target)
                X_samples[:, d_idx] = beta.ppf(U_samples[:, d_idx], alpha, beta_val)
            elif dtype == "lognormal":
                shape_s, scale_val = (calculate_lognormal_underlying_params(mu_target, std_target))
                X_samples[:, d_idx] = lognorm.ppf(U_samples[:, d_idx], s=shape_s, scale=scale_val)
            elif dtype == "normal":
                # For Gaussian marginals, preserve the copula correlation
                # directly through the correlated Gaussian samples.
                X_samples[:, d_idx] = (mu_target + std_target * Z_samples[:, d_idx])
            else:
                raise ValueError(f"Unsupported distribution type: {dtype}")

        all_samples_list.append(X_samples)

    # ------------------------------------------------------------------
    # Combine samples
    # ------------------------------------------------------------------

    final_nodes = np.vstack(all_samples_list)

    # Absolute Monte Carlo weights
    mc_weights = np.full(len(final_nodes), number_density / len(final_nodes))

    # ------------------------------------------------------------------
    # Compute absolute raw moments
    # ------------------------------------------------------------------

    moments_tensor = np.zeros(N_moments_shape)

    ranges = [
        range(n)
        for n in N_moments_shape
    ]

    print("Computing Monte Carlo Moments...")

    for idx in itertools.product(*ranges):

        monomial = np.prod(
            [
                final_nodes[:, j] ** idx[j]
                for j in range(D)
            ],
            axis=0
        )

        moments_tensor[idx] = np.sum(mc_weights * monomial)

    print("Done.")

    return (moments_tensor, final_nodes, mc_weights)