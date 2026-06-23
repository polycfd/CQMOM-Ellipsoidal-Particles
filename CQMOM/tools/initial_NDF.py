import numpy as np
import itertools
from scipy.stats import norm, beta, lognorm

def MC_Gaussian_moments(mu, cov, N, num_seeds = int(1e6)):
    """
    Generate a dictionnary of bivariate Gaussian moments using Monte Carlo method.
    """
    
    nodes = np.random.multivariate_normal(mu, cov, size=num_seeds)

    # Compute weights from the bivariate PDF
    pdf_values = multivariate_normal.pdf(nodes, mean=mu, cov=cov)
    weights = pdf_values / np.sum(pdf_values)
    
    moments = np.zeros(N)
    
    for idx in itertools.product(*[range(n) for n in N]):
        moments[idx] = np.sum(weights * np.prod([nodes[:,i] ** idx[i] for i in range(len(idx))], axis = 0))
    return moments, nodes, weights

import numpy as np
import itertools
from scipy.stats import multivariate_normal

def calculate_beta_params(mu, std):
    """
    Converts desired Mean and Std Dev of a Beta distribution [0,1]
    into shape parameters alpha (a) and beta (b).
    """
    
    var = std**2
    
    # Constraint check: variance cannot exceed mu*(1-mu) for a Beta on [0,1]
    max_var = mu * (1.0 - mu)
    if var >= max_var * 0.999:
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

def generate_copula_mixture_conditions(
    mixture_weights,
    mode_configs,
    dim_types,
    N_moments_shape,
    num_seeds=int(1e6)
):
    """
    Generates initial conditions using a Gaussian Copula Mixture Model.
    Allows mixing Beta, Lognormal, and Normal marginals with bimodal behavior.

    Args:
        mixture_weights (list): Probabilities for each mode [p1, p2...]. Sum to 1.0.
        
        mode_configs (list of dicts): A list where each item defines a mode. Each dict needs:
            - 'means': List of PHYSICAL target means for [d, ar, u, v, w]
            - 'stds':  List of PHYSICAL target standard deviations
            - 'correlations' (optional): (D,D) Correlation matrix. Defaults to Identity.
            
        dim_types (list of strings): Defines distribution for each coordinate.
                                     Supported: 'lognormal', 'beta', 'normal'.
                                     e.g., ['lognormal', 'beta', 'normal', 'normal', 'normal']
        
        N_moments_shape (tuple): Shape of required moment tensor.
        num_seeds (int): Total MC samples.

    Returns:
        tuple: (moments_tensor, nodes_array, weights_array)
    """
    num_modes = len(mixture_weights)
    D = len(dim_types)
    
    # 1. Determine samples per mode
    samples_per_mode = np.random.multinomial(num_seeds, mixture_weights)
    all_samples_list = []
    
    print(f"Generating Copula Mixture ({num_modes} modes):")

    # --- LOOP OVER MODES ---
    for m_idx in range(num_modes):
        count = samples_per_mode[m_idx]
        if count == 0: continue
        
        config = mode_configs[m_idx]
        phys_means = config['means']
        phys_stds = config['stds']
        corr_matrix = config.get('correlations', np.eye(D))
        
        print(f"  - Mode {m_idx+1}: Generating {count} samples...")
        
        # A. Generate correlated Standard Normal samples (The "Copula" base)
        # Mean 0, covariance equals correlation matrix (std=1)
        Z_samples = np.random.multivariate_normal(np.zeros(D), corr_matrix, size=count)
        
        # B. Transform Z -> U (Uniform on [0,1])
        U_samples = norm.cdf(Z_samples)
        
        # C. Transform U -> X (Physical variables using Inverse CDFs)
        X_samples = np.zeros_like(U_samples)
        
        for d_idx in range(D):
            dtype = dim_types[d_idx]
            mu_target = phys_means[d_idx]
            std_target = phys_stds[d_idx]
            
            if dtype == 'beta':
                # Calculate alpha/beta from physical mu/std
                a, b = calculate_beta_params(mu_target, std_target)
                # Apply Inverse CDF
                X_samples[:, d_idx] = beta.ppf(U_samples[:, d_idx], a, b)
                
            elif dtype == 'lognormal':
                # Calculate underlying shape/scale from physical mu/std
                shape_s, scale_val = calculate_lognormal_underlying_params(mu_target, std_target)
                # Apply Inverse CDF
                X_samples[:, d_idx] = lognorm.ppf(U_samples[:, d_idx], s=shape_s, scale=scale_val)
                
            elif dtype == 'normal':
                # Apply Inverse CDF (which is just shifting and scaling Z)
                # X = mu + std * Z
                X_samples[:, d_idx] = norm.ppf(U_samples[:, d_idx], loc=mu_target, scale=std_target)
            
            else:
                raise ValueError(f"Unsupported distribution type: {dtype}")
                
        all_samples_list.append(X_samples)
        
    # 3. Combine populations
    final_nodes = np.vstack(all_samples_list)
    mc_weights = np.full(num_seeds, 1.0 / num_seeds)

    # 4. Compute Exact Sample Moments
    moments_tensor = np.zeros(N_moments_shape)
    ranges = [range(n) for n in N_moments_shape]
    
    print("Computing Monte Carlo Moments...")
    for idx in itertools.product(*ranges):
        mvals = np.prod([final_nodes[:, j] ** idx[j] for j in range(D)], axis=0)
        moments_tensor[idx] = np.sum(mc_weights * mvals)

    print("Done.")
    return moments_tensor, final_nodes, mc_weights  