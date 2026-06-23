import numpy as np

def wheeler(moments, n, adaptive=False, rmin=1e-10, eabs=1e-10, cutoff=1e-30):
    """
    Inverts moments into 1D quadrature weights and abscissas using the adaptive Wheeler algorithm.

    The function calculates quadrature nodes and weights by inverting the provided statistical moments of a
    probability density function (PDF) using an adaptive Wheeler approach. This method is used to find the
    nodes and weights that are consistent with the moments of the distribution.

    :param moments: Statistical moments of the transported PDF.
    :type moments: array-like
    :param n: Maximum number of nodes (abscissas) for the quadrature.
    :type n: int
    :param adaptive: Flag to indicate whether to use the adaptive Wheeler algorithm.
    :type adaptive: bool, optional
    :param rmin: Minimum ratio of the weights for the nodes.
    :type rmin: float, optional
    :param eabs: Minimum absolute distance between distinct abscissas (nodes).
    :type eabs: float, optional
    :param cutoff: Minimum value for the weights.
    :type cutoff: float, optional
    :return: A tuple of (abscissas, weights, werror flag).
    :rtype: tuple of (ndarray, ndarray, int)
    """
    werror = 0

    if moments[0] < 0:
        print("Wheeler: Moments are NOT realizable (moment[0] < 0.0). Run failed.")
        werror = 1
        return np.array([moments[1] / moments[0]]), np.array([np.abs(moments[0])]), werror

    if moments[0] < cutoff:
        print("Wheeler: Moments are NOT realizable (moment[0] = 0.0). Run failed.")
        werror = 1
        return np.array([0]), np.array([1.0]), werror

    if n == 1 or (adaptive and moments[0] <= rmin):
        return np.array([moments[1] / moments[0]]), np.array([moments[0]]), werror

    nu = moments.copy()
    ind = n

    a = np.zeros(ind)
    b = np.zeros(ind)
    sigma = np.zeros((2 * ind + 1, 2 * ind + 1))

    for i in range(1, 2 * ind + 1):
        sigma[1, i] = nu[i - 1]

    a[0] = nu[1] / nu[0]
    b[0] = 0

    for k in range(2, n + 1):
        for l in range(k, 2 * ind - k + 2):
            sigma[k, l] = (
                sigma[k - 1, l + 1]
                - a[k - 2] * sigma[k - 1, l]
                - b[k - 2] * sigma[k - 2, l]
            )
        a[k - 1] = sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
        b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    if adaptive:
        for k in range(ind, 1, -1):
            if sigma[k, k] < cutoff:
                n = k - 1
                if n == 1:
                    return np.array([moments[1] / moments[0]]), np.array([moments[0]]), werror

        a = np.zeros(n)
        b = np.zeros(n)
        sigma = np.zeros((2 * n + 1, 2 * n + 1))

        for i in range(1, 2 * n + 1):
            sigma[1, i] = nu[i - 1]

        a[0] = nu[1] / nu[0]
        b[0] = 0
        for k in range(2, n + 1):
            for l in range(k, 2 * n - k + 2):
                sigma[k, l] = (
                    sigma[k - 1, l + 1]
                    - a[k - 2] * sigma[k - 1, l]
                    - b[k - 2] * sigma[k - 2, l]
                )
            a[k - 1] = (
                sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
            )
            b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    if b.min() < 0:
        print("Moments in Wheeler_moments are not realizable! b.min()<0.")
        werror = 1
        return np.array([moments[1] / moments[0]]), np.array([moments[0]]), werror

    for n1 in range(n, 0, -1):
        if n1 == 1:
            return np.array([moments[1] / moments[0]]), np.array([moments[0]]), werror

        sqrt_b = np.sqrt(b[1:n1])
        jacobi = np.diag(a[:n1]) + np.diag(sqrt_b, -1) + np.diag(sqrt_b, 1)

        eigenvalues, eigenvectors = np.linalg.eig(jacobi)
        idx = eigenvalues.argsort()
        x = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        w = moments[0] * eigenvectors[0, :] ** 2

        if adaptive:
            dab = np.zeros(n1)
            mab = np.zeros(n1)
            for i in range(n1 - 1, 0, -1):
                dab[i] = min(abs(x[i] - x[0:i]))
                mab[i] = max(abs(x[i] - x[0:i]))
            mindab = min(dab[1:n1])
            maxmab = max(mab[1:n1])
            if n1 == 2:
                maxmab = 1
            if min(w) / max(w) > rmin and mindab / maxmab > eabs:
                return np.array(x), np.array(w), werror
        else:
            return np.array(x), np.array(w), werror

def compute_conditional_moments(xi, w, moments_slice, n_nodes, rcond):
    """
    Compute conditional moments for the next dimension given the current nodes and weights.

    Uses the Vandermonde-based inversion: c_m = pinv(V @ diag(w)) @ moments_slice,
    where V is the Vandermonde matrix built from xi with shape (n_nodes, n_nodes).

    :param xi: Quadrature nodes for the current dimension, shape (n_nodes,).
    :param w: Quadrature weights for the current dimension, shape (n_nodes,).
    :param moments_slice: Moment array slice with first axis of length >= n_nodes.
    :param n_nodes: Number of nodes (rank of the Vandermonde system).
    :param rcond: Relative condition number cutoff for the pseudoinverse.
    :return: Conditional moments array with the same shape as moments_slice but
             first axis replaced by n_nodes.
    """
    V = np.vander(xi, n_nodes, increasing=True).T  # (n_nodes, n_nodes)
    VR = V @ np.diag(w)                             # (n_nodes, n_nodes)
    inv_VR = np.linalg.pinv(VR, rcond=rcond)        # (n_nodes, n_nodes)

    trailing_shape = moments_slice.shape[1:]
    c_m = np.zeros((n_nodes,) + trailing_shape)
    for idx in np.ndindex(trailing_shape):
        c_m[(slice(None),) + idx] = inv_VR @ moments_slice[(slice(None, n_nodes),) + idx]
    return c_m

def cqmom(N, m, adaptive=False, rmin=None, eabs=None, cutoff=1e-2, rcond=1e-30):
    """
    Compute a multivariate quadrature approximation using CQMOM for an arbitrary number of dimensions.

    This is a general implementation of the Conditional Quadrature Method of Moments (CQMOM).
    It supports any number of internal coordinates by applying the Wheeler algorithm recursively,
    dimension by dimension, using conditional moment inversion at each level.

    Parameters
    ----------
    N : tuple of int
        Number of quadrature nodes in each dimension, e.g. (N1, N2, ..., Nd).
    m : ndarray
        Moment array. The shape must be (2*N1, 2*N2, ..., 2*Nd) or larger along each axis.
        Axis ``k`` corresponds to the moment order in dimension ``k+1``.
    adaptive : bool, optional
        Whether to use the adaptive node reduction in the Wheeler algorithm. Default is False.
    rmin : list of float, optional
        Minimum weight ratio threshold per dimension (length must equal len(N)).
        Defaults to [1e-2] * len(N).
    eabs : list of float, optional
        Minimum relative node separation per dimension (length must equal len(N)).
        Defaults to [1e-8] * len(N).
    cutoff : float, optional
        Minimum weight threshold (relative to zeroth moment) for a node to be considered
        valid in the first dimension. Default is 1e-2.
    rcond : float, optional
        Relative condition number cutoff for pseudoinverse calculations. Default is 1e-30.

    Returns
    -------
    w : ndarray, shape (total_nodes,)
        Quadrature weights.
    xi : ndarray, shape (d, total_nodes)
        Quadrature nodes, one row per dimension.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Compute 1D quadrature (xi1, w1) from the marginal moments m[:, 0, 0, ...].
    2. For each node xi1[i], invert the Vandermonde system to obtain conditional moments
       in dimension 2, then compute (xi2, w2) via Wheeler.
    3. Repeat recursively for each subsequent dimension, conditioning on all previous nodes.
    4. Combine all node tuples and multiply the corresponding per-dimension weights.
    """
    d = len(N)
    if rmin is None:
        rmin = [1e-2] * d
    if eabs is None:
        eabs = [1e-8] * d

    # --- Step 1: First dimension ---
    marginal = m[tuple([slice(None)] + [0] * (d - 1))]
    xi1, w1, werror = wheeler(marginal, N[0], adaptive, rmin[0], eabs[0])

    if werror > 0:
        print("1D quadrature failed on step 1!")
        total = int(np.prod(N))
        return np.zeros(total), np.zeros((d, total))

    N1_actual = len(w1)
    m0 = m[tuple([0] * d)]
    for i in range(N1_actual):
        if abs(w1[i]) / m0 < cutoff:
            print("One of the weights is null! Reduce the number of nodes in direction 1.")
            total = int(np.prod(N))
            return np.zeros(total), np.zeros((d, total))

    # --- Recursive conditional moment inversion ---
    # Each element of `nodes` is a dict with keys:
    #   'xi'      : tuple of scalar node coordinates (one per dimension processed so far)
    #   'w'       : product weight accumulated so far
    #   'c_m'     : conditional moment array for the remaining dimensions
    #   'dim_idx' : which dimension we are about to process next (0-based)
    nodes = []
    c_m_1 = compute_conditional_moments(xi1, w1, m, N1_actual, rcond)

    for i in range(N1_actual):
        nodes.append({
            'xi': (xi1[i],),
            'w': w1[i],
            'c_m': c_m_1[i],   # shape: (2*N2, 2*N3, ...) for the remaining dims
            'dim_idx': 1,
        })

    # --- Steps 2 .. d: process each remaining dimension ---
    for dim in range(1, d):
        next_nodes = []
        n_dim = N[dim]

        for node in nodes:
            c_m_parent = node['c_m']  # shape: (2*n_dim, 2*N_{dim+1}, ...)

            # Extract 1D moments for this dimension: index 0 in all trailing axes
            moments_1d = c_m_parent[tuple([slice(None)] + [0] * (d - dim - 1))]

            try:
                xi_new, w_new, werror = wheeler(
                    moments_1d, n_dim, adaptive, rmin[dim], eabs[dim]
                )
            except Exception as ex:
                print(f"Exception in Wheeler at dimension {dim + 1}: {ex}")
                xi_new, w_new, werror = np.array([]), np.array([]), 1

            if werror > 0:
                print(f"1D quadrature failed on step {dim + 1}!")
                total = int(np.prod(N))
                return np.zeros(total), np.zeros((d, total))

            if len(xi_new) == 0:
                print(f"Empty quadrature result at dimension {dim + 1}.")
                total = int(np.prod(N))
                return np.zeros(total), np.zeros((d, total))

            # If this is not the last dimension, compute conditional moments for dim+1
            if dim < d - 1:
                n_new = len(xi_new)
                c_m_next = compute_conditional_moments(xi_new, w_new, c_m_parent, n_new, rcond)
            else:
                c_m_next = [None] * len(xi_new)

            for k, (xi_k, w_k) in enumerate(zip(xi_new, w_new)):
                next_nodes.append({
                    'xi': node['xi'] + (xi_k,),
                    'w': node['w'] * w_k,
                    'c_m': c_m_next[k],
                    'dim_idx': dim + 1,
                })

        nodes = next_nodes

    # --- Assemble output ---
    if len(nodes) == 0:
        return np.zeros(0), np.zeros((d, 0))

    w_out = np.array([node['w'] for node in nodes])
    xi_out = np.array([node['xi'] for node in nodes]).T  # (d, total_nodes)

    return w_out, xi_out