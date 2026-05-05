import numpy as np

def zeroth_order_local_flux(t, nodes, weights, k, function = lambda t,xi,alpha: 0):
    """
    Calculate the multivariate general zeroth order point process local flux for a given set of nodes and weights.
    Useful for: nucleation, growth, and diffusion processes.
    
    
    Parameters:
    nodes (numpy.ndarray): Array of nodes in phase space. (xi1, xi2, ...)
    weights (numpy.ndarray): Array of weights corresponding to the nodes. (w1, w2, ...)
    k (numpy.ndarray): order of the moment. (k1, k2, ...)
    function (callable): Function to compute the local flux.
    
            Defaults to a function that returns 0.
            
            An example of a function could be:
            def function(xi, k):
                return k[0] * xi[0]**(k[0]-1) * xi[1]**k[1] * ...
            
    Returns:
    numpy.ndarray: The calculated local flux for the order k.
    """
    # Ensure the input arrays are numpy arrays
    nodes = np.asarray(nodes)
    weights = np.asarray(weights)

    local_flux = 0
    if weights.ndim == 0:
        weights = np.array([weights])
        nodes = np.array([nodes])
        
    for i in range(len(weights)):
        if len(nodes.shape) == 1:
            local_flux += weights[i] * function(t,nodes[i],k)
        else:
            local_flux += weights[i] * function(t,nodes[:,i],k)
        
    return local_flux
