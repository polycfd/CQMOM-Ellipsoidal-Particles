import numpy as np

def ssp_rk3(state, time_step, t, momidx, compute_flux, adaptive=False):
    """
    Advances the state with a Runge-Kutta 2/3 SSP scheme using compute_flux.

    Parameters:
    - state: numpy array, the current solution moments
    - time_step: float, timestep size
    - t: float, current time
    - momidx: index or identifiers for the moments (depends on your system)
    - compute_flux: function handle, should be compute_flux(t, moments, momidx)
                    and return the RHS (same shape as state)
    - adaptive: bool, whether to compute adaptive timestep error

    Returns:
    - new_state: updated state after RK3 step
    - ts_error: timestep error estimate (if adaptive), else None
    """

    # Allocate stage arrays
    stage_state = [state.copy() for _ in range(3)]
    stage_k = [np.zeros_like(state) for _ in range(3)]

    # Stage 1
    stage_k[0] = compute_flux(t, stage_state[0], momidx)
    stage_state[1] = stage_state[0] + time_step * stage_k[0]

    # Stage 2
    stage_k[1] = compute_flux(t+time_step, stage_state[1], momidx)
    test_state = 0.5 * (stage_state[0] + (stage_state[1] + time_step * stage_k[1]))

    # Stage 3
    stage_state[2] = 0.75 * stage_state[0] + 0.25 * (stage_state[1] + time_step * stage_k[1])
    stage_k[2] = compute_flux(t+0.5*time_step, stage_state[2], momidx)

    # Final update
    new_state = (stage_state[0] + 2.0 * (stage_state[2] + time_step * stage_k[2])) / 3.0

    # Adaptive error estimate
    ts_error = None
    if adaptive:
        ts_error = np.linalg.norm(new_state - test_state) / np.linalg.norm(new_state)

    return new_state, ts_error


def adapt_time_step(current_dt, ts_error, error_tol, min_dt, max_dt):
    """
    Adapt the time step based on timestep error and user-specified tolerance.

    Parameters:
    - current_dt: float, current time step
    - ts_error: float, timestep error from last update
    - error_tol: float, target tolerance for error
    - min_dt: float, minimum allowed timestep
    - max_dt: float, maximum allowed timestep

    Returns:
    - new_dt: float, updated (adapted) timestep
    """
    
    error_fraction = np.sqrt(0.5 * error_tol / ts_error)
    time_step_factor = min(max(error_fraction, 0.3), 2.0)
    new_dt = time_step_factor * current_dt
    new_dt = min(max(0.9 * new_dt, min_dt), max_dt)

    return new_dt
