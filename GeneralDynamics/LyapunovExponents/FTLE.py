import numpy as np
import math






import numpy as np


def compute_FTLE_2D_gridDomain(
    flow_map,
    final_time_index,
    time_array=None,
    initial_time_index=0,
    custom_time=False
):
    """
    Compute FTLE values on a 2D uniform grid directly from a 2D flow map.

    Parameters
    ----------
    flow_map : ndarray, shape (nx, ny, nt, 2)
        Flow map values.
        flow_map[i, j, t, 0] = x-position of particle (i,j) at time t
        flow_map[i, j, t, 1] = y-position of particle (i,j) at time t

    final_time_index : int
        Last time index up to which FTLE should be computed.

    time_array : array-like, optional
        Array of time values of length nt.
        If None and custom_time=False, the discrete indices [0,1,2,...,nt-1]
        are used as time values.

    initial_time_index : int, default=0
        Initial time index from which FTLE is measured.

    custom_time : bool, default=False
        If True, use the provided time_array.
        If False, use discrete integer time values.

    Returns
    -------
    FTLE : ndarray, shape (nx, ny, final_time_index+1)
        FTLE field for each time index from 0 to final_time_index.
        Boundary values are left as NaN because central differences are used.

    Notes
    -----
    The deformation gradient is computed as

        F = D Phi_t

    and the Cauchy-Green tensor is

        C = F^T F.

    Then FTLE is computed as

        FTLE = (1 / (2 |T|)) * log(lambda_max(C))

    where T = t_final - t_initial.
    """

    flow_map = np.asarray(flow_map)

    if flow_map.ndim != 4 or flow_map.shape[-1] != 2:
        raise ValueError(
            "flow_map must have shape (nx, ny, nt, 2), "
            "where the last axis stores (x,y)."
        )

    nx, ny, nt, dim = flow_map.shape

    if final_time_index >= nt:
        raise ValueError("final_time_index must be less than the number of time steps.")

    if initial_time_index < 0 or initial_time_index >= nt:
        raise ValueError("initial_time_index is out of bounds.")

    if final_time_index < initial_time_index:
        raise ValueError("final_time_index must be >= initial_time_index.")

    if custom_time:
        if time_array is None:
            raise ValueError("custom_time=True requires a valid time_array.")
        time_array = np.asarray(time_array)
        if len(time_array) != nt:
            raise ValueError("time_array must have length equal to the number of time steps.")
    else:
        time_array = np.arange(nt, dtype=float)

    # Initial grid spacing is taken from the initial time slice
    x0 = flow_map[:, :, initial_time_index, 0]
    y0 = flow_map[:, :, initial_time_index, 1]

    # Use the first interior spacings
    delta_x = x0[1, 0] - x0[0, 0]
    delta_y = y0[0, 1] - y0[0, 0]

    if delta_x == 0 or delta_y == 0:
        raise ValueError("Detected zero grid spacing in the initial grid.")

    delta_x = float(delta_x)
    delta_y = float(delta_y)

    FTLE = np.full((nx, ny, final_time_index + 1), np.nan, dtype=float)

    for t in range(0, final_time_index + 1):
        T = time_array[t] - time_array[0]

        # FTLE is undefined at zero integration time
        if T == 0:
            FTLE[:, :, t] = 0.0
            continue

        Xt = flow_map[:, :, t, 0]
        Yt = flow_map[:, :, t, 1]

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):

                local_values = np.array([
                    Xt[i - 1, j], Xt[i + 1, j], Xt[i, j - 1], Xt[i, j + 1],
                    Yt[i - 1, j], Yt[i + 1, j], Yt[i, j - 1], Yt[i, j + 1]
                ])

                if not np.all(np.isfinite(local_values)):
                    FTLE[i, j, t] = np.nan
                    continue

                # Deformation gradient F = D Phi_t
                dX_dx0 = (Xt[i + 1, j] - Xt[i - 1, j]) / (2.0 * delta_x)
                dX_dy0 = (Xt[i, j + 1] - Xt[i, j - 1]) / (2.0 * delta_y)
                dY_dx0 = (Yt[i + 1, j] - Yt[i - 1, j]) / (2.0 * delta_x)
                dY_dy0 = (Yt[i, j + 1] - Yt[i, j - 1]) / (2.0 * delta_y)

                F = np.array([
                    [dX_dx0, dX_dy0],
                    [dY_dx0, dY_dy0]
                ], dtype=float)

                C = F.T @ F

                # Since C should be symmetric positive semidefinite,
                # eigvalsh is the appropriate stable choice.
                eigvals = np.linalg.eigvalsh(C)
                lambda_max = np.max(eigvals)

                if lambda_max <= 0 or not np.isfinite(lambda_max):
                    FTLE[i, j, t] = np.nan
                    continue

                FTLE[i, j, t] = (1.0 / (2.0 * abs(T))) * np.log(lambda_max)

    return FTLE






def compute_FTLE_2D_sparse(flow_map, initial_time_index, final_time_index, time_array, custom_time=False):
    """
    compute FTLE values from a flow on a subset of R^2
    
    ---
    flow_map : nd_array [number_points, 2, time_steps]



    
    """


    if custom_time:
        time_array = time_array
    else:
        time_array = [i for i in range(0, len(flow_map[0,0, :]))] # generate discrete indexes of time values 0, 1, 3, to number of time steps in flow map




