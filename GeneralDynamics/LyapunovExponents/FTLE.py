import numpy as np
from scipy.spatial import cKDTree


import numpy as np





def compute_FTLE_2D_gridDomain(flow_map, final_time_index, time_array=None):
    """
    Compute FTLE values on a 2D uniform grid from a flow map.

    Parameters
    ----------
    flow_map : ndarray, shape (nx, ny, nt, 2)
        Flow map values.
        flow_map[i, j, t, 0] = x-position at time t
        flow_map[i, j, t, 1] = y-position at time t

    final_time_index : int
        Last time index to compute FTLE for.

    time_array : array-like or None, optional
        Time values associated with the third axis of flow_map.
        If None, the default discrete time array [0, 1, 2, ..., nt-1] is used.

    Returns
    -------
    FTLE : ndarray, shape (nx, ny, final_time_index+1)
        FTLE field.
        FTLE[:, :, 0] = 0 by definition.

    Notes
    -----
    The initial time is always taken to be index 0.

    The deformation gradient is computed as

        F = D Phi_t

    and the Cauchy-Green tensor is

        C = F^T F.

    Then FTLE is computed as

        FTLE = (1 / (2 |T|)) * log(lambda_max(C))

    where

        T = time_array[t] - time_array[0].
    """

    flow_map = np.asarray(flow_map, dtype=float)

    if flow_map.ndim != 4 or flow_map.shape[-1] != 2:
        raise ValueError("flow_map must have shape (nx, ny, nt, 2)")

    nx, ny, nt, _ = flow_map.shape

    if final_time_index < 0 or final_time_index >= nt:
        raise ValueError("final_time_index must satisfy 0 <= final_time_index < nt.")

    if time_array is None:
        time_array = np.arange(nt, dtype=float)
    else:
        time_array = np.asarray(time_array, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time_array must be one-dimensional.")
        if len(time_array) != nt:
            raise ValueError("time_array must have length equal to the number of time steps.")
        if not np.all(np.isfinite(time_array)):
            raise ValueError("time_array must contain only finite values.")

    # Initial grid (always t = 0)
    x0 = flow_map[:, :, 0, 0]
    y0 = flow_map[:, :, 0, 1]

    if nx < 2 or ny < 2:
        raise ValueError("flow_map must have at least 2 grid points in each spatial direction.")

    delta_x = x0[1, 0] - x0[0, 0]
    delta_y = y0[0, 1] - y0[0, 0]

    if not np.isfinite(delta_x) or not np.isfinite(delta_y):
        raise ValueError("Initial grid spacing is not finite.")

    if delta_x == 0 or delta_y == 0:
        raise ValueError("Zero grid spacing detected.")

    delta_x = float(delta_x)
    delta_y = float(delta_y)

    FTLE = np.full((nx, ny, final_time_index + 1), np.nan, dtype=float)
    FTLE[:, :, 0] = 0.0

    for t in range(1, final_time_index + 1):
        T = float(time_array[t] - time_array[0])

        if T == 0:
            FTLE[:, :, t] = np.nan
            continue

        Xt = flow_map[:, :, t, 0]
        Yt = flow_map[:, :, t, 1]

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                local_values = np.array([
                    Xt[i - 1, j], Xt[i + 1, j],
                    Xt[i, j - 1], Xt[i, j + 1],
                    Yt[i - 1, j], Yt[i + 1, j],
                    Yt[i, j - 1], Yt[i, j + 1]
                ], dtype=float)

                if not np.all(np.isfinite(local_values)):
                    continue

                # Deformation gradient
                dX_dx0 = (Xt[i + 1, j] - Xt[i - 1, j]) / (2.0 * delta_x)
                dX_dy0 = (Xt[i, j + 1] - Xt[i, j - 1]) / (2.0 * delta_y)
                dY_dx0 = (Yt[i + 1, j] - Yt[i - 1, j]) / (2.0 * delta_x)
                dY_dy0 = (Yt[i, j + 1] - Yt[i, j - 1]) / (2.0 * delta_y)

                F = np.array([
                    [dX_dx0, dX_dy0],
                    [dY_dx0, dY_dy0]
                ], dtype=float)

                C = F.T @ F
                eigvals = np.linalg.eigvalsh(C)
                lambda_max = np.max(eigvals)

                if not np.isfinite(lambda_max) or lambda_max <= 0:
                    continue

                FTLE[i, j, t] = (1.0 / (2.0 * abs(T))) * np.log(lambda_max)

    return FTLE






def compute_FTLE_2D_sparse(flow_map, final_time_index, k_neighbors=10, time_array=None):
    """
    Compute 2D FTLE values from a sparse flow map using local least-squares
    estimation of the deformation gradient.

    Parameters
    ----------
    flow_map : ndarray, shape (n_points, 2, n_times)
        Sparse flow map data.

        flow_map[p, 0, t] = x-coordinate of point p at time t
        flow_map[p, 1, t] = y-coordinate of point p at time t

    final_time_index : int
        Last time index up to which FTLE is computed.

    k_neighbors : int, default=10
        Number of nearest neighbors (including the point itself from KDTree query,
        which is discarded internally) used to estimate the local deformation
        gradient.

    time_array : array-like or None, optional
        Time values associated with the third axis of flow_map.
        If None, the default discrete time array [0, 1, 2, ..., n_times-1] is used.

    Returns
    -------
    FTLE : ndarray, shape (n_points, final_time_index + 1)
        FTLE values for each point and time index.
        FTLE[:, 0] = 0 by convention.

    Notes
    -----
    The initial time is always taken to be index 0.

    For each point p and time t, we estimate F by solving

        ΔX_t ≈ F ΔX_0

    over the neighbors of p in the initial configuration.

    Then the Cauchy-Green tensor is

        C = F^T F

    and the FTLE is

        FTLE = (1 / (2 * |T|)) * log(lambda_max(C))

    where

        T = time_array[t] - time_array[0].
    """

    flow_map = np.asarray(flow_map, dtype=float)

    if flow_map.ndim != 3:
        raise ValueError("flow_map must have shape (n_points, 2, n_times).")

    n_points, dim, n_times = flow_map.shape

    if dim != 2:
        raise ValueError("flow_map must have shape (n_points, 2, n_times).")

    if final_time_index < 0 or final_time_index >= n_times:
        raise ValueError("final_time_index must satisfy 0 <= final_time_index < n_times.")

    if k_neighbors < 3:
        raise ValueError("k_neighbors must be at least 3.")

    if k_neighbors >= n_points:
        raise ValueError("k_neighbors must be smaller than the number of points.")

    if time_array is None:
        time_array = np.arange(n_times, dtype=float)
    else:
        time_array = np.asarray(time_array, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time_array must be one-dimensional.")
        if len(time_array) != n_times:
            raise ValueError("time_array must have length equal to the number of time steps.")
        if not np.all(np.isfinite(time_array)):
            raise ValueError("time_array must contain only finite values.")

    FTLE = np.full((n_points, final_time_index + 1), np.nan, dtype=float)
    FTLE[:, 0] = 0.0

    # Initial positions at t = 0
    X0 = flow_map[:, :, 0]

    # Check validity of initial positions
    valid_initial = np.all(np.isfinite(X0), axis=1)
    valid_indices = np.where(valid_initial)[0]

    if len(valid_indices) < k_neighbors + 1:
        raise ValueError("Not enough valid initial points for the requested k_neighbors.")

    # KDTree on valid initial points
    X0_valid = X0[valid_indices]
    tree = cKDTree(X0_valid)

    # Precompute neighbors in the initial configuration
    neighbor_map = {}

    for global_idx in valid_indices:
        _, nbrs_local = tree.query(X0[global_idx], k=k_neighbors + 1)

        nbrs_local = np.atleast_1d(nbrs_local)
        nbrs_global = valid_indices[nbrs_local]

        # Remove self
        nbrs_global = nbrs_global[nbrs_global != global_idx]

        # Keep requested number of neighbors
        nbrs_global = nbrs_global[:k_neighbors]

        neighbor_map[global_idx] = nbrs_global

    for t in range(1, final_time_index + 1):
        T = float(time_array[t] - time_array[0])

        if T == 0:
            FTLE[:, t] = np.nan
            continue

        Xt = flow_map[:, :, t]

        for p in valid_indices:
            nbrs = neighbor_map[p]

            if len(nbrs) < 2:
                continue

            x0_p = X0[p]
            xt_p = Xt[p]

            if not np.all(np.isfinite(xt_p)):
                continue

            dX0_list = []
            dXt_list = []

            for q in nbrs:
                x0_q = X0[q]
                xt_q = Xt[q]

                if not np.all(np.isfinite(x0_q)) or not np.all(np.isfinite(xt_q)):
                    continue

                dX0 = x0_q - x0_p
                dXt = xt_q - xt_p

                if np.allclose(dX0, 0.0):
                    continue

                dX0_list.append(dX0)
                dXt_list.append(dXt)

            if len(dX0_list) < 2:
                continue

            A = np.asarray(dX0_list, dtype=float)   # shape (m, 2)
            B = np.asarray(dXt_list, dtype=float)   # shape (m, 2)

            # Need a genuinely 2D local neighborhood in the initial configuration
            if np.linalg.matrix_rank(A) < 2:
                continue

            # Solve A @ M ≈ B, then F = M^T
            M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            F = M.T

            C = F.T @ F
            eigvals = np.linalg.eigvalsh(C)
            lambda_max = np.max(eigvals)

            if not np.isfinite(lambda_max) or lambda_max <= 0:
                continue

            FTLE[p, t] = (1.0 / (2.0 * abs(T))) * np.log(lambda_max)

    return FTLE


