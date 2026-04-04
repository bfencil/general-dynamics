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

    if nx < 3 or ny < 3:
        raise ValueError("flow_map must have at least 3 grid points in each spatial direction.")

    # Initial grid spacing from t = 0
    x0 = flow_map[:, :, 0, 0]
    y0 = flow_map[:, :, 0, 1]

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

    # Interior slice
    interior = (slice(1, -1), slice(1, -1))

    for t in range(1, final_time_index + 1):
        T = float(time_array[t] - time_array[0])

        if T == 0:
            continue

        Xt = flow_map[:, :, t, 0]
        Yt = flow_map[:, :, t, 1]

        # Central differences on the interior
        dX_dx0 = (Xt[2:, 1:-1] - Xt[:-2, 1:-1]) / (2.0 * delta_x)
        dX_dy0 = (Xt[1:-1, 2:] - Xt[1:-1, :-2]) / (2.0 * delta_y)
        dY_dx0 = (Yt[2:, 1:-1] - Yt[:-2, 1:-1]) / (2.0 * delta_x)
        dY_dy0 = (Yt[1:-1, 2:] - Yt[1:-1, :-2]) / (2.0 * delta_y)

        # Validity mask: only compute where all needed values are finite
        valid = (
            np.isfinite(Xt[2:, 1:-1]) &
            np.isfinite(Xt[:-2, 1:-1]) &
            np.isfinite(Xt[1:-1, 2:]) &
            np.isfinite(Xt[1:-1, :-2]) &
            np.isfinite(Yt[2:, 1:-1]) &
            np.isfinite(Yt[:-2, 1:-1]) &
            np.isfinite(Yt[1:-1, 2:]) &
            np.isfinite(Yt[1:-1, :-2])
        )

        # Entries of C = F^T F, where
        # F = [[a, b],
        #      [c, d]]
        a = dX_dx0
        b = dX_dy0
        c = dY_dx0
        d = dY_dy0

        C11 = a * a + c * c
        C12 = a * b + c * d
        C22 = b * b + d * d

        # Largest eigenvalue of symmetric 2x2 matrix:
        # lambda_max = 0.5 * (trace + sqrt((C11-C22)^2 + 4*C12^2))
        trace = C11 + C22
        disc = (C11 - C22) ** 2 + 4.0 * (C12 ** 2)
        disc = np.maximum(disc, 0.0)  # numerical safety

        lambda_max = 0.5 * (trace + np.sqrt(disc))

        valid &= np.isfinite(lambda_max) & (lambda_max > 0.0)

        ftle_slice = np.full((nx - 2, ny - 2), np.nan, dtype=float)
        ftle_slice[valid] = (1.0 / (2.0 * abs(T))) * np.log(lambda_max[valid])

        FTLE[interior[0], interior[1], t] = ftle_slice

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


