import numpy as np
from scipy.spatial import cKDTree


def compute_FTLE_2D_gridDomain(flow_map, final_time_index):
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

    Returns
    -------
    FTLE : ndarray, shape (nx, ny, final_time_index+1)
        FTLE field.
        FTLE[:,:,0] = 0 by definition.
    """

    flow_map = np.asarray(flow_map)

    if flow_map.ndim != 4 or flow_map.shape[-1] != 2:
        raise ValueError(
            "flow_map must have shape (nx, ny, nt, 2)"
        )

    nx, ny, nt, _ = flow_map.shape

    if final_time_index >= nt:
        raise ValueError("final_time_index must be less than number of time steps.")

    # Initial grid (t = 0)
    x0 = flow_map[:, :, 0, 0]
    y0 = flow_map[:, :, 0, 1]

    delta_x = x0[1, 0] - x0[0, 0]
    delta_y = y0[0, 1] - y0[0, 0]

    if delta_x == 0 or delta_y == 0:
        raise ValueError("Zero grid spacing detected.")

    delta_x = float(delta_x)
    delta_y = float(delta_y)

    FTLE = np.full((nx, ny, final_time_index + 1), np.nan, dtype=float)

    # FTLE at t=0
    FTLE[:, :, 0] = 0.0

    for t in range(1, final_time_index + 1):

        T = float(t)

        Xt = flow_map[:, :, t, 0]
        Yt = flow_map[:, :, t, 1]

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):

                local_values = np.array([
                    Xt[i - 1, j], Xt[i + 1, j],
                    Xt[i, j - 1], Xt[i, j + 1],
                    Yt[i - 1, j], Yt[i + 1, j],
                    Yt[i, j - 1], Yt[i, j + 1]
                ])

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

                if lambda_max <= 0 or not np.isfinite(lambda_max):
                    continue

                FTLE[i, j, t] = (1.0 / (2.0 * T)) * np.log(lambda_max)

    return FTLE








def compute_FTLE_2D_sparse(flow_map, final_time_index, k_neighbors=10):
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

    Returns
    -------
    FTLE : ndarray, shape (n_points, final_time_index + 1)
        FTLE values for each point and time index.
        FTLE[:, 0] = 0 by convention.

    Notes
    -----
    For each point p and time t, we estimate F by solving

        ΔX_t ≈ F ΔX_0

    over the neighbors of p in the initial configuration.

    Then the Cauchy-Green tensor is

        C = F^T F

    and the FTLE is

        FTLE = (1 / (2 * t)) * log(lambda_max(C))

    since the time array is assumed to be [0, 1, 2, ...].
    """

    flow_map = np.asarray(flow_map, dtype=float)

    if flow_map.ndim != 3:
        raise ValueError("flow_map must have shape (n_points, 2, n_times).")

    n_points, dim, n_times = flow_map.shape

    if dim != 2:
        raise ValueError("flow_map must have shape (n_points, 2, n_times).")

    if final_time_index >= n_times:
        raise ValueError("final_time_index must be less than the number of time steps.")

    if k_neighbors < 3:
        raise ValueError("k_neighbors must be at least 3.")

    if k_neighbors >= n_points:
        raise ValueError("k_neighbors must be smaller than the number of points.")

    FTLE = np.full((n_points, final_time_index + 1), np.nan, dtype=float)
    FTLE[:, 0] = 0.0

    # Initial positions
    X0 = flow_map[:, :, 0]

    # Check validity of initial positions
    valid_initial = np.all(np.isfinite(X0), axis=1)

    # Build KDTree on valid initial points only
    valid_indices = np.where(valid_initial)[0]
    if len(valid_indices) < k_neighbors + 1:
        raise ValueError("Not enough valid initial points for the requested k_neighbors.")

    X0_valid = X0[valid_indices]
    tree = cKDTree(X0_valid)

    # For each valid point, precompute neighbors in the initial configuration
    neighbor_map = {}

    for local_idx, global_idx in enumerate(valid_indices):
        dists, nbrs_local = tree.query(X0[global_idx], k=k_neighbors + 1)

        # Remove self if present
        nbrs_local = np.atleast_1d(nbrs_local)
        nbrs_global = valid_indices[nbrs_local]
        nbrs_global = nbrs_global[nbrs_global != global_idx]

        # Keep exactly k_neighbors if possible
        nbrs_global = nbrs_global[:k_neighbors]

        neighbor_map[global_idx] = nbrs_global

    for t in range(1, final_time_index + 1):
        Xt = flow_map[:, :, t]

        for p in valid_indices:
            nbrs = neighbor_map[p]

            if len(nbrs) < 2:
                FTLE[p, t] = np.nan
                continue

            x0_p = X0[p]
            xt_p = Xt[p]

            if not np.all(np.isfinite(xt_p)):
                FTLE[p, t] = np.nan
                continue

            # Build neighbor offset matrices
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
                FTLE[p, t] = np.nan
                continue

            # Shape: (m, 2)
            A = np.asarray(dX0_list, dtype=float)
            B = np.asarray(dXt_list, dtype=float)

            # Need rank 2 in the initial neighbor offsets
            if np.linalg.matrix_rank(A) < 2:
                FTLE[p, t] = np.nan
                continue

            # Solve A @ M ≈ B  =>  M ≈ deformation map transpose
            # Then F = M^T
            M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            F = M.T

            C = F.T @ F
            eigvals = np.linalg.eigvalsh(C)
            lambda_max = np.max(eigvals)

            if not np.isfinite(lambda_max) or lambda_max <= 0:
                FTLE[p, t] = np.nan
                continue

            FTLE[p, t] = (1.0 / (2.0 * t)) * np.log(lambda_max)

    return FTLE


 


