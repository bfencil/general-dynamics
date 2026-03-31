import numpy as np



def compute_flow_map_RK4_2D(
    vector_field,
    x_grid=None,
    y_grid=None,
    time_array=None,
    constant_in_time=False,
    fill_value=np.nan
):
    """
    Compute a 2D flow map from a discrete vector field using RK4.

    Parameters
    ----------
    vector_field : ndarray
        If constant_in_time=False:
            shape (nx, ny, nt, 2)
        If constant_in_time=True:
            shape (nx, ny, 2)

        The last axis stores velocity components:
            vector_field[..., 0] = u(x,y,t)
            vector_field[..., 1] = v(x,y,t)

    x_grid : 1D ndarray or None
        x-coordinates of the grid points, length nx.
        If None, uses [0, 1, ..., nx-1].

    y_grid : 1D ndarray or None
        y-coordinates of the grid points, length ny.
        If None, uses [0, 1, ..., ny-1].

    time_array : 1D ndarray or None
        Time values of length nt.
        If None:
            - for time-varying field: uses [0, 1, ..., nt-1]
            - for constant field: a default 2-step time array [0, 1] is used

    constant_in_time : bool, default=False
        Whether the vector field is constant in time.

    fill_value : float, default=np.nan
        Value assigned when a particle leaves the domain or interpolation fails.

    Returns
    -------
    flow_map : ndarray, shape (nx, ny, nt, 2)
        Flow map from the initial grid positions.

    Notes
    -----
    This computes trajectories starting from each grid point at time_array[0]
    and stores the particle positions at each time in time_array.

    For particles that leave the grid domain, all later values are set to fill_value.
    """

    vector_field = np.asarray(vector_field, dtype=float)

    if constant_in_time:
        if vector_field.ndim != 3 or vector_field.shape[-1] != 2:
            raise ValueError(
                "For constant_in_time=True, vector_field must have shape (nx, ny, 2)."
            )
        nx, ny, _ = vector_field.shape

        if time_array is None:
            time_array = np.array([0.0, 1.0], dtype=float)
        else:
            time_array = np.asarray(time_array, dtype=float)

        nt = len(time_array)

    else:
        if vector_field.ndim != 4 or vector_field.shape[-1] != 2:
            raise ValueError(
                "For constant_in_time=False, vector_field must have shape (nx, ny, nt, 2)."
            )
        nx, ny, nt, _ = vector_field.shape

        if time_array is None:
            time_array = np.arange(nt, dtype=float)
        else:
            time_array = np.asarray(time_array, dtype=float)
            if len(time_array) != nt:
                raise ValueError("time_array must have length nt.")

    if x_grid is None:
        x_grid = np.arange(nx, dtype=float)
    else:
        x_grid = np.asarray(x_grid, dtype=float)

    if y_grid is None:
        y_grid = np.arange(ny, dtype=float)
    else:
        y_grid = np.asarray(y_grid, dtype=float)

    if len(x_grid) != nx:
        raise ValueError("x_grid must have length nx.")
    if len(y_grid) != ny:
        raise ValueError("y_grid must have length ny.")
    if len(time_array) < 2:
        raise ValueError("time_array must contain at least two time points.")

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    if np.any(np.abs(np.diff(x_grid) - dx) > 1e-12):
        raise ValueError("x_grid must be uniformly spaced.")
    if np.any(np.abs(np.diff(y_grid) - dy) > 1e-12):
        raise ValueError("y_grid must be uniformly spaced.")

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    t_min, t_max = time_array[0], time_array[-1]

    flow_map = np.full((nx, ny, len(time_array), 2), fill_value, dtype=float)

    X0, Y0 = np.meshgrid(x_grid, y_grid, indexing='ij')
    flow_map[:, :, 0, 0] = X0
    flow_map[:, :, 0, 1] = Y0

    def point_in_domain(x, y):
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def bilinear_interpolate_space(field2d, x, y):
        """
        field2d shape: (nx, ny)
        """
        if not point_in_domain(x, y):
            return np.nan

        # Handle right/top boundary exactly
        if x == x_max:
            i0 = nx - 2
            i1 = nx - 1
            tx = 1.0
        else:
            fx = (x - x_min) / dx
            i0 = int(np.floor(fx))
            i1 = i0 + 1
            tx = fx - i0

        if y == y_max:
            j0 = ny - 2
            j1 = ny - 1
            ty = 1.0
        else:
            fy = (y - y_min) / dy
            j0 = int(np.floor(fy))
            j1 = j0 + 1
            ty = fy - j0

        f00 = field2d[i0, j0]
        f10 = field2d[i1, j0]
        f01 = field2d[i0, j1]
        f11 = field2d[i1, j1]

        if not np.all(np.isfinite([f00, f10, f01, f11])):
            return np.nan

        return (
            (1 - tx) * (1 - ty) * f00
            + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01
            + tx * ty * f11
        )

    def interpolate_velocity_timevarying(x, y, t):
        if not point_in_domain(x, y):
            return np.array([np.nan, np.nan], dtype=float)

        if t < t_min or t > t_max:
            return np.array([np.nan, np.nan], dtype=float)

        # exact endpoint
        if t == t_max:
            k0 = nt - 2
            k1 = nt - 1
            tau = 1.0
        else:
            dt_ref = time_array[1] - time_array[0]
            if np.any(np.abs(np.diff(time_array) - dt_ref) > 1e-12):
                raise ValueError("time_array must be uniformly spaced for this implementation.")
            ft = (t - t_min) / dt_ref
            k0 = int(np.floor(ft))
            k1 = min(k0 + 1, nt - 1)
            tau = ft - k0

        u0 = bilinear_interpolate_space(vector_field[:, :, k0, 0], x, y)
        u1 = bilinear_interpolate_space(vector_field[:, :, k1, 0], x, y)
        v0 = bilinear_interpolate_space(vector_field[:, :, k0, 1], x, y)
        v1 = bilinear_interpolate_space(vector_field[:, :, k1, 1], x, y)

        if not np.all(np.isfinite([u0, u1, v0, v1])):
            return np.array([np.nan, np.nan], dtype=float)

        u = (1 - tau) * u0 + tau * u1
        v = (1 - tau) * v0 + tau * v1
        return np.array([u, v], dtype=float)

    def interpolate_velocity_constant(x, y):
        if not point_in_domain(x, y):
            return np.array([np.nan, np.nan], dtype=float)

        u = bilinear_interpolate_space(vector_field[:, :, 0], x, y)
        v = bilinear_interpolate_space(vector_field[:, :, 1], x, y)

        if not np.all(np.isfinite([u, v])):
            return np.array([np.nan, np.nan], dtype=float)

        return np.array([u, v], dtype=float)

    def velocity(x, y, t):
        if constant_in_time:
            return interpolate_velocity_constant(x, y)
        return interpolate_velocity_timevarying(x, y, t)

    def rk4_step(x, y, t, dt):
        p = np.array([x, y], dtype=float)

        k1 = velocity(p[0], p[1], t)
        if not np.all(np.isfinite(k1)):
            return np.array([fill_value, fill_value], dtype=float)

        k2 = velocity(*(p + 0.5 * dt * k1), t + 0.5 * dt)
        if not np.all(np.isfinite(k2)):
            return np.array([fill_value, fill_value], dtype=float)

        k3 = velocity(*(p + 0.5 * dt * k2), t + 0.5 * dt)
        if not np.all(np.isfinite(k3)):
            return np.array([fill_value, fill_value], dtype=float)

        k4 = velocity(*(p + dt * k3), t + dt)
        if not np.all(np.isfinite(k4)):
            return np.array([fill_value, fill_value], dtype=float)

        p_next = p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if not point_in_domain(p_next[0], p_next[1]):
            return np.array([fill_value, fill_value], dtype=float)

        return p_next

    for i in range(nx):
        for j in range(ny):
            x = flow_map[i, j, 0, 0]
            y = flow_map[i, j, 0, 1]

            if not np.all(np.isfinite([x, y])):
                continue

            for n in range(len(time_array) - 1):
                t_n = time_array[n]
                dt_n = time_array[n + 1] - time_array[n]

                p_next = rk4_step(x, y, t_n, dt_n)

                flow_map[i, j, n + 1, :] = p_next

                if not np.all(np.isfinite(p_next)):
                    break

                x, y = p_next

    return flow_map