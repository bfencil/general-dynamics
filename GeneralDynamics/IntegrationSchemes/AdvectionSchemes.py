import numpy as np
from numba import njit


@njit
def _point_in_domain(x, y, x_min, x_max, y_min, y_max):
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


@njit
def _bilinear_interpolate_space(field2d, x, y, x_min, x_max, y_min, y_max, dx, dy):
    nx, ny = field2d.shape

    if not _point_in_domain(x, y, x_min, x_max, y_min, y_max):
        return np.nan

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

    if not (np.isfinite(f00) and np.isfinite(f10) and np.isfinite(f01) and np.isfinite(f11)):
        return np.nan

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


@njit
def _interpolate_velocity_constant(vector_field, x, y, x_min, x_max, y_min, y_max, dx, dy):
    if not _point_in_domain(x, y, x_min, x_max, y_min, y_max):
        return np.nan, np.nan

    u = _bilinear_interpolate_space(
        vector_field[:, :, 0], x, y, x_min, x_max, y_min, y_max, dx, dy
    )
    v = _bilinear_interpolate_space(
        vector_field[:, :, 1], x, y, x_min, x_max, y_min, y_max, dx, dy
    )

    if not (np.isfinite(u) and np.isfinite(v)):
        return np.nan, np.nan

    return u, v


@njit
def _interpolate_velocity_timevarying(
    vector_field, x, y, t, time_array,
    x_min, x_max, y_min, y_max, dx, dy
):
    nx, ny, nt, _ = vector_field.shape

    if not _point_in_domain(x, y, x_min, x_max, y_min, y_max):
        return np.nan, np.nan

    t_min = time_array[0]
    t_max = time_array[-1]

    if t < t_min or t > t_max:
        return np.nan, np.nan

    dt_ref = time_array[1] - time_array[0]

    if t == t_max:
        k0 = nt - 2
        k1 = nt - 1
        tau = 1.0
    else:
        ft = (t - t_min) / dt_ref
        k0 = int(np.floor(ft))
        if k0 < 0:
            k0 = 0
        if k0 > nt - 2:
            k0 = nt - 2
        k1 = k0 + 1
        tau = ft - k0

    u0 = _bilinear_interpolate_space(
        vector_field[:, :, k0, 0], x, y, x_min, x_max, y_min, y_max, dx, dy
    )
    u1 = _bilinear_interpolate_space(
        vector_field[:, :, k1, 0], x, y, x_min, x_max, y_min, y_max, dx, dy
    )
    v0 = _bilinear_interpolate_space(
        vector_field[:, :, k0, 1], x, y, x_min, x_max, y_min, y_max, dx, dy
    )
    v1 = _bilinear_interpolate_space(
        vector_field[:, :, k1, 1], x, y, x_min, x_max, y_min, y_max, dx, dy
    )

    if not (np.isfinite(u0) and np.isfinite(u1) and np.isfinite(v0) and np.isfinite(v1)):
        return np.nan, np.nan

    u = (1.0 - tau) * u0 + tau * u1
    v = (1.0 - tau) * v0 + tau * v1
    return u, v


@njit
def _rk4_step_constant(
    vector_field, x, y, t, dt,
    x_min, x_max, y_min, y_max, dx, dy, fill_value
):
    k1x, k1y = _interpolate_velocity_constant(
        vector_field, x, y, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k1x) and np.isfinite(k1y)):
        return fill_value, fill_value

    k2x, k2y = _interpolate_velocity_constant(
        vector_field,
        x + 0.5 * dt * k1x,
        y + 0.5 * dt * k1y,
        x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k2x) and np.isfinite(k2y)):
        return fill_value, fill_value

    k3x, k3y = _interpolate_velocity_constant(
        vector_field,
        x + 0.5 * dt * k2x,
        y + 0.5 * dt * k2y,
        x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k3x) and np.isfinite(k3y)):
        return fill_value, fill_value

    k4x, k4y = _interpolate_velocity_constant(
        vector_field,
        x + dt * k3x,
        y + dt * k3y,
        x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k4x) and np.isfinite(k4y)):
        return fill_value, fill_value

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    if not _point_in_domain(x_next, y_next, x_min, x_max, y_min, y_max):
        return fill_value, fill_value

    return x_next, y_next


@njit
def _rk4_step_timevarying(
    vector_field, x, y, t, dt, time_array,
    x_min, x_max, y_min, y_max, dx, dy, fill_value
):
    k1x, k1y = _interpolate_velocity_timevarying(
        vector_field, x, y, t, time_array, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k1x) and np.isfinite(k1y)):
        return fill_value, fill_value

    k2x, k2y = _interpolate_velocity_timevarying(
        vector_field,
        x + 0.5 * dt * k1x,
        y + 0.5 * dt * k1y,
        t + 0.5 * dt,
        time_array,
        x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k2x) and np.isfinite(k2y)):
        return fill_value, fill_value

    k3x, k3y = _interpolate_velocity_timevarying(
        vector_field,
        x + 0.5 * dt * k2x,
        y + 0.5 * dt * k2y,
        t + 0.5 * dt,
        time_array,
        x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k3x) and np.isfinite(k3y)):
        return fill_value, fill_value

    k4x, k4y = _interpolate_velocity_timevarying(
        vector_field,
        x + dt * k3x,
        y + dt * k3y,
        t + dt,
        time_array,
        x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k4x) and np.isfinite(k4y)):
        return fill_value, fill_value

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    if not _point_in_domain(x_next, y_next, x_min, x_max, y_min, y_max):
        return fill_value, fill_value

    return x_next, y_next


@njit
def _compute_flow_map_constant_numba(vector_field, x_grid, y_grid, time_array, fill_value):
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    nt = time_array.shape[0]

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    x_min = x_grid[0]
    x_max = x_grid[-1]
    y_min = y_grid[0]
    y_max = y_grid[-1]

    flow_map = np.full((nx, ny, nt, 2), fill_value, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            flow_map[i, j, 0, 0] = x_grid[i]
            flow_map[i, j, 0, 1] = y_grid[j]

    for i in range(nx):
        for j in range(ny):
            x = flow_map[i, j, 0, 0]
            y = flow_map[i, j, 0, 1]

            for n in range(nt - 1):
                t_n = time_array[n]
                dt_n = time_array[n + 1] - time_array[n]

                x_next, y_next = _rk4_step_constant(
                    vector_field, x, y, t_n, dt_n,
                    x_min, x_max, y_min, y_max, dx, dy, fill_value
                )

                flow_map[i, j, n + 1, 0] = x_next
                flow_map[i, j, n + 1, 1] = y_next

                if not (np.isfinite(x_next) and np.isfinite(y_next)):
                    break

                x = x_next
                y = y_next

    return flow_map


@njit
def _compute_flow_map_timevarying_numba(vector_field, x_grid, y_grid, time_array, fill_value):
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    nt = time_array.shape[0]

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    x_min = x_grid[0]
    x_max = x_grid[-1]
    y_min = y_grid[0]
    y_max = y_grid[-1]

    flow_map = np.full((nx, ny, nt, 2), fill_value, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            flow_map[i, j, 0, 0] = x_grid[i]
            flow_map[i, j, 0, 1] = y_grid[j]

    for i in range(nx):
        for j in range(ny):
            x = flow_map[i, j, 0, 0]
            y = flow_map[i, j, 0, 1]

            for n in range(nt - 1):
                t_n = time_array[n]
                dt_n = time_array[n + 1] - time_array[n]

                x_next, y_next = _rk4_step_timevarying(
                    vector_field, x, y, t_n, dt_n, time_array,
                    x_min, x_max, y_min, y_max, dx, dy, fill_value
                )

                flow_map[i, j, n + 1, 0] = x_next
                flow_map[i, j, n + 1, 1] = y_next

                if not (np.isfinite(x_next) and np.isfinite(y_next)):
                    break

                x = x_next
                y = y_next

    return flow_map


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

    x_grid : 1D ndarray or None
    y_grid : 1D ndarray or None
    time_array : 1D ndarray or None
    constant_in_time : bool, default=False
    fill_value : float, default=np.nan

    Returns
    -------
    flow_map : ndarray, shape (nx, ny, nt, 2)
    """

    vector_field = np.asarray(vector_field, dtype=np.float64)

    if constant_in_time:
        if vector_field.ndim != 3 or vector_field.shape[-1] != 2:
            raise ValueError(
                "For constant_in_time=True, vector_field must have shape (nx, ny, 2)."
            )
        nx, ny, _ = vector_field.shape

        if time_array is None:
            time_array = np.array([0.0, 1.0], dtype=np.float64)
        else:
            time_array = np.asarray(time_array, dtype=np.float64)

    else:
        if vector_field.ndim != 4 or vector_field.shape[-1] != 2:
            raise ValueError(
                "For constant_in_time=False, vector_field must have shape (nx, ny, nt, 2)."
            )
        nx, ny, nt, _ = vector_field.shape

        if time_array is None:
            time_array = np.arange(nt, dtype=np.float64)
        else:
            time_array = np.asarray(time_array, dtype=np.float64)
            if len(time_array) != nt:
                raise ValueError("time_array must have length nt.")

    if x_grid is None:
        x_grid = np.arange(nx, dtype=np.float64)
    else:
        x_grid = np.asarray(x_grid, dtype=np.float64)

    if y_grid is None:
        y_grid = np.arange(ny, dtype=np.float64)
    else:
        y_grid = np.asarray(y_grid, dtype=np.float64)

    if len(x_grid) != nx:
        raise ValueError("x_grid must have length nx.")
    if len(y_grid) != ny:
        raise ValueError("y_grid must have length ny.")
    if len(time_array) < 2:
        raise ValueError("time_array must contain at least two time points.")

    if len(x_grid) < 2 or len(y_grid) < 2:
        raise ValueError("x_grid and y_grid must each contain at least two points.")

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    if np.any(np.abs(np.diff(x_grid) - dx) > 1e-12):
        raise ValueError("x_grid must be uniformly spaced.")
    if np.any(np.abs(np.diff(y_grid) - dy) > 1e-12):
        raise ValueError("y_grid must be uniformly spaced.")

    if not constant_in_time:
        dt_ref = time_array[1] - time_array[0]
        if np.any(np.abs(np.diff(time_array) - dt_ref) > 1e-12):
            raise ValueError(
                "time_array must be uniformly spaced for the time-varying implementation."
            )

    if constant_in_time:
        return _compute_flow_map_constant_numba(
            vector_field, x_grid, y_grid, time_array, float(fill_value)
        )

    return _compute_flow_map_timevarying_numba(
        vector_field, x_grid, y_grid, time_array, float(fill_value)
    )










