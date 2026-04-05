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
def _interpolate_velocity_constant_regular(vector_field, x, y, x_min, x_max, y_min, y_max, dx, dy):
    u = _bilinear_interpolate_space(vector_field[:, :, 0], x, y, x_min, x_max, y_min, y_max, dx, dy)
    v = _bilinear_interpolate_space(vector_field[:, :, 1], x, y, x_min, x_max, y_min, y_max, dx, dy)

    if not (np.isfinite(u) and np.isfinite(v)):
        return np.nan, np.nan

    return u, v


@njit
def _interpolate_velocity_timevarying_regular(
    vector_field, x, y, t, time_array, x_min, x_max, y_min, y_max, dx, dy
):
    nt = vector_field.shape[2]
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

    u0 = _bilinear_interpolate_space(vector_field[:, :, k0, 0], x, y, x_min, x_max, y_min, y_max, dx, dy)
    u1 = _bilinear_interpolate_space(vector_field[:, :, k1, 0], x, y, x_min, x_max, y_min, y_max, dx, dy)
    v0 = _bilinear_interpolate_space(vector_field[:, :, k0, 1], x, y, x_min, x_max, y_min, y_max, dx, dy)
    v1 = _bilinear_interpolate_space(vector_field[:, :, k1, 1], x, y, x_min, x_max, y_min, y_max, dx, dy)

    if not (np.isfinite(u0) and np.isfinite(u1) and np.isfinite(v0) and np.isfinite(v1)):
        return np.nan, np.nan

    u = (1.0 - tau) * u0 + tau * u1
    v = (1.0 - tau) * v0 + tau * v1
    return u, v


@njit
def _rk4_step_regular_constant(
    vector_field, x, y, t, dt, x_min, x_max, y_min, y_max, dx, dy, fill_value
):
    k1x, k1y = _interpolate_velocity_constant_regular(vector_field, x, y, x_min, x_max, y_min, y_max, dx, dy)
    if not (np.isfinite(k1x) and np.isfinite(k1y)):
        return fill_value, fill_value

    k2x, k2y = _interpolate_velocity_constant_regular(
        vector_field, x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k2x) and np.isfinite(k2y)):
        return fill_value, fill_value

    k3x, k3y = _interpolate_velocity_constant_regular(
        vector_field, x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k3x) and np.isfinite(k3y)):
        return fill_value, fill_value

    k4x, k4y = _interpolate_velocity_constant_regular(
        vector_field, x + dt * k3x, y + dt * k3y, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k4x) and np.isfinite(k4y)):
        return fill_value, fill_value

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    if not _point_in_domain(x_next, y_next, x_min, x_max, y_min, y_max):
        return fill_value, fill_value

    return x_next, y_next


@njit
def _rk4_step_regular_timevarying(
    vector_field, x, y, t, dt, time_array, x_min, x_max, y_min, y_max, dx, dy, fill_value
):
    k1x, k1y = _interpolate_velocity_timevarying_regular(
        vector_field, x, y, t, time_array, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k1x) and np.isfinite(k1y)):
        return fill_value, fill_value

    k2x, k2y = _interpolate_velocity_timevarying_regular(
        vector_field, x + 0.5 * dt * k1x, y + 0.5 * dt * k1y,
        t + 0.5 * dt, time_array, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k2x) and np.isfinite(k2y)):
        return fill_value, fill_value

    k3x, k3y = _interpolate_velocity_timevarying_regular(
        vector_field, x + 0.5 * dt * k2x, y + 0.5 * dt * k2y,
        t + 0.5 * dt, time_array, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k3x) and np.isfinite(k3y)):
        return fill_value, fill_value

    k4x, k4y = _interpolate_velocity_timevarying_regular(
        vector_field, x + dt * k3x, y + dt * k3y,
        t + dt, time_array, x_min, x_max, y_min, y_max, dx, dy
    )
    if not (np.isfinite(k4x) and np.isfinite(k4y)):
        return fill_value, fill_value

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    if not _point_in_domain(x_next, y_next, x_min, x_max, y_min, y_max):
        return fill_value, fill_value

    return x_next, y_next


@njit
def _compute_flow_map_regular_seeded_constant(vector_field, x_grid, y_grid, initial_positions, time_array, fill_value):
    mx, my, _ = initial_positions.shape
    nt = time_array.shape[0]

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    x_min = x_grid[0]
    x_max = x_grid[-1]
    y_min = y_grid[0]
    y_max = y_grid[-1]

    flow_map = np.full((mx, my, nt, 2), fill_value, dtype=np.float64)
    flow_map[:, :, 0, :] = initial_positions

    for i in range(mx):
        for j in range(my):
            x = initial_positions[i, j, 0]
            y = initial_positions[i, j, 1]

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            for n in range(nt - 1):
                t_n = time_array[n]
                dt_n = time_array[n + 1] - time_array[n]

                x_next, y_next = _rk4_step_regular_constant(
                    vector_field, x, y, t_n, dt_n, x_min, x_max, y_min, y_max, dx, dy, fill_value
                )

                flow_map[i, j, n + 1, 0] = x_next
                flow_map[i, j, n + 1, 1] = y_next

                if not (np.isfinite(x_next) and np.isfinite(y_next)):
                    break

                x = x_next
                y = y_next

    return flow_map


@njit
def _compute_flow_map_regular_seeded_timevarying(vector_field, x_grid, y_grid, initial_positions, time_array, fill_value):
    mx, my, _ = initial_positions.shape
    nt = time_array.shape[0]

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    x_min = x_grid[0]
    x_max = x_grid[-1]
    y_min = y_grid[0]
    y_max = y_grid[-1]

    flow_map = np.full((mx, my, nt, 2), fill_value, dtype=np.float64)
    flow_map[:, :, 0, :] = initial_positions

    for i in range(mx):
        for j in range(my):
            x = initial_positions[i, j, 0]
            y = initial_positions[i, j, 1]

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            for n in range(nt - 1):
                t_n = time_array[n]
                dt_n = time_array[n + 1] - time_array[n]

                x_next, y_next = _rk4_step_regular_timevarying(
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
    initial_positions,
    vec_x_grid=None,
    vec_y_grid=None,
    time_array=None,
    constant_in_time=False,
    fill_value=np.nan
):
    """
    Compute a flow map using a vector field on a regular grid,
    but with arbitrary initial particle positions.

    Parameters
    ----------
    vector_field : ndarray
        (nx, ny, nt, 2) if time-varying
        (nx, ny, 2) if constant_in_time=True

    initial_positions : ndarray, shape (mx, my, 2)
        Initial particle positions.

    vec_x_grid, vec_y_grid : 1D arrays
        Grid on which the vector field is sampled.

    Returns
    -------
    flow_map : ndarray, shape (mx, my, nt, 2)
    """
    vector_field = np.asarray(vector_field, dtype=np.float64)
    initial_positions = np.asarray(initial_positions, dtype=np.float64)

    if initial_positions.ndim != 3 or initial_positions.shape[-1] != 2:
        raise ValueError("initial_positions must have shape (mx, my, 2).")

    if constant_in_time:
        if vector_field.ndim != 3 or vector_field.shape[-1] != 2:
            raise ValueError("For constant_in_time=True, vector_field must have shape (nx, ny, 2).")
        nx, ny, _ = vector_field.shape
        if time_array is None:
            time_array = np.array([0.0, 1.0], dtype=np.float64)
        else:
            time_array = np.asarray(time_array, dtype=np.float64)
    else:
        if vector_field.ndim != 4 or vector_field.shape[-1] != 2:
            raise ValueError("For constant_in_time=False, vector_field must have shape (nx, ny, nt, 2).")
        nx, ny, nt, _ = vector_field.shape
        if time_array is None:
            time_array = np.arange(nt, dtype=np.float64)
        else:
            time_array = np.asarray(time_array, dtype=np.float64)
            if len(time_array) != nt:
                raise ValueError("time_array must have length nt.")

    if vec_x_grid is None:
        vec_x_grid = np.arange(nx, dtype=np.float64)
    else:
        vec_x_grid = np.asarray(vec_x_grid, dtype=np.float64)

    if vec_y_grid is None:
        vec_y_grid = np.arange(ny, dtype=np.float64)
    else:
        vec_y_grid = np.asarray(vec_y_grid, dtype=np.float64)

    if len(vec_x_grid) != nx or len(vec_y_grid) != ny:
        raise ValueError("vec_x_grid and y_grid must match vector_field shape.")

    if len(vec_x_grid) < 2 or len(vec_y_grid) < 2:
        raise ValueError("vec_x_grid and y_grid must each contain at least two points.")

    dx = vec_x_grid[1] - vec_x_grid[0]
    dy = vec_y_grid[1] - vec_y_grid[0]

    if np.any(np.abs(np.diff(vec_x_grid) - dx) > 1e-12):
        raise ValueError("vec_x_grid must be uniformly spaced.")
    if np.any(np.abs(np.diff(vec_y_grid) - dy) > 1e-12):
        raise ValueError("vec_y_grid must be uniformly spaced.")

    if not constant_in_time:
        dt_ref = time_array[1] - time_array[0]
        if np.any(np.abs(np.diff(time_array) - dt_ref) > 1e-12):
            raise ValueError("time_array must be uniformly spaced for the time-varying implementation.")

    if constant_in_time:
        return _compute_flow_map_regular_seeded_constant(
            vector_field, vec_x_grid, vec_y_grid, initial_positions, time_array, float(fill_value)
        )

    return _compute_flow_map_regular_seeded_timevarying(
        vector_field, vec_x_grid, vec_y_grid, initial_positions, time_array, float(fill_value)
    )









def compute_flow_map_RK4_2D_general_positions(
    vector_field,
    field_positions,
    initial_positions=None,
    time_array=None,
    constant_in_time=False,
    fill_value=np.nan
):
    """
    Compute a flow map from a vector field whose sample positions are given explicitly.

    Parameters
    ----------
    vector_field : ndarray
        (nx, ny, nt, 2) if time-varying
        (nx, ny, 2) if constant_in_time=True

    field_positions : ndarray
        Same spatial layout as vector_field, but last axis = 2.
        If constant_in_time=False: shape (nx, ny, nt, 2)
        If constant_in_time=True:  shape (nx, ny, 2)

    initial_positions : ndarray or None
        Shape (mx, my, 2). If None, uses field_positions at t=0.

    Returns
    -------
    flow_map : ndarray, shape (mx, my, nt, 2)
    """
    vector_field = np.asarray(vector_field, dtype=float)
    field_positions = np.asarray(field_positions, dtype=float)

    if constant_in_time:
        if vector_field.ndim != 3 or vector_field.shape[-1] != 2:
            raise ValueError("For constant_in_time=True, vector_field must have shape (nx, ny, 2).")
        if field_positions.shape != vector_field.shape:
            raise ValueError("For constant_in_time=True, field_positions must have same shape as vector_field.")
        nx, ny, _ = vector_field.shape
        if time_array is None:
            time_array = np.array([0.0, 1.0], dtype=float)
        else:
            time_array = np.asarray(time_array, dtype=float)
        if initial_positions is None:
            initial_positions = field_positions.copy()
    else:
        if vector_field.ndim != 4 or vector_field.shape[-1] != 2:
            raise ValueError("For constant_in_time=False, vector_field must have shape (nx, ny, nt, 2).")
        if field_positions.shape != vector_field.shape:
            raise ValueError("For constant_in_time=False, field_positions must have same shape as vector_field.")
        nx, ny, nt, _ = vector_field.shape
        if time_array is None:
            time_array = np.arange(nt, dtype=float)
        else:
            time_array = np.asarray(time_array, dtype=float)
            if len(time_array) != nt:
                raise ValueError("time_array must have length nt.")
        if initial_positions is None:
            initial_positions = field_positions[:, :, 0, :].copy()

    initial_positions = np.asarray(initial_positions, dtype=float)
    if initial_positions.ndim != 3 or initial_positions.shape[-1] != 2:
        raise ValueError("initial_positions must have shape (mx, my, 2).")

    mx, my, _ = initial_positions.shape
    nt_out = len(time_array)

    flow_map = np.full((mx, my, nt_out, 2), fill_value, dtype=float)
    flow_map[:, :, 0, :] = initial_positions

    def make_interpolators_for_time(k):
        pts = field_positions[:, :, k, :].reshape(-1, 2)
        u = vector_field[:, :, k, 0].reshape(-1)
        v = vector_field[:, :, k, 1].reshape(-1)

        valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(u) & np.isfinite(v)
        pts = pts[valid]
        u = u[valid]
        v = v[valid]

        interp_u = LinearNDInterpolator(pts, u, fill_value=np.nan)
        interp_v = LinearNDInterpolator(pts, v, fill_value=np.nan)
        return interp_u, interp_v

    if constant_in_time:
        pts = field_positions.reshape(-1, 2)
        u = vector_field[:, :, 0].reshape(-1)
        v = vector_field[:, :, 1].reshape(-1)
        valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(u) & np.isfinite(v)
        interp_u = LinearNDInterpolator(pts[valid], u[valid], fill_value=np.nan)
        interp_v = LinearNDInterpolator(pts[valid], v[valid], fill_value=np.nan)

        def vel(x, y, t):
            return float(interp_u(x, y)), float(interp_v(x, y))
    else:
        if len(time_array) < 2:
            raise ValueError("time_array must contain at least two time points.")
        interpolators = [make_interpolators_for_time(k) for k in range(vector_field.shape[2])]
        dt_ref = time_array[1] - time_array[0]
        if np.any(np.abs(np.diff(time_array) - dt_ref) > 1e-12):
            raise ValueError("time_array must be uniformly spaced for this implementation.")

        def vel(x, y, t):
            if t < time_array[0] or t > time_array[-1]:
                return np.nan, np.nan
            if t == time_array[-1]:
                k0 = len(time_array) - 2
                k1 = len(time_array) - 1
                tau = 1.0
            else:
                ft = (t - time_array[0]) / dt_ref
                k0 = int(np.floor(ft))
                if k0 < 0:
                    k0 = 0
                if k0 > len(time_array) - 2:
                    k0 = len(time_array) - 2
                k1 = k0 + 1
                tau = ft - k0

            iu0, iv0 = interpolators[k0]
            iu1, iv1 = interpolators[k1]

            u0 = float(iu0(x, y))
            u1 = float(iu1(x, y))
            v0 = float(iv0(x, y))
            v1 = float(iv1(x, y))

            if not np.all(np.isfinite([u0, u1, v0, v1])):
                return np.nan, np.nan

            u = (1.0 - tau) * u0 + tau * u1
            v = (1.0 - tau) * v0 + tau * v1
            return u, v

    def rk4_step(x, y, t, dt):
        k1x, k1y = vel(x, y, t)
        if not np.all(np.isfinite([k1x, k1y])):
            return fill_value, fill_value

        k2x, k2y = vel(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, t + 0.5 * dt)
        if not np.all(np.isfinite([k2x, k2y])):
            return fill_value, fill_value

        k3x, k3y = vel(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, t + 0.5 * dt)
        if not np.all(np.isfinite([k3x, k3y])):
            return fill_value, fill_value

        k4x, k4y = vel(x + dt * k3x, y + dt * k3y, t + dt)
        if not np.all(np.isfinite([k4x, k4y])):
            return fill_value, fill_value

        x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        return x_next, y_next

    for i in range(mx):
        for j in range(my):
            x = initial_positions[i, j, 0]
            y = initial_positions[i, j, 1]

            if not np.all(np.isfinite([x, y])):
                continue

            for n in range(nt_out - 1):
                t_n = time_array[n]
                dt_n = time_array[n + 1] - time_array[n]

                x_next, y_next = rk4_step(x, y, t_n, dt_n)
                flow_map[i, j, n + 1, 0] = x_next
                flow_map[i, j, n + 1, 1] = y_next

                if not np.all(np.isfinite([x_next, y_next])):
                    break

                x, y = x_next, y_next

    return flow_map







