from GeneralDynamics.LyapunovExponents.FTLE import compute_FTLE_2D_gridDomain
from GeneralDynamics.IntegrationSchemes.AdvectionSchemes import compute_flow_map_RK4_2D

import numpy as np

import matplotlib.pyplot as plt







def plot_FTLE_2D(
    FTLE,
    time_index=-1,
    x=None,
    y=None,
    title=None,
    cmap='viridis',
    vmin=None,
    vmax=None,
    show_colorbar=True
):
    """
    Plot a 2D FTLE field at a given time index.

    Parameters
    ----------
    FTLE : ndarray (nx, ny, nt)
        FTLE field.

    time_index : int, default=-1
        Time slice to plot.

    x, y : 1D arrays or None
        Grid coordinates. If None, uses index grid.

    title : str or None
        Plot title.

    cmap : str
        Colormap.

    vmin, vmax : float or None
        Color limits.

    show_colorbar : bool
        Whether to show colorbar.
    """

    FTLE_slice = FTLE[:, :, time_index]

    # Mask invalid values
    FTLE_masked = np.ma.masked_invalid(FTLE_slice)

    nx, ny = FTLE_slice.shape

    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)

    X, Y = np.meshgrid(y, x)  # NOTE: consistent with array indexing

    plt.figure(figsize=(6, 5))

    im = plt.pcolormesh(
        X, Y,
        FTLE_masked,
        cmap=cmap,
        shading='auto',
        vmin=vmin,
        vmax=vmax
    )

    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.set_label("FTLE")

    plt.xlabel("y")
    plt.ylabel("x")

    if title is None:
        plt.title(f"FTLE at time index {time_index}")
    else:
        plt.title(title)

    plt.tight_layout()
    plt.show()




nx, ny, nt =60, 60, 20

x_grid = np.linspace(-1, 1, nx)
y_grid = np.linspace(-1, 1, ny)
time_array = np.linspace(0, 1, nt)

vector_field = np.zeros((nx, ny, nt, 2))

X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

for k, t in enumerate(time_array):
    vector_field[:, :, k, 0] = -Y * np.cos(t)
    vector_field[:, :, k, 1] =  X * np.cos(t)

flow_map = compute_flow_map_RK4_2D(
    vector_field,
    x_grid=x_grid,
    y_grid=y_grid,
    time_array=time_array,
    constant_in_time=False
)





FTLE = compute_FTLE_2D_gridDomain(flow_map, final_time_index=nt-1)





plot_FTLE_2D(
    FTLE,
    time_index=10,
    x=np.linspace(-1, 1, FTLE.shape[0]),
    y=np.linspace(-1, 1, FTLE.shape[1])
)



plot_FTLE_2D(
    FTLE,
    time_index=20,
    x=np.linspace(-1, 1, FTLE.shape[0]),
    y=np.linspace(-1, 1, FTLE.shape[1])
)










