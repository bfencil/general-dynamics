import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from itertools import combinations

x_start = -1
y_start = -1

number_points = 300
starting_points = np.zeros((number_points, 2))

starting_points[:, 0] = x_start
starting_points[:, 1] = y_start




mult = 20
x_low_bound = -1*mult
x_high_bound = 1*mult
y_low_bound = -1*mult
y_high_bound = 1*mult

resolution = 1000

x_sample_range = np.linspace(x_low_bound, x_high_bound, resolution)
y_sample_range = np.linspace(y_low_bound, y_high_bound, resolution)

# starting_points[:, 0] = np.random.choice(x_sample_range, size=number_points)
# starting_points[:, 1] = np.random.choice(y_sample_range, size=number_points)

number_steps = 3000
ball_radius = 0.3
number_angles = 24







def HyperbolicLCS_2D(xP,yP, final_time, direction,eps,lam=1e-10):
    """
    Computes hyperbolic LCSs in 2D trajectory datasets

    Args:
        xP, yP: 2D arrays
            Coordinates of particle trajectories, with lines representing
            different particles and columns representing different times
        tv: 1D array
            Times values corresponding to the columns of xP, yP
        direction: string
            Type of hyperbolic LCS, 'forward' or 'backward'
        eps: scalar
            Neighborhood radius for particles to feed in the least squares
            algorithm (refer to paper for more details)
        lam: scalar, optional
            Weight of regularization term in the least squares objective
            function (refer to paper for more details)

    Returns:
        FTLE: 1D array
            FTLE values at each particle location; the forward
            and backward FTLE are displayed at the beginning and end of the 
            time window, respectively
    """

    # Number of particles

    Time = number_steps

    # Extract initial and final time snapshots
    if direction == 'forward':
        xP0, yP0 = xP[0, :], yP[0, :]
        xPf, yPf = xP[final_time, :], yP[final_time, :]
    elif direction == 'backward':
        xP0, yP0 = xP[-1, :], yP[-1, :]
        xPf, yPf = xP[0, :], yP[0, :]
    else:
        print('Error: direction argument must be \'forward\' or \'backward\'')

    # Calculate FTLE values at each particle
    FTLE = np.zeros(number_points)
    for i in range(number_points):
        # Compute initial distances
        dxP = xP0-xP0[i]
        dyP = yP0-yP0[i]
        
        # Find pairwise combinations of neighbor indices
        neighbors = np.flatnonzero((dxP**2+dyP**2)<eps**2)
        combs = list(combinations(range(len(neighbors)),2))
        ind1 = [comb[0] for comb in combs]
        ind2 = [comb[1] for comb in combs]
        
        # Form X and Y data matrices
        X = np.zeros((2,len(combs)))
        Y = np.zeros((2,len(combs)))
        X[0,:] = xP0[neighbors[ind1]]-xP0[neighbors[ind2]]
        X[1,:] = yP0[neighbors[ind1]]-yP0[neighbors[ind2]]
        Y[0,:] = xPf[neighbors[ind1]]-xPf[neighbors[ind2]]
        Y[1,:] = yPf[neighbors[ind1]]-yPf[neighbors[ind2]]
        
        # Least square fit of flow map gradient
        A = Y@X.T + lam*max(1,len(neighbors))*np.eye(2)
        B = X@X.T + lam*max(1,len(neighbors))*np.eye(2)
        DF = A@np.linalg.inv(B)
        
        # Calculate FTLE as the largest singular value of DF
        FTLE[i] = np.log(np.linalg.norm(DF,2))/Time

    return FTLE




angles = np.linspace(0, 2*np.pi - 2*np.pi/number_angles, number_angles)






def random_walk(starting_points, number_steps, ball_radius, number_angles, x_boundary = [-1, 1], y_boundary = [-1, 1], torus_boundary = False, mobius_strip_boundary = False, soft_boundary = True, spherical_boundary=False):

    truth_sum = torus_boundary + mobius_strip_boundary + soft_boundary
    if truth_sum < 1:
        raise ValueError("All boundary conditions are set to False, one condition is required to be True.")
    if truth_sum > 1:
        raise ValueError("More than one boundary condition is set to True, only one can be active.")

    if len(x_boundary) != 2:
        raise ValueError("The array specifiying x_boundary must have 2 and only 2 entries")
    if len(y_boundary) != 2:
        raise ValueError("The array specifiying y_boundary must have 2 and only 2 entries")


    if torus_boundary:
        x_wrap = np.abs(x_boundary[1] - x_boundary[0]) 
        y_wrap = np.abs(y_boundary[1] - y_boundary[0])
    
    walk = np.zeros((number_steps, number_points, 2))
    print(walk.shape)
    walk[0, :, 0] = starting_points[:, 0]
    walk[0, :, 1] = starting_points[:, 1]

    angles = np.linspace(0, 2*np.pi - 2*np.pi/number_angles, number_angles)

    current_x = starting_points[:, 0]
    current_y = starting_points[:, 1]
    for i in range(number_steps-1):
        print("iteration: ", i)
        random_angle = np.random.choice(angles, size=number_points)  # in radians

        new_x = current_x + ball_radius * np.cos(random_angle)
        new_y = current_y + ball_radius * np.sin(random_angle)


        if torus_boundary:
            new_x[new_x >= x_boundary[1]] -= x_wrap
            new_x[new_x <= x_boundary[0]] += x_wrap

            new_y[new_y >= y_boundary[1]] -= y_wrap
            new_y[new_y <= y_boundary[0]] += y_wrap

        if spherical_boundary:
            new_x[new_x >= x_boundary[1]] -= x_wrap
            new_x[new_x <= x_boundary[0]] += x_wrap

            new_y[new_y >= y_boundary[1]] -= y_wrap
            new_y[new_y <= y_boundary[0]] += y_wrap

        current_x = new_x
        current_y = new_y
        walk[i+1, :, 0] = new_x
        walk[i+1, :, 1] = new_y

    return walk




###
# general dimension random walks based on meshes
###




###
# 3D random walks based on meshes
###



###
# 2D random walks based on meshes
###





###
# 3D random walks based on boundary conditions
###




###
# 2D random walks based on boundary conditions
###

def random_walk_2D_no_boundary(starting_points, number_steps, ball_radius, number_angles):
    walk = np.zeros((number_steps, number_points, 2))
    
    walk[0, :, 0] = starting_points[:, 0]
    walk[0, :, 1] = starting_points[:, 1]

    angles = np.linspace(0, 2*np.pi - 2*np.pi/number_angles, number_angles)

    current_x = starting_points[:, 0]
    current_y = starting_points[:, 1]
    for i in range(number_steps-1):
        print("iteration: ", i)
        random_angle = np.random.choice(angles, size=number_points)  # in radians

        new_x = current_x + ball_radius * np.cos(random_angle)
        new_y = current_y + ball_radius * np.sin(random_angle)

        current_x = new_x
        current_y = new_y
        walk[i+1, :, 0] = new_x
        walk[i+1, :, 1] = new_y

    return walk

def random_walk_2D_torus(starting_points, number_steps, ball_radius, number_angles, x_boundary = [-1, 1], y_boundary = [-1, 1]):

    
    x_wrap = np.abs(x_boundary[1] - x_boundary[0]) 
    y_wrap = np.abs(y_boundary[1] - y_boundary[0])
    
    walk = np.zeros((number_steps, number_points, 2))
    
    walk[0, :, 0] = starting_points[:, 0]
    walk[0, :, 1] = starting_points[:, 1]

    angles = np.linspace(0, 2*np.pi - 2*np.pi/number_angles, number_angles)

    current_x = starting_points[:, 0]
    current_y = starting_points[:, 1]
    for i in range(number_steps-1):
        print("iteration: ", i)
        random_angle = np.random.choice(angles, size=number_points)  # in radians

        new_x = current_x + ball_radius * np.cos(random_angle)
        new_y = current_y + ball_radius * np.sin(random_angle)

        
        new_x[new_x >= x_boundary[1]] -= x_wrap
        new_x[new_x <= x_boundary[0]] += x_wrap

        new_y[new_y >= y_boundary[1]] -= y_wrap
        new_y[new_y <= y_boundary[0]] += y_wrap


        current_x = new_x
        current_y = new_y
        walk[i+1, :, 0] = new_x
        walk[i+1, :, 1] = new_y

    return walk


def random_walk_2D_general_domain(starting_points, number_steps, ball_radius, number_angles, signed_distance_function):

    # use a signed distance function(SDF) to specify the domains shape, the domains boundary will be assumed to be hard(block movement)

    return 0 



def reflect_into_interval(y, y0, y1):
    """
    Reflect y into [y0, y1] with hard walls, robust to arbitrary overshoot.
    Vectorized.
    """
    L = y1 - y0
    # map to [0, 2L) then fold
    t = (y - y0) % (2 * L)
    y_ref = np.where(t <= L, y0 + t, y1 - (t - L))
    return y_ref

def mobius_wrap_x(new_x, new_y, x_boundary, y_boundary, mobius_twists=1):
    x0, x1 = x_boundary
    y0, y1 = y_boundary
    Lx = x1 - x0

    # number of wraps (can be negative, can be >1 if step is big)
    k = np.floor((new_x - x0) / Lx).astype(int)
    new_x = new_x - k * Lx  # now in [x0, x1)

    # flip y when (k * mobius_twists) is odd
    flip = ((k * mobius_twists) & 1).astype(bool)
    new_y = np.where(flip, (y0 + y1) - new_y, new_y)

    return new_x, new_y


def random_walk_2D_mobius_strip(
    starting_points,
    number_steps,
    ball_radius,
    number_angles,
    x_boundary=(-1, 1),
    y_boundary=(-1, 1),
    mobius_twists=1,
):
    starting_points = np.asarray(starting_points, dtype=float)
    number_points = starting_points.shape[0]

    x0, x1 = x_boundary
    y0, y1 = y_boundary

    walk = np.zeros((number_steps, number_points, 2), dtype=float)
    walk[0] = starting_points

    angles = np.linspace(0, 2*np.pi, number_angles, endpoint=False)

    current_x = starting_points[:, 0].copy()
    current_y = starting_points[:, 1].copy()

    for i in range(number_steps - 1):
        random_angle = np.random.choice(angles, size=number_points)

        new_x = current_x + ball_radius * np.cos(random_angle)
        new_y = current_y + ball_radius * np.sin(random_angle)

        # Möbius wrap in x (includes y flip on crossing x seam)
        new_x, new_y = mobius_wrap_x(new_x, new_y, x_boundary, y_boundary, mobius_twists)

        # Hard walls in y (reflect into [y0, y1], robust to overshoot)
        new_y = reflect_into_interval(new_y, y0, y1)

        current_x, current_y = new_x, new_y
        walk[i + 1, :, 0] = new_x
        walk[i + 1, :, 1] = new_y

    return walk


def random_walk_2D_hard_boundary(starting_points, number_steps, ball_radius, number_angles, x_boundary = [-1, 1], y_boundary = [-1, 1]):


    return 0


def random_walk_2D_sphere(starting_points, number_steps, ball_radius, number_angles, x_boundary = [-1, 1], y_boundary = [-1, 1]):


    return 0



def random_walks_2D(starting_points, number_steps, ball_radius, number_angles, x_boundary = [-1, 1], y_boundary = [-1, 1], torus_boundary = False, mobius_strip_boundary = False, mobius_twists = 1, hard_boundary = False, spherical_boundary=False, no_boundary=False):


    truth_sum = torus_boundary + mobius_strip_boundary + hard_boundary + no_boundary
    if truth_sum < 1:
        raise ValueError("All boundary conditions are set to False, one condition is required to be True.")
    if truth_sum > 1:
        raise ValueError("More than one boundary condition is set to True, only one can be active.")

    if len(x_boundary) != 2:
        raise ValueError("The array specifiying x_boundary must have 2 and only 2 entries")
    if len(y_boundary) != 2:
        raise ValueError("The array specifiying y_boundary must have 2 and only 2 entries")


    if torus_boundary:
        walk = random_walk_2D_torus(starting_points, number_steps, ball_radius, number_angles, x_boundary = x_boundary, y_boundary = y_boundary)

    elif mobius_strip_boundary:
        walk = random_walk_2D_mobius_strip(starting_points, number_steps, ball_radius, number_angles, x_boundary = x_boundary, y_boundary = y_boundary, mobius_twists = mobius_twists)
    elif hard_boundary:
        0
    elif spherical_boundary:
        0
    elif no_boundary:
        walk = random_walk_2D_no_boundary(starting_points, number_steps, ball_radius, number_angles)


    return walk


#walk = random_walk(starting_points, number_steps, ball_radius, number_angles, x_boundary = [x_low_bound, x_high_bound], y_boundary = [y_low_bound, y_high_bound], torus_boundary = True, mobius_strip_boundary = False, soft_boundary = False, spherical_boundary=False)


mobius_twists = 30
walk = random_walks_2D(starting_points, number_steps, ball_radius, number_angles, x_boundary = [x_low_bound, x_high_bound], y_boundary = [y_low_bound, y_high_bound], torus_boundary = False, mobius_strip_boundary=True, mobius_twists=mobius_twists)



max_radius = number_steps*ball_radius

effective_radius = 0.15*max_radius






def estimate_u_stretch(R=2.0, w=0.6, twists=1, samples_u=400, samples_v=30):
    # estimate mean ||∂F/∂u|| over the band
    u = np.linspace(0, 2*np.pi, samples_u, endpoint=False)
    v = np.linspace(-w, w, samples_v)
    U, V = np.meshgrid(u, v)

    A = twists * U / 2.0

    # F(U,V)
    X = (R + V*np.cos(A)) * np.cos(U)
    Y = (R + V*np.cos(A)) * np.sin(U)
    Z = V*np.sin(A)

    # finite-diff in u
    du = 2*np.pi / samples_u
    U2 = U + du
    A2 = twists * U2 / 2.0
    X2 = (R + V*np.cos(A2)) * np.cos(U2)
    Y2 = (R + V*np.cos(A2)) * np.sin(U2)
    Z2 = V*np.sin(A2)

    dXu = np.sqrt((X2-X)**2 + (Y2-Y)**2 + (Z2-Z)**2) / du
    return dXu.mean()



def walk_to_mobius_xyz_visual(walk, x_boundary=(-1, 1), y_boundary=(-1, 1),
                             R=2.0, w=0.6, twists=1, u_stretch=None):
    xmin, xmax = x_boundary
    ymin, ymax = y_boundary
    Lx = (xmax - xmin)
    Ly = (ymax - ymin)

    u = 2*np.pi * (walk[..., 0] - xmin) / Lx
    v = -w + (2*w) * (walk[..., 1] - ymin) / Ly

    if u_stretch is None:
        u_stretch = estimate_u_stretch(R=R, w=w, twists=twists)

    u2 = u / u_stretch  # compress the long direction (purely visual!)

    A = twists * u2 / 2.0
    X = (R + v*np.cos(A)) * np.cos(u2)
    Y = (R + v*np.cos(A)) * np.sin(u2)
    Z = v*np.sin(A)

    return np.stack([X, Y, Z], axis=-1)


def walk_to_mobius_xyz(walk, x_boundary=(-1, 1), y_boundary=(-1, 1), R=2.0, w=0.6, twists=1):
    """
    Convert a Möbius-strip-wrapped 2D walk in [xmin,xmax)×[ymin,ymax] to 3D Möbius band coordinates.

    Model:
      u in [0, 2π), v in [-w, w]
      X = (R + v cos(twists*u/2)) cos u
      Y = (R + v cos(twists*u/2)) sin u
      Z = v sin(twists*u/2)

    walk: (T, N, 2) array
    returns xyz: (T, N, 3) array
    """
    xmin, xmax = x_boundary
    ymin, ymax = y_boundary
    Lx = (xmax - xmin)
    Ly = (ymax - ymin)

    # Map x -> u in [0, 2π)
    u = 2*np.pi * (walk[..., 0] - xmin) / Lx  # (T,N)

    # Map y -> v in [-w, w]
    v = -w + (2*w) * (walk[..., 1] - ymin) / Ly  # (T,N)

    half_twist_angle = (twists * u) / 2.0
    cu = np.cos(u)
    su = np.sin(u)
    c = np.cos(half_twist_angle)
    s = np.sin(half_twist_angle)

    X = (R + v * c) * cu
    Y = (R + v * c) * su
    Z = v * s

    xyz = np.stack([X, Y, Z], axis=-1)
    return xyz


def plot_walk_on_mobius_3d(
    walk,
    x_boundary=(-1, 1),
    y_boundary=(-1, 1),
    R=2.0,
    w=0.6,
    twists=1,
    stride=20,
    pause=0.05,
    draw_surface=True,
    surface_res_u=140,
    surface_res_v=40,
    elev=25,
    azim=35,
):
    """
    Animate the walk on a 3D Möbius band.
    walk: (T, N, 2)
    """
    xyz = walk_to_mobius_xyz_visual(
        walk, x_boundary=x_boundary, y_boundary=y_boundary, R=R, w=w, twists=twists
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Optional Möbius surface for context
    if draw_surface:
        u = np.linspace(0, 2*np.pi, surface_res_u)
        v = np.linspace(-w, w, surface_res_v)
        U, V = np.meshgrid(u, v)

        A = (twists * U) / 2.0
        Xs = (R + V * np.cos(A)) * np.cos(U)
        Ys = (R + V * np.cos(A)) * np.sin(U)
        Zs = V * np.sin(A)

        ax.plot_surface(Xs, Ys, Zs, alpha=0.18, linewidth=0)

    # Axis limits
    lim = R + w
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-w, w)
    ax.set_box_aspect([1, 1, w/lim])  # helps avoid visual stretching

    # Initial scatter
    scat = ax.scatter(xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2], s=20)

    for t in range(0, xyz.shape[0], stride):
        xs, ys, zs = xyz[t, :, 0], xyz[t, :, 1], xyz[t, :, 2]
        scat._offsets3d = (xs, ys, zs)
        ax.set_title(f"Möbius walk | step {t} | twists={twists}")
        plt.pause(pause)

    plt.show()
    plt.close(fig)




def walk_to_torus_xyz(walk, x_boundary=(-1, 1), y_boundary=(-1, 1), R=2.0, r=0.6):
    """
    Convert a torus-wrapped 2D walk in [xmin,xmax)×[ymin,ymax) to 3D torus coordinates.

    walk: (T, N, 2) array
    returns xyz: (T, N, 3) array
    """
    xmin, xmax = x_boundary
    ymin, ymax = y_boundary
    Lx = (xmax - xmin)
    Ly = (ymax - ymin)

    # Map to angles in [0, 2π)
    theta = 2*np.pi * (walk[..., 0] - xmin) / Lx   # (T,N)
    phi   = 2*np.pi * (walk[..., 1] - ymin) / Ly   # (T,N)

    X = (R + r*np.cos(phi)) * np.cos(theta)
    Y = (R + r*np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)

    xyz = np.stack([X, Y, Z], axis=-1)
    return xyz

def plot_walk_on_torus_3d(
    walk,
    x_boundary=(-1, 1),
    y_boundary=(-1, 1),
    R=2.0,
    r=0.6,
    stride=20,
    pause=0.05,
    draw_surface=True,
    surface_res=60,
    elev=25,
    azim=35,
):
    """
    Animate the walk on a 3D torus.
    walk: (T, N, 2)
    """
    xyz = walk_to_torus_xyz(walk, x_boundary=x_boundary, y_boundary=y_boundary, R=R, r=r)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Optional torus surface for context
    if draw_surface:
        u = np.linspace(0, 2*np.pi, surface_res)
        v = np.linspace(0, 2*np.pi, surface_res)
        U, V = np.meshgrid(u, v)
        Xs = (R + r*np.cos(V)) * np.cos(U)
        Ys = (R + r*np.cos(V)) * np.sin(U)
        Zs = r*np.sin(V)
        ax.plot_surface(Xs, Ys, Zs, alpha=0.15, linewidth=0)

    # Set axis limits nicely
    lim = R + r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-r, r)
    ax.set_box_aspect([1, 1, r/lim])  # helps the torus not look stretched

    # Initial scatter
    scat = ax.scatter(xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2], s=20)

    for t in range(0, xyz.shape[0], stride):
        # Update scatter positions
        xs, ys, zs = xyz[t, :, 0], xyz[t, :, 1], xyz[t, :, 2]
        scat._offsets3d = (xs, ys, zs)

        ax.set_title(f"step {t}")
        plt.pause(pause)

    plt.show()
    plt.close(fig)





plot_walk_on_mobius_3d(
    walk,
    x_boundary=(x_low_bound, x_high_bound),
    y_boundary=(y_low_bound, y_high_bound),
    R=2.0,
    w=0.6,          # half-width of the strip in 3D
    twists=mobius_twists,       # 1 = standard Möbius
    stride=20,
    pause=0.1,
    draw_surface=True,
)



plot_walk_on_torus_3d(
    walk,
    x_boundary=(x_low_bound, x_high_bound),
    y_boundary=(y_low_bound, y_high_bound),
    R=2.0,
    r=0.6,
    stride=20,
    pause=0.1,
    draw_surface=True,
)




plt.figure(figsize=(10, 6))


plt.scatter(walk[:, :, 0], walk[:, :, 1])
plt.xlim(x_low_bound, x_high_bound)
plt.ylim(y_low_bound, y_high_bound)

plt.show()