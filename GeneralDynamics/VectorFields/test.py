import numpy as np
from __future__ import annotations
from itertools import combinations
from collections import Counter, defaultdict, deque
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict



# Optional dependency for integer homology:
try:
    import sympy as sp
    _HAVE_SYMPY = True
except ImportError:
    _HAVE_SYMPY = False




# ------------------------------------------------------------
# 2D polygons (ordered vertices)
# ------------------------------------------------------------

def build_polygon_edge_indices(ordered_points: np.ndarray) -> np.ndarray:
    """
    Create an edge-index list for a simple closed polygon whose vertices are given in order.

    Parameters
    ----------
    ordered_points : np.ndarray
        Array of shape (num_vertices, 2). The order encodes boundary connectivity.

    Returns
    -------
    np.ndarray
        Integer array of shape (num_vertices, 2). Each row (i, j) is an edge
        from vertex i to vertex j, with wrap-around closure (last -> first).
    """
    num_vertices = len(ordered_points)
    vertex_indices = np.arange(num_vertices, dtype=int)
    next_indices = (vertex_indices + 1) % num_vertices
    edge_indices = np.column_stack([vertex_indices, next_indices])
    return edge_indices


def polygon_vertex_degrees_are_two(num_vertices: int, edge_indices: np.ndarray) -> bool:
    """
    Check that each vertex in a 2D polygon has degree exactly 2 (simple closed loop).

    Parameters
    ----------
    num_vertices : int
        Number of vertices.
    edge_indices : np.ndarray
        Array of shape (num_edges, 2) with vertex index pairs.

    Returns
    -------
    bool
        True if every vertex degree equals 2; False otherwise.
    """
    degrees = np.zeros(num_vertices, dtype=int)
    for v0, v1 in edge_indices:
        degrees[v0] += 1
        degrees[v1] += 1
    return np.all(degrees == 2)


def _point_on_segment_2d(point: np.ndarray,
                         endpoint_a: np.ndarray,
                         endpoint_b: np.ndarray,
                         tol: float = 1e-12) -> bool:
    """
    Helper: check if 'point' lies on the segment AB within a tolerance.
    Uses colinearity (cross product ~ 0) and bounding-box test.
    """
    ax, ay = endpoint_a
    bx, by = endpoint_b
    px, py = point

    # Vector cross-product magnitude (2D) for colinearity
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > tol:
        return False

    # Within bounding box?
    min_x, max_x = min(ax, bx) - tol, max(ax, bx) + tol
    min_y, max_y = min(ay, by) - tol, max(ay, by) + tol
    return (min_x <= px <= max_x) and (min_y <= py <= max_y)


def point_in_polygon_even_odd(point: np.ndarray,
                              ordered_polygon: np.ndarray,
                              include_boundary: bool = True) -> bool:
    """
    Even–odd (ray casting) point-in-polygon test for an ordered simple polygon.

    Parameters
    ----------
    point : np.ndarray
        Shape (2,), the query point (x, y).
    ordered_polygon : np.ndarray
        Shape (num_vertices, 2). Ordered boundary vertices.
    include_boundary : bool, optional
        If True, points on edges/vertices are considered inside.

    Returns
    -------
    bool
        True if point is inside (or on boundary if include_boundary=True); False otherwise.
    """
    x, y = point
    vertices = ordered_polygon
    num_vertices = len(vertices)

    # Boundary check (optional)
    if include_boundary:
        for idx in range(num_vertices):
            a = vertices[idx]
            b = vertices[(idx + 1) % num_vertices]
            if _point_on_segment_2d(point, a, b):
                return True

    # Even–odd rule: count ray intersections with edges (toggle on each crossing)
    inside = False
    for idx in range(num_vertices):
        x0, y0 = vertices[idx]
        x1, y1 = vertices[(idx + 1) % num_vertices]

        # Does the edge straddle the horizontal line y = point.y?
        straddles = (y0 > y) != (y1 > y)
        if not straddles:
            continue

        # Compute x-coordinate of the intersection of the edge with the horizontal ray
        x_intersect = x0 + (x1 - x0) * (y - y0) / (y1 - y0)

        # If the ray to the right hits the edge, toggle inside
        if x < x_intersect:
            inside = not inside

    return inside


pts = np.array([[0,0],[2,0],[2,1],[1,2],[0,1]])
edges = build_polygon_edge_indices(pts)     # (N,2)
segments = np.stack([pts[edges[:,0]], pts[edges[:,1]]], axis=1)  # (N,2,2)

fig, ax = plt.subplots()
ax.add_collection(LineCollection(segments))
ax.autoscale()
ax.set_aspect('equal')
plt.show()










# ------------------------------------------------------------
# General utilities for simplicial complexes
# ------------------------------------------------------------


def build_skeleton_from_top_simplices(simplex_array: np.ndarray) -> List[List[Tuple[int, ...]]]:
    """
    Build the full skeleton (list of k-face lists) from top-dimensional simplices.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1). The complex is assumed pure.

    Returns
    -------
    List[List[Tuple[int, ...]]]
        faces_by_dim[k] is a list of sorted k-faces (as tuples of vertex indices),
        for k = 0..d.
    """
    simplices = np.asarray(simplex_array, dtype=int)
    d_dim = simplices.shape[1] - 1
    faces_by_dim: List[List[Tuple[int, ...]]] = []
    for k_dim in range(d_dim + 1):
        faces_k = unique_k_faces_from_simplices(simplices, k_dim)
        faces_by_dim.append(faces_k)
    return faces_by_dim



def build_segments_from_edge_indices(points: np.ndarray, edge_indices: np.ndarray) -> np.ndarray:
    """
    Construct geometric segments from a vertex array and an edge-index list.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (num_vertices, dim). Vertex coordinates in R^dim.
    edge_indices : np.ndarray
        Array of shape (num_edges, 2) with vertex index pairs (i, j).

    Returns
    -------
    np.ndarray
        Array of shape (num_edges, 2, dim). Each row contains the two endpoints of a segment.
    """
    points = np.asarray(points)
    edge_indices = np.asarray(edge_indices, dtype=int)
    return np.stack([points[edge_indices[:, 0]], points[edge_indices[:, 1]]], axis=1)


def unique_edges_from_simplices(simplices: np.ndarray) -> np.ndarray:
    """
    Extract the unique undirected edges (1-faces) from a list of simplices.

    Parameters
    ----------
    simplices : np.ndarray
        Integer array of shape (num_simplices, simplex_vertex_count).
        Each row lists vertex indices of one simplex (e.g., triangles → 3, tets → 4, etc.).

    Returns
    -------
    np.ndarray
        Array of shape (num_edges, 2) with sorted vertex index pairs for unique edges.
    """
    simplices = np.asarray(simplices, dtype=int)
    edge_set = set()
    for simplex_vertices in simplices:
        for vertex_i, vertex_j in combinations(simplex_vertices, 2):
            edge_set.add(tuple(sorted((vertex_i, vertex_j))))
    if not edge_set:
        return np.empty((0, 2), dtype=int)
    return np.array(sorted(edge_set), dtype=int)


def unique_faces_from_simplices(simplices: np.ndarray, face_dimension: int) -> np.ndarray:
    """
    Extract unique k-faces (by vertex indices) from a list of simplices, for any k.

    Parameters
    ----------
    simplices : np.ndarray
        Integer array of shape (num_simplices, simplex_vertex_count).
        Each simplex row lists its vertex indices.
    face_dimension : int
        Desired face dimension k (e.g., k=1 for edges, k=2 for triangles, ...).
        Must satisfy 0 <= k <= simplex_dim, where simplex_dim = simplex_vertex_count - 1.

    Returns
    -------
    np.ndarray
        Array of shape (num_faces, k+1) containing sorted vertex index tuples (as rows).
        Returns an empty array with shape (0, k+1) if no faces are found.
    """
    simplices = np.asarray(simplices, dtype=int)
    if simplices.ndim != 2:
        raise ValueError("simplices must be a 2D integer array: (num_simplices, simplex_vertex_count).")

    simplex_vertex_count = simplices.shape[1]
    simplex_dim = simplex_vertex_count - 1
    if not (0 <= face_dimension <= simplex_dim):
        raise ValueError(f"face_dimension must be in [0, {simplex_dim}] for these simplices.")

    face_vertex_count = face_dimension + 1
    face_set = set()

    for simplex_vertices in simplices:
        for face_vertices in combinations(simplex_vertices, face_vertex_count):
            face_set.add(tuple(sorted(face_vertices)))

    if not face_set:
        return np.empty((0, face_vertex_count), dtype=int)
    return np.array(sorted(face_set), dtype=int)



points = np.random.rand(10, 5)       # 10 vertices in R^5 (example)
simplices = np.array([[0,1,3,6], [1,2,3, 5], [1,3,4,5]])  # triangles (2-simplices) as indices

# Get all unique edges (1-faces) and build segments in R^5:
edges = unique_edges_from_simplices(simplices)      # (E,2)
segments_nd = build_segments_from_edge_indices(points, edges)     # (E,2,5)

# If you also want all triangular faces (for 3D rendering):
tri_faces = unique_faces_from_simplices(simplices, face_dim=2)  # (F,3)







# ------------------------------------------------------------
# Faces & incidence
# ------------------------------------------------------------

def unique_k_faces_from_simplices(simplex_array: np.ndarray, k: int) -> List[Tuple[int, ...]]:
    """
    Extract all unique k-faces from a pure d-dimensional simplicial complex.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array of shape (num_simplices, d+1). Each row lists the vertex
        indices of one d-simplex. The complex is assumed pure (all rows same length).
    k : int
        Face dimension, 0 <= k <= d. For example:
        - k=0 → vertices
        - k=1 → edges
        - ...
        - k=d → the d-simplices themselves

    Returns
    -------
    List[Tuple[int, ...]]
        Sorted unique k-faces, each as a sorted tuple of vertex indices, in lexicographic order.
    """
    simplices = np.asarray(simplex_array, dtype=int)
    if simplices.ndim != 2:
        raise ValueError("simplex_array must be 2D (num_simplices, d+1).")
    d = simplices.shape[1] - 1
    if not (0 <= k <= d):
        raise ValueError(f"k must satisfy 0 <= k <= d (d={d}).")

    face_vertex_count = k + 1
    face_set = set()

    for simplex_vertices in simplices:
        for face_vertices in combinations(simplex_vertices, face_vertex_count):
            face_set.add(tuple(sorted(face_vertices)))

    return sorted(face_set)


def count_face_incidences(simplex_array: np.ndarray, codimension: int = 1) -> Counter:
    """
    Count how many d-simplices meet at each (d - codimension)-face.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array of shape (num_simplices, d+1) describing d-simplices.
    codimension : int, optional
        Which face dimension to count incidences for, measured as a codimension from d.
        - codimension=1 → (d-1)-faces (e.g., triangle neighbors in 3D; edges in 2D)
        - codimension=2 → (d-2)-faces, etc.

    Returns
    -------
    collections.Counter
        Maps each face (as a sorted tuple of vertex indices) to the number of incident d-simplices.
    """
    simplices = np.asarray(simplex_array, dtype=int)
    if simplices.ndim != 2:
        raise ValueError("simplex_array must be 2D (num_simplices, d+1).")

    d = simplices.shape[1] - 1
    target_face_dim = d - codimension
    if not (0 <= target_face_dim <= d):
        raise ValueError(f"Invalid codimension={codimension} for d={d}.")

    face_vertex_count = target_face_dim + 1
    incidence_counter: Counter = Counter()

    for simplex_vertices in simplices:
        for face_vertices in combinations(simplex_vertices, face_vertex_count):
            face = tuple(sorted(face_vertices))
            incidence_counter[face] += 1

    return incidence_counter




# ------------------------------------------------------------
# Manifold/closedness checks
# ------------------------------------------------------------

def check_closed_manifold_like_boundary(simplex_array: np.ndarray) -> Tuple[bool, Dict[str, List[Tuple[int, ...]]]]:
    """
    'Watertight' test for a putative boundary made of (n-1)-simplices:
    every (n-2)-face must be incident to exactly TWO (n-1)-simplices.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1) where d = n-1.

    Returns
    -------
    ok : bool
        True if every (d-1)-face has incidence 2 and no (d-1)-face has incidence > 2.
    report : dict
        {
          "boundary_faces":   faces with incidence == 1  (holes / punctures),
          "nonmanifold_faces": faces with incidence > 2  (pinches / non-manifold)
        }
    """
    (incidence) = count_face_incidences(simplex_array, codimension=1)

    boundary_faces = [face for face, count in incidence.items() if count == 1]
    nonmanifold_faces = [face for face, count in incidence.items() if count > 2]
    ok = (len(boundary_faces) == 0) and (len(nonmanifold_faces) == 0)

    report = {
        "boundary_faces": boundary_faces,
        "nonmanifold_faces": nonmanifold_faces,
    }
    return ok, report





# ------------------------------------------------------------
# Adjacency & components
# ------------------------------------------------------------

def build_simplex_adjacency_via_codim1(simplex_array: np.ndarray) -> List[List[int]]:
    """
    Build the adjacency graph of d-simplices: two d-simplices are adjacent
    if they share a (d-1)-face.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1).

    Returns
    -------
    adjacency : List[List[int]]
        adjacency[i] lists indices of d-simplices adjacent to simplex i.
    """
    simplices = np.asarray(simplex_array, dtype=int)
    if simplices.ndim != 2:
        raise ValueError("simplex_array must be 2D (num_simplices, d+1).")

    d = simplices.shape[1] - 1
    face_to_simplex_indices: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    # Map each (d-1)-face to all d-simplices that contain it
    for simplex_index, simplex_vertices in enumerate(simplices):
        for face_vertices in combinations(simplex_vertices, d):  # drop one vertex → (d-1)-face
            face = tuple(sorted(face_vertices))
            face_to_simplex_indices[face].append(simplex_index)

    # Build adjacency: simplices that share a face are neighbors
    num_simplices = len(simplices)
    adjacency: List[List[int]] = [[] for _ in range(num_simplices)]

    for simplex_indices in face_to_simplex_indices.values():
        if len(simplex_indices) < 2:
            continue  # faces used by only one simplex don't contribute adjacency
        first = simplex_indices[0]
        for other in simplex_indices[1:]:
            adjacency[first].append(other)
            adjacency[other].append(first)

    return adjacency


def count_connected_components(simplex_array: np.ndarray) -> int:
    """
    Count connected components in the d-simplex adjacency graph
    (adjacent if share a (d-1)-face). Uses recursive DFS (no while-loops).

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1).

    Returns
    -------
    int
        Number of connected components.
    """
    adjacency = build_simplex_adjacency_via_codim1(simplex_array)
    num_simplices = len(adjacency)
    if num_simplices == 0:
        return 0

    visited = [False] * num_simplices

    def dfs(simplex_id: int) -> None:
        """Depth-first search marking all simplices reachable from simplex_id."""
        visited[simplex_id] = True
        for neighbor_id in adjacency[simplex_id]:
            if not visited[neighbor_id]:
                dfs(neighbor_id)

    components = 0
    for simplex_id in range(num_simplices):
        if not visited[simplex_id]:
            components += 1
            dfs(simplex_id)

    return components





# ------------------------------------------------------------
# Euler characteristic
# ------------------------------------------------------------

def euler_characteristic_from_pure_d_simplices(simplex_array: np.ndarray) -> int:
    """
    Compute Euler characteristic chi = sum_{k=0}^d (-1)^k * f_k
    for a pure d-dimensional simplicial complex given by its d-simplices.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1). The complex is assumed pure:
        all simplices have the same dimension d.

    Returns
    -------
    int
        Euler characteristic of the complex.
    """
    simplices = np.asarray(simplex_array, dtype=int)
    if simplices.ndim != 2:
        raise ValueError("simplex_array must be 2D (num_simplices, d+1).")

    d = simplices.shape[1] - 1
    euler_chi = 0

    # Count unique k-faces for k = 0..d
    for k_dim in range(d + 1):
        k_faces = unique_k_faces_from_simplices(simplices, k_dim)
        num_k_faces = len(k_faces)
        euler_chi += ((-1) ** k_dim) * num_k_faces

    return euler_chi

def sphere_euler_characteristic(d):
    # sphere of dimension d has chi = 1 + (-1)^d
    return 1 + ((-1)**d)







# ============================================================
# Boundary matrices
# ============================================================

def build_boundary_matrix(
    k_faces: List[Tuple[int, ...]],
    k_minus_1_faces: List[Tuple[int, ...]],
    modulo_two: bool = False
) -> np.ndarray:
    """
    Construct the boundary matrix ∂_k : C_k → C_{k-1} in the face bases.

    Conventions:
    - k-faces and (k-1)-faces are provided as *sorted* vertex tuples.
    - Orientation is induced by the sorted order: for σ=(v0<...<vk),
      ∂σ = Σ_i (-1)^i σ \ {vi}. Over Z2, signs are ignored.

    Parameters
    ----------
    k_faces : List[Tuple[int, ...]]
        Basis of C_k (column order). Each entry is a sorted k-face.
    k_minus_1_faces : List[Tuple[int, ...]]
        Basis of C_{k-1} (row order). Each entry is a sorted (k-1)-face.
    modulo_two : bool, optional
        If True, build the matrix over F2 (entries in {0,1}). If False, over Z (entries in {-1,0,1}).

    Returns
    -------
    np.ndarray
        Array of shape (num_(k-1)_faces, num_k_faces) representing ∂_k.
    """
    num_rows = len(k_minus_1_faces)
    num_cols = len(k_faces)
    if num_rows == 0 or num_cols == 0:
        return np.zeros((num_rows, num_cols), dtype=int)

    # Map (k-1)-faces to row indices for quick lookup
    face_to_row: Dict[Tuple[int, ...], int] = {face: idx for idx, face in enumerate(k_minus_1_faces)}

    boundary = np.zeros((num_rows, num_cols), dtype=int)

    for col_index, k_face in enumerate(k_faces):
        # k_face is a tuple (v0, v1, ..., vk), sorted
        k_vertices = list(k_face)
        k_dim = len(k_vertices) - 1

        for delete_pos in range(k_dim + 1):
            # Remove vertex at delete_pos to get a (k-1)-face
            face_vertices = tuple(v for j, v in enumerate(k_vertices) if j != delete_pos)
            row_index = face_to_row.get(face_vertices, None)
            if row_index is None:
                continue  # should not happen for a proper complex built from top simplices

            if modulo_two:
                boundary[row_index, col_index] ^= 1  # add 1 mod 2
            else:
                sign = -1 if (delete_pos % 2 == 1) else 1
                boundary[row_index, col_index] += sign

    return boundary


# ============================================================
# Linear algebra helpers
# ============================================================

def rank_mod2(matrix: np.ndarray) -> int:
    """
    Compute rank over F2 (mod 2) using in-place Gaussian elimination with XOR.
    """
    A = (matrix.copy() % 2).astype(np.uint8)
    num_rows, num_cols = A.shape
    pivot_row = 0

    for col in range(num_cols):
        # Find a row with a 1 in this column at or below pivot_row
        pivot_candidates = [r for r in range(pivot_row, num_rows) if A[r, col] == 1]
        if not pivot_candidates:
            continue

        r0 = pivot_candidates[0]
        # Swap rows if needed
        if r0 != pivot_row:
            A[[pivot_row, r0]] = A[[r0, pivot_row]]

        # Eliminate 1s below the pivot
        for r in range(pivot_row + 1, num_rows):
            if A[r, col] == 1:
                A[r, :] ^= A[pivot_row, :]

        pivot_row += 1
        if pivot_row == num_rows:
            break

    return pivot_row  # number of pivots


def smith_normal_form_integer(matrix_int: np.ndarray):
    """
    Wrapper for SymPy's Smith Normal Form. Returns (U, D, V) such that U*A*V = D.
    Raises if sympy is not available.
    """
    if not _HAVE_SYMPY:
        raise ImportError("SymPy is required for integer homology (Z coefficients). Please install sympy.")
    M = sp.Matrix(matrix_int.tolist())
    U, D, V = M.smith_normal_form()
    return U, D, V


# ============================================================
# Homology over F2
# ============================================================

def compute_homology_Z2(simplex_array: np.ndarray) -> Dict[str, object]:
    """
    Compute Betti numbers over F2 for a pure simplicial complex given by its top simplices.

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1). The complex is assumed pure.

    Returns
    -------
    dict
        {
          "betti_numbers": List[int] of length d+1 (β_k for k=0..d),
          "boundary_shapes": List[Tuple[int,int]] for ∂_k shapes,
          "faces_by_dim": List[List[Tuple[int,...]]] (the skeleton),
        }
    """
    faces_by_dim = build_skeleton_from_top_simplices(simplex_array)
    top_dim = len(faces_by_dim) - 1

    # Pre-build boundary matrices ∂_k : C_k → C_{k-1}
    boundaries: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    for k_dim in range(1, top_dim + 1):
        d_matrix = build_boundary_matrix(
            k_faces=faces_by_dim[k_dim],
            k_minus_1_faces=faces_by_dim[k_dim - 1],
            modulo_two=True
        )
        boundaries.append(d_matrix)
        shapes.append(d_matrix.shape)

    betti_numbers: List[int] = []
    for k_dim in range(0, top_dim + 1):
        num_k_faces = len(faces_by_dim[k_dim])
        rank_dk = rank_mod2(boundaries[k_dim - 1]) if k_dim >= 1 else 0
        rank_dkplus = rank_mod2(boundaries[k_dim]) if k_dim <= top_dim - 1 else 0
        beta_k = (num_k_faces - rank_dk) - rank_dkplus
        betti_numbers.append(int(beta_k))

    return {
        "betti_numbers": betti_numbers,
        "boundary_shapes": shapes,         # for k=1..d
        "faces_by_dim": faces_by_dim,
    }


# ============================================================
# Homology over Z (Betti + torsion)
# ============================================================

def compute_homology_Z(simplex_array: np.ndarray) -> Dict[str, object]:
    """
    Compute simplicial homology over Z using Smith Normal Form.

    Algorithm (for each k):
      - Let A_k   = ∂_k : Z^{n_k} → Z^{n_{k-1}}
      - Let A_{k+1} = ∂_{k+1} : Z^{n_{k+1}} → Z^{n_k}
      - Compute SNF: U_k * A_k * V_k = D_k (diag d1..dr, 0..0), rank r_k
      - Kernel lattice of ∂_k is spanned by columns of V_k corresponding to zero
        diagonal entries (last n_k - r_k columns).
      - Express im(∂_{k+1}) in this V_k-basis: B = V_k^{-1} * A_{k+1}
      - Project to the kernel block: B_ker = B[r_k : n_k, :]
      - SNF of B_ker gives diagonal entries t1..ts (nonzero):
            H_k ≅ Z^{(n_k - r_k) - s}  ⊕  ⊕_{i=1..s} Z_{t_i}
        (Only t_i > 1 contribute torsion.)

    Parameters
    ----------
    simplex_array : np.ndarray
        Integer array (num_simplices, d+1). The complex is assumed pure.

    Returns
    -------
    dict
        {
          "betti_numbers": List[int] (β_k for k=0..d),
          "torsion": List[List[int]] (torsion invariants per k, empty if none),
          "faces_by_dim": List[List[Tuple[int,...]]],
          "boundary_shapes": List[Tuple[int,int]],  # for k=1..d
        }
    """
    if not _HAVE_SYMPY:
        raise ImportError("SymPy is required for integer homology (Z coefficients). Please install sympy.")

    faces_by_dim = build_skeleton_from_top_simplices(simplex_array)
    top_dim = len(faces_by_dim) - 1

    # Build integer boundary matrices
    boundaries_Z: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    for k_dim in range(1, top_dim + 1):
        d_matrix = build_boundary_matrix(
            k_faces=faces_by_dim[k_dim],
            k_minus_1_faces=faces_by_dim[k_dim - 1],
            modulo_two=False
        )
        boundaries_Z.append(d_matrix.astype(int))
        shapes.append(d_matrix.shape)

    betti_numbers: List[int] = []
    torsion_invariants: List[List[int]] = []

    # Precompute SNF of all A_k
    snf_data: List[Tuple[sp.Matrix, sp.Matrix, sp.Matrix, int]] = []  # (U_k, D_k, V_k, r_k)
    for k_dim in range(0, top_dim + 1):
        if k_dim == 0:
            # ∂_0 is the zero map: rank = 0, V_0 = I, D_0 = 0
            n_k = len(faces_by_dim[0])
            U0 = sp.eye(len(faces_by_dim[-1]) if top_dim >= 0 else 0)  # unused
            D0 = sp.zeros(len(faces_by_dim[-1]) if top_dim >= 0 else 0)
            V0 = sp.eye(n_k)
            r0 = 0
            snf_data.append((U0, D0, V0, r0))
            continue

        A_k = boundaries_Z[k_dim - 1]  # shape: (n_{k-1}, n_k)
        U_k, D_k, V_k = smith_normal_form_integer(A_k)
        diag_vals = [int(D_k[i, i]) for i in range(min(D_k.shape))]
        r_k = sum(1 for v in diag_vals if v != 0)
        snf_data.append((U_k, D_k, V_k, r_k))

    # Compute H_k using the algorithm described above
    for k_dim in range(0, top_dim + 1):
        n_k = len(faces_by_dim[k_dim])

        # Data for ∂_k
        _, D_k, V_k, r_k = snf_data[k_dim]
        V_k_inv = V_k.inv()  # unimodular over Z

        # Matrix for ∂_{k+1}
        if k_dim <= top_dim - 1:
            A_kplus = boundaries_Z[k_dim]  # shape: (n_k, n_{k+1})
            # Express im(∂_{k+1}) in V_k coordinates
            B = V_k_inv * sp.Matrix(A_kplus.tolist())
            # Project onto kernel block (rows r_k : n_k)
            if n_k - r_k > 0:
                B_ker = B[r_k:n_k, :]
                # SNF of B_ker
                U, D, V = B_ker.smith_normal_form()
                diag_vals = [int(D[i, i]) for i in range(min(D.shape))]
                nonzero = [abs(v) for v in diag_vals if v != 0]
                s = len(nonzero)  # rank over Z of image inside kernel
                torsion_k = [t for t in nonzero if t > 1]
                beta_k = (n_k - r_k) - s
            else:
                # Kernel is trivial
                torsion_k = []
                beta_k = 0
        else:
            # Top dimension: H_d = ker(∂_d)
            torsion_k = []
            beta_k = n_k - r_k

        betti_numbers.append(int(beta_k))
        torsion_invariants.append(torsion_k)

    return {
        "betti_numbers": betti_numbers,
        "torsion": torsion_invariants,
        "faces_by_dim": faces_by_dim,
        "boundary_shapes": shapes,  # for k=1..d
    }




# ============================================================
# Convenience wrapper
# ============================================================

def compute_simplicial_homology(
    simplex_array: np.ndarray,
    coefficients: str = "Z2"
) -> Dict[str, object]:
    """
    Compute simplicial homology of a pure complex given by its top-dimensional simplices.

    Parameters
    ----------
    simplex_array : np.ndarray
        (num_simplices, d+1) integer array listing the vertex indices of each d-simplex.
        The complex is assumed pure; all faces of top simplices are considered present.
    coefficients : {"Z2", "Z"}
        Field/ring of coefficients. "Z2" is fast and orientation-free.
        "Z" requires sympy and returns torsion invariants as well.

    Returns
    -------
    dict
        If coefficients == "Z2":
            { "betti_numbers": List[int], "boundary_shapes": List[Tuple[int,int]], "faces_by_dim": ... }
        If coefficients == "Z":
            { "betti_numbers": List[int], "torsion": List[List[int]], "boundary_shapes": ..., "faces_by_dim": ... }

    Notes
    -----
    - For general (non-pure) complexes where lower-dimensional simplices are not necessarily
      faces of top simplices, you can extend this by accepting an explicit skeleton (faces_by_dim).
    - Orientation is induced by sorted vertex order; over Z2, signs are ignored.
    """
    if coefficients.upper() == "Z2":
        return compute_homology_Z2(simplex_array)
    if coefficients.upper() == "Z":
        return compute_homology_Z(simplex_array)
    raise ValueError('coefficients must be one of {"Z2", "Z"}')