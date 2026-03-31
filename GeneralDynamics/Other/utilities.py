import numpy as np


def cartesianProduct(range_list):
    """
    Computes the Cartesian product of a list of lists (1D iterables).
    Returns a NumPy array of shape (total_points, num_ranges)
    without using meshgrid or itertools.
    """
    num_ranges = len(range_list)
    if num_ranges == 0:
        return np.empty((0, 0))

    row_sizes = [len(row) for row in range_list]
    total_points = np.prod(row_sizes, dtype=int)
    print(total_points)
    # Prepare output array
    product_array = np.zeros((total_points, num_ranges), dtype=object)

    # Fill each column explicitly
    repeat_block = total_points 
    for row_index, row in enumerate(range_list):
        repeat_block //= row_sizes[row_index]
        pattern = np.repeat(row, repeat_block)   # repeat each element
        product_array[:, row_index] = np.tile(pattern, total_points // (len(pattern)))  # tile pattern
    return product_array



x_range = list(range(-1, 4))
y_range = list(range(0, 3))
z_range = list(range(0, 3))

lists = []
lists.append(x_range)
lists.append(y_range)
lists.append(z_range)

print("list:", lists)
print("list first row(x_range):" ,lists[0][:])
print(cartesianProduct(lists))





