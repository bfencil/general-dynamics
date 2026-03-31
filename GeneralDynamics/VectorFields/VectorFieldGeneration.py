import numpy as np
import random
from GeneralDynamics.Other.utilities import cartesianProduct

# lambda function inputs, with domain 
def simpleVectorField(vector_valued_function, domain_boundary_points, total_samples, random_samples=False):
    """
    Takes in a vector valued function(over R^n) and returns an array of 
    vectors in the given sub-domain(as defined by the boundary points) 

    Inputs:  
        vector_valued_function: vector valued lambda function V:R^n(positions) to R^n(vectors)
        domain_boundary_points: array of points in R^n for that lie on the boundary of 
                    the desired domain, dimensions of should follow [number_points, points_dimension] 
        total_samples: the amount of positions that will be sampled in the domain 
        random_sample: if True then points will be randomely sampled from in the domain
            if False then points will be sampled along the grid
    """
    data_dimension = domain_boundary_points.shape[1]
    if data_dimension == 0:
        raise ValueError("Error: Insufficient dimension of domain boundary points")

    #function dimension test
    if len(vector_valued_function(domain_boundary_points[0, :])) != data_dimension:
        raise ValueError("Error: Vector valued functions domain is of the incorrect dimension for the given domain")

    # each row is the min-max interval for each row of domain_boundary_points 
    domain_bounds = np.array([np.min(domain_boundary_points, axis=0 ) , np.max(domain_boundary_points, axis=0)]).T

    if random_samples:
        0

    else:
        domain_ranges = [[]]

        

    return 0


# specifcy simple vector fields for given sub-domains
# non perfecting areas will interpolate between the given SVF's
def piecewiseVectorField():

    return 0

# Takes in a vector field, along with a lambda function that will 
# be used to weight interpolation of the vector field itself,
# specifically, given a point p we look at the nearest k points to p
# and update the vector at p based on averaging the vectors of the k nearest points(including p)
# as a weighted sum of the lambda function evaulating the components of the each position where
# the vectors are defined
def nonlinearWeightedLocalVectorFieldInterpolation():

    return 0
