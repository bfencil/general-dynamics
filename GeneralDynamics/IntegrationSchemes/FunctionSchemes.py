import numpy as np
import math
import inspect


f = lambda x, y, z, g: x+y


print(f.__code__.co_argcount)



def NumericalIntegration(func=None, sub_domain=None, sub_divisions=1000):
    """
    General numerical integration scheme for integrating functions of the form f:R^n to R.

    Inputs:
        func: the function to be integrated over. given as a lambda function, default value is None.
        sub_domain: list of points in R^n for which to integrate func:R^n to R over, default is None.
        sub_divisions: the amount of sample points over the given integration region.
    
    Outputs:
        numeric_value: the value of the integration
    """

    if func == None:
        raise ValueError("Function is None. Must input a defined function")
    if sub_domain == None:
        raise ValueError("Sub domain is None. Must specify a region of integration")

    number_points = len(sub_domain)
    number_dimensions = func.__code__.co_argcount

    for p in sub_domain:
        if len(p) != number_dimensions:
            raise ValueError("Inconsistent dimensionality in point ", p, " with respect to the dimensionality of the given function.")

    if number_points < number_dimensions + 1:
        raise ValueError("Additional points needed in sub_domain to minimally specify region. Currently there are only ", number_points, " number of points. At least ", number_dimensions + 1, " is needed.")

    


    if number_dimensions < 4:
        0 # Gauss–Kronrod quadrature here
    else:
        0 # monte carlo method here



    return 0