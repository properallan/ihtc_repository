
from scipy.special import comb
import numpy as np

def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        #M_ = np.linalg.pinv(M)
        x, _, _, _ = np.linalg.lstsq(M, points)
        return x
        #return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final

def find_control_points(points, n):
    return np.array(get_bezier_parameters(points[:,0],points[:,1],n-1))

def bezier_curve(control_points, num_points=100):
    # Make sure the input array is a numpy array
    control_points = np.array(control_points)
    
    # Number of control points
    n = len(control_points)
    
    # Construct the parameter vector t
    t = np.linspace(0, 1, num_points)
    
    # Construct the matrix of Bernstein basis functions
    bernstein = np.zeros((n, num_points))
    for i in range(n):
        bernstein[i,:] = np.power(t, i) * np.power(1-t, n-i-1) * np.math.factorial(n-1) / (np.math.factorial(i) * np.math.factorial(n-i-1))
    
    # Compute the curve points
    curve_points = np.dot(control_points.T, bernstein)
    
    return curve_points.T