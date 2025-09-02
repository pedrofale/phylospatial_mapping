import numpy as np

def diag_equation(x, y, slope, offset):
    return slope*(x-y)+offset

def circ_equation(x, y, radius, center_x, center_y):
    return (x-center_x)**2 + (y-center_y)**2 - radius**2

def sinusoidal_curve_equation(x, y, period, amplitude=1.0, phase=0.0, y_offset=0.0):
    """
    Curve form: y = y_offset + amplitude * sin(2π x / period + phase)
    Returns the residual y - y_sin so 'prior' emphasizes the curve.
    """
    y_sin = y_offset + amplitude * np.sin(2 * np.pi * x / period + phase)
    return y - y_sin


def prior(grid_size, equation, decay_factor=1, **kwargs):
    """
    Generate a grid of values that are 1/distance^decay_factor from the equation.
    The equation is a function of x and y, and the grid is a 2D array of size grid_size x grid_size.
    The decay_factor is a scalar that controls the decay of the prior.
    """
    mat = np.zeros((grid_size, grid_size))  
    for x in range(grid_size):
        for y in range(grid_size):
            mat[x, y] = np.abs((equation(x, y, **kwargs)))
    mat = np.exp(-mat*decay_factor)

    return mat

def map_points_to_grid_lowerleft(x, y, xmin, ymin, dx, dy):
    """
    Map scatter points to the lower-left corner of their grid cell.

    Args:
        x, y   : arrays of point coordinates
        xmin.., ymin.. : grid bounds
        dx, dy : cell size

    Returns:
        gx, gy : coordinates of the lower-left corner of the cell
    """
    ix = np.floor((x - xmin) / dx).astype(int)
    iy = np.floor((y - ymin) / dy).astype(int)

    gx = xmin + ix * dx
    gy = ymin + iy * dy

    return gx, gy

mat = prior(200, diag_equation, decay_factor=.01, slope=10, offset=0)

mat = prior(200, circ_equation, decay_factor=.001, radius=50, center_x=100, center_y=100)

mat = prior(200, sinusoidal_curve_equation, decay_factor=.1, period=100, amplitude=10.0, phase=0.0, y_offset=100.0)
