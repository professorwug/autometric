# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/0d2 Some Example Datasets.ipynb.

# %% auto 0
__all__ = ['spiral', 'petals', 'moon_petals', 'sprial_petals', 'sinewave_petals', 'make_swiss_roll', 'generate_sine_wave_dataset']

# %% ../nbs/0d2 Some Example Datasets.ipynb 0
import numpy as np

def spiral(N, **kwargs):
    A=1
    omega=1
    phi=0
    t_min=0
    t_max=10
    """
    Generate a dataset of a spiral.

    Parameters:
    - N: Number of points
    - A: Amplitude
    - omega: Angular frequency
    - phi: Phase
    - t_min: Minimum time value
    - t_max: Maximum time value

    Returns:
    - X: 2D array of shape (N, 2), where each row is a point [x(t), y(t)]
    - t: 1D array of time values
    """
    # Generate time values
    t = np.linspace(t_min, t_max, N)
    
    # Generate the spiral points using the parametric equations
    x = A * t * np.cos(omega * t + phi)
    y = A * t * np.sin(omega * t + phi)
    
    # Combine x and y coordinates into a single 2D array
    X = np.column_stack((x, y))
    
    return X, t

def petals(N, **kwargs):
    """Generate petal data set."""
    X = []  # points in respective petals
    Y = []  # auxiliary array (points on outer circle)
    C = []

    assert N > 4, "Require more than four data points"

    # Number of 'petals' to point into the data set. This is required to
    # ensure that the full space is used.
    M = int(np.floor(np.sqrt(N)))
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)

    for theta in thetas:
        Y.append(np.asarray([np.cos(theta), np.sin(theta)]))

    # Radius of the smaller cycles is half of the chord distance between
    # two 'consecutive' points on the circle.
    r = 0.5 * np.linalg.norm(Y[0] - Y[1])

    for i, x in enumerate(Y):
        for theta in thetas:
            X.append(np.asarray([r * np.cos(theta) - x[0], r * np.sin(theta) - x[1]]))

            # Indicates that this point belongs to the $i$th circle.
            C.append(i)

    return np.asarray(X), np.asarray(C)

def moon_petals(N, degree=np.pi, R=1, **kwargs):
    """Generate moon petal data set."""
    X = []  # points in respective petals
    Y = []  # auxiliary array (points on outer circle)
    C = []

    assert N > 4, "Require more than four data points"

    # Number of 'petals' to point into the data set. This is required to
    # ensure that the full space is used.
    M = int(np.floor(np.sqrt(N)))
    thetas_outer = np.linspace(0, degree, M, endpoint=False)
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)


    for theta in thetas_outer:
        Y.append(np.asarray([np.cos(theta), np.sin(theta)]))

    # Radius of the smaller cycles is half of the chord distance between
    # two 'consecutive' points on the circle.
    r = 0.5 * np.linalg.norm(Y[0] - Y[1])

    for i, x in enumerate(Y):
        for theta in thetas:
            X.append(np.asarray([r * np.cos(theta) - x[0], r * np.sin(theta) - x[1]]))

            # Indicates that this point belongs to the $i$th circle.
            C.append(i)

    return np.asarray(X), np.asarray(C)



def sprial_petals(turns=2, a=0.1, base_delta_theta=0.1 * 2 * np.pi, points_per_unit_radius=100):
    """
    Generate points forming tangent circles along a spiral with even distribution of spiral points.
    
    Parameters:
    - turns: Number of turns for the spiral.
    - a: Parameter controlling the distance between successive turns of the spiral.
    - base_delta_theta: Base angular increment for the spiral points at theta = 2*pi.
    - points_per_unit_radius: Number of points per unit radius for the circle.
    
    Returns:
    - A numpy array containing the points forming the tangent circles.
    """
    
    # Generate points on the spiral with varying delta_theta
    def generate_even_spiral_points(turns, a, base_delta_theta):
        theta = 0.0
        spiral_points = []
        
        while theta < 2 * np.pi * turns:
            x = a * theta * np.cos(theta)
            y = a * theta * np.sin(theta)
            spiral_points.append([x, y])

            # Adjust delta_theta to be inversely proportional to theta, but ensure it's never zero
            delta_theta = base_delta_theta / (1 + theta / (2 * np.pi))
            theta += delta_theta
        
        return np.array(spiral_points)

    # Generate points on a circle
    def generate_circle_points(center, r, num_points):
        thetas = np.linspace(0, 2 * np.pi, num_points)
        x = r * np.cos(thetas) + center[0]
        y = r * np.sin(thetas) + center[1]
        return np.column_stack((x, y))
    
    spiral_points = generate_even_spiral_points(turns, a, base_delta_theta)
    circle_points_list = []

    for i in range(1, len(spiral_points) - 1):
        prev_point = spiral_points[i - 1]
        curr_point = spiral_points[i]
        next_point = spiral_points[i + 1]

        # Calculate radius as half the average distance to adjacent points
        r = 0.25 * (np.linalg.norm(curr_point - prev_point) + np.linalg.norm(next_point - curr_point))

        # Number of points for this circle proportional to its radius
        num_circle_points = int(r * points_per_unit_radius)

        # Generate circle points and append to the list
        circle_points = generate_circle_points(curr_point, r, num_circle_points)
        circle_points_list.append(circle_points)

    # Combine all circle points into a single array
    all_circle_points = np.vstack(circle_points_list)
    
    return all_circle_points, None


def sinewave_petals(length=8, A=2, B=1, C=0, delta_t=0.3, points_per_unit_radius=50):
    """
    Generate points forming tangent circles along a sine wave in the xy-plane with constant delta_t.
    
    Parameters:
    - length: Length of the sine wave along x-axis.
    - A: Amplitude of the sine wave.
    - B: Determines the period of the wave.
    - C: Phase shift of the sine wave.
    - delta_t: Constant increment for t.
    - points_per_unit_radius: Number of points per unit radius for the circle.
    
    Returns:
    - A numpy array containing the points forming the tangent circles.
    """
    
    # Generate points on the sine wave with constant delta_t
    def generate_even_sine_wave_points(length, A, B, C, delta_t):
        t_values = np.arange(0, length, delta_t)
        x_values = t_values
        y_values = A * np.sin(B * t_values + C)
        return np.column_stack((x_values, y_values))

    # Generate points on a circle
    def generate_circle_points(center, r, num_points):
        thetas = np.linspace(0, 2 * np.pi, num_points)
        x = r * np.cos(thetas) + center[0]
        y = r * np.sin(thetas) + center[1]
        return np.column_stack((x, y))
    
    sine_wave_points = generate_even_sine_wave_points(length, A, B, C, delta_t)
    circle_points_list = []

    for i in range(1, len(sine_wave_points) - 1):
        prev_point = sine_wave_points[i - 1]
        curr_point = sine_wave_points[i]
        next_point = sine_wave_points[i + 1]

        # Calculate radius as half the average distance to adjacent points
        r = 0.25 * (np.linalg.norm(curr_point - prev_point) + np.linalg.norm(next_point - curr_point))

        # Number of points for this circle proportional to its radius
        num_circle_points = int(r * points_per_unit_radius)

        # Generate circle points and append to the list
        circle_points = generate_circle_points(curr_point, r, num_circle_points)
        circle_points_list.append(circle_points)

    # Combine all circle points into a single array
    all_circle_points = np.vstack(circle_points_list)
    
    return all_circle_points, None

def make_swiss_roll(turns=2, a=0.1, base_delta_theta=0.1 * 2 * np.pi, theta=1 * np.pi, noise=0.0):
    thetas = []
    while theta < 2 * np.pi * turns:
        thetas.append(theta)
        # Adjust delta_theta to be inversely proportional to theta, but ensure it's never zero
        delta_theta = base_delta_theta / (1 + theta / (2 * np.pi))
        theta += delta_theta
    thetas = np.array(thetas)
    tmp_x = (a * thetas * np.cos(thetas))
    xmin, xmax = np.min(tmp_x), np.max(tmp_x)
    w = np.linspace(xmin, xmax, len(thetas))
    t, w = np.meshgrid(thetas, w)
    t = t.flatten()
    w = w.flatten()
    x = a * t * np.cos(t)
    y = a * t * np.sin(t)
    z = w
    noises_t = np.random.normal(0, noise * (t.max()-t.min()), len(t))
    noises_w = np.random.normal(0, noise * (w.max()-w.min()), len(w))
    t += noises_t
    w += noises_w
    return np.column_stack((x, y, z)), np.column_stack((t, w))

def generate_sine_wave_dataset(num_points=50, amplitude=1, frequencies=(1, 1)):
    """
    Generates a dataset of points in 3D space where the Z value is a combination
    of sine waves of X and Y.

    Parameters:
    num_points (int): Number of points to generate.
    amplitude (float): The peak deviation of the sine wave from zero.
    frequencies (tuple): The frequency of the sine wave cycles per unit of distance
                         for X and Y respectively.

    Returns:
    dataset (numpy array): A num_points x 3 array where each row represents the X, Y, and Z
                           coordinates of a point on the sine wave surface.
    """
    # Generate random values for x and y within the range [0, 2*pi]
    x = np.random.uniform(0, 2 * np.pi, num_points)
    y = np.random.uniform(0, 2 * np.pi, num_points)

    # Calculate the z values using the sine function for both x and y
    z = amplitude * (np.sin(frequencies[0] * x) + np.sin(frequencies[1] * y))

    # Combine x, y, and z into a single dataset
    dataset = np.vstack((x, y, z)).T

    return dataset
