import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import numpy as np
import numpy as np
import numpy as np
# Import the necessary libraries
import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, solve_triangular
import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, solve_triangular
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

import torch
import pyro
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve, cho_factor


# Function g(x)
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve



import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return -np.sin(6 * np.pi * x) ** 2 + 6 * x ** 2 - 5 * x ** 4 + 3 / 2

def kernel(x1, x2, length_scale=0.2, sigma_f=1.0):
    dist = np.abs(np.subtract.outer(x1, x2))  # Absolute distance between points
    return sigma_f ** 2 * (1 + np.sqrt(3) * dist / length_scale) * np.exp(-np.sqrt(3) * dist / length_scale)

def trapezoidal_weights(l):
    weights = np.ones(l) / (l - 1)
    weights[0] /= 2
    weights[-1] /= 2
    return weights

def gp_with_integral_constraint(x, q, length_scale, sigma_f, sigma_n):
    K = kernel(x, x, length_scale, sigma_f) + sigma_n ** 2 * np.eye(len(x))
    w = trapezoidal_weights(len(x))
    W = np.outer(w, w) * K  # Used for debugging or further calculations if needed
    Kw = K @ w
    wKw = w @ Kw
    # Adjusted covariance and mean with scaling by q
    K_cond = K - np.outer(Kw, Kw) / wKw
    mean_cond = Kw / wKw * q
    return mean_cond, K_cond

def plot_samples_for_q(mean, cov, x, q, title):
    plt.figure(figsize=(10, 5))
    scaled_mean = mean * q
    samples = np.random.multivariate_normal(scaled_mean, cov, size=5)
    for sample in samples:
        plt.plot(x, sample, alpha=0.6)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

if __name__ == "__main__":
    l = 101  # Number of grid points
    x = np.linspace(0, 1, l)
    length_scale = 0.2
    sigma_f = 1.0
    sigma_n = 0.1
    mean_cond, cov_cond = gp_with_integral_constraint(x, q=1, length_scale=length_scale,
                                                      sigma_f=sigma_f, sigma_n=sigma_n)
    # Plot for q = 0
    plot_samples_for_q(mean_cond, cov_cond, x, q=0, title='Constrained GP Samples for q=0')
    # Plot for q = 5
    plot_samples_for_q(mean_cond, cov_cond, x, q=5, title='Constrained GP Samples for q=5')
    # Plot for q = 10
    plot_samples_for_q(mean_cond, cov_cond, x, q=10, title='Constrained GP Samples for q=10')

# Implementation of the GP posterior computation and visualization using provided dataset D
