# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/branch-datasets.ipynb.

# %% auto 0
__all__ = ['random_polynomial', 'Stick', 'Branch']

# %% ../../nbs/library/branch-datasets.ipynb 3
import sympy as sp
import numpy as np
import itertools

def random_polynomial(
        vars, # variables to construct polynomial from
        degree = 2, # maximum degree of terms
):
    num_variables = len(vars)
    terms = []
    for d in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(num_variables), d):
            terms.append(np.prod([vars[i] for i in indices]))
    coeffs = np.random.normal(size = len(terms))
    return sum([coeffs[i] * terms[i] for i in range(len(terms))])


# %% ../../nbs/library/branch-datasets.ipynb 4
from fastcore.all import *
import sympy as sp

class Stick():
    def __init__(
        self,
        dimension,
        degree,
        start_point,
        time_range = 1
    ):
        store_attr()
        # construct a unique polynomial for yourself
        x = sp.symbols('x')
        p = random_polynomial(
            [x],degree
        )
        self.polynomial = p
        self.polynomial_np = sp.lambdify([x], self.polynomial, "numpy")

        # random direction for polynomial, scaled to unit length
        self.direction = np.random.randn(self.dimension)
        self.direction /= np.linalg.norm(self.direction)

    def sample_at_time(self,t):
        return self.polynomial_np(self.direction*t) + self.start_point

    def end_point(self):
        return self.sample_at_time(self.time_range)

    def sample(self, n_samples):
        ts = np.random.rand(n_samples)*self.time_range
        Xs = [self.sample_at_time(t) for t in ts]
        return np.array(Xs)
        
    def length(self):
        # integrate the polynomial over the time range
        # using sympy
        x = sp.symbols('x')
        # path length integrand
        integrand = sp.sqrt(1 + sp.diff(self.polynomial, x)**2)
        integral = sp.integrate(
            integrand, (x, 0, self.time_range))
        return float(integral)

# %% ../../nbs/library/branch-datasets.ipynb 7
import random
import numpy as np
import torch
class Branch():
    def __init__(
        self,
        dimension,
        polynomial_degree=4,
        max_branches=5,
        path_length = 5,
        seed = None,
    ):
        store_attr()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.sticks = [
            Stick(
                dimension,
                polynomial_degree,
                np.zeros(dimension)
            )
        ]
        self.branching_nums = [np.random.randint(1,max_branches)]
        self.num_branches_per_point = []
        self.branch_lengths = []

        stick_idx = 0
        for i in range(path_length):
            # go through all sticks after stick_idx and create new sticks at their ends
            new_stick_idx =len(self.sticks)
            for j in range(stick_idx,len(self.sticks)):
                num_new_sticks = self.branching_nums[j]
                for k in range(num_new_sticks):
                    # create a stick
                    stick = Stick(
                        dimension,
                        polynomial_degree,
                        self.sticks[j].end_point(),
                    )
                    self.sticks.append(stick)
                    self.branching_nums.append(
                        np.random.randint(1,max_branches)
                    )
            stick_idx = new_stick_idx
        self.branching_nums = np.array(self.branching_nums)

    def sample(self,n_samples=5000):
        Xs = []
        samples_per_stick = n_samples // len(self.sticks)
        for i, stick in enumerate(self.sticks):
            Xs.append(np.vstack([stick.sample(samples_per_stick-1),stick.end_point()]))
            self.num_branches_per_point.append(np.append(np.zeros(n_samples-1), self.branching_nums[i]))
            self.branch_lengths.append(
                np.ones(samples_per_stick)*stick.length()
            )
        self.num_branches_per_point = np.concatenate(self.num_branches_per_point)
        self.branch_lengths = np.concatenate(self.branch_lengths)
        return np.concatenate(Xs,axis=0)