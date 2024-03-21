# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/manifolds.ipynb.

# %% auto 0
__all__ = ['LOWER_EPSILON', 'BIGGER_LOWER_EPSILON', 'BIGGEST_LOWER_EPSILON', 'UPPER_EPSILON', 'SMALLER_UPPER_EPSILON',
           'RiemannianManifold']

# %% ../../nbs/library/manifolds.ipynb 1
"""
DiffGeo file containing everything related to a manifold
"""

import os
import traceback

import numpy as np
import geomstats.backend as gs


# lower bound for numerical stability
LOWER_EPSILON = 1e-20
BIGGER_LOWER_EPSILON = 1e-12
BIGGEST_LOWER_EPSILON = 1e-10
UPPER_EPSILON = 1e20
SMALLER_UPPER_EPSILON = 1e12


# os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch
from torch.func import jacrev, vmap

from .utils import batch_jacobian

from geomstats.geometry.manifold import Manifold


class RiemannianManifold(Manifold):
    """
    Class for manifolds.

    :param dim : intd
            Dimension of the manifold.
    :param shape : tuple of int
            Shape of one element of the manifold.
            Optional, default : None.
    :param metric : RiemannianMetric
            Metric object to use on the manifold.
    :param default_point_type : str, {\'vector\', \'matrix\'}
            Point type.
            Optional, default: 'vector'.
    :param default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
            Coordinate type.
            Optional, default: 'intrinsic'.
    """

    def __init__(self, dim, shape, metric=None, connection=None, default_point_type=None,
                 default_coords_type="intrinsic", **kwargs):
        super().__init__(dim, shape, metric=metric, default_point_type=default_point_type,
                         default_coords_type=default_coords_type, **kwargs)

        self.connection = connection

    def christoffel_derivative(self, base_point=None):
        """
        Calculate the derivative of the christoffel symbols
        :return: derivative of christoffel
        """

        gamma_derivative = batch_jacobian(self.connection.christoffels, base_point)

        return gamma_derivative

    def metric_det(self, base_point=None):
        """
        Calculate the determinant of the metric matrix at base_point
        :param base_point: the point under consideration
        :param metric_matrix: the metric at point base_point
        :return: the determinant
        """

        metric_matrix = self.metric.metric_matrix(base_point=base_point)

        det = torch.linalg.det(metric_matrix)

        return det
    
    def metric_logdet(self, base_point=None):
        """
        Calculate the log determinant of the metric matrix at base_point
        :param base_points: the points under consideration
        :param metric_matrix: the metric at point base_point
        :return: the determinant
        """

        metric_matrix = self.metric.metric_matrix(base_point=base_point)

        logdet = torch.logdet(metric_matrix)

        return logdet

    def riemannian_curvature_tensor(self, base_point=None):
        """
        Returns the curvature tensor symbols
        :param base_point: the base point
        :return: the coordinates of the curvature tensor, contravariant index in the first dimension
        """

        gamma = self.connection.christoffels(base_point)

        gamma_derivative = self.christoffel_derivative(base_point=base_point)

        term_1 = torch.einsum("...ljki->...lijk", gamma_derivative)
        term_2 = torch.einsum("...likj->...lijk", gamma_derivative)
        term_3 = torch.einsum("...mjk,...lim->...lijk", gamma, gamma)
        term_4 = torch.einsum("...mik,...ljm->...lijk", gamma, gamma)

        R = term_1 - term_2 + term_3 - term_4

        return R

    def riemannian_curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None):
        """
        :param tangent_vec_a:
        :param tangent_vec_b:
        :param tangent_vec_c:
        :param base_point:
        :return:
        """

        R = self.riemannian_curvature_tensor(base_point=base_point)

        s = torch.einsum("...lijk,i->...ljk", R, tangent_vec_a)
        s = torch.einsum("...ljk,j->...lk", s, tangent_vec_b)
        s = torch.einsum("...lk,k->...l", s, tangent_vec_c)

        return s
    

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Compute the sectional curvature
        :param tangent_vec_a: first vector
        :param tangent_vec_b: second vector
        :param base_point: the point under consideration
        :return: sectional curvature at base_point
        """

        if base_point.ndim == 1:
            base_point = torch.unsqueeze(base_point, 0)

        metric = self.metric.metric_matrix(base_point)
        # aab, aba, baa
        # bba, bab, abb
        curvature = self.riemannian_curvature(tangent_vec_a, tangent_vec_b, tangent_vec_b, base_point)

        sectional = self.metric.inner_product(curvature, tangent_vec_a, matrix=metric)

        # norm_a = self.metric.norm(tangent_vec_a, matrix=metric)
        # norm_b = self.metric.norm(tangent_vec_b, matrix=metric)
        norm_a = self.metric.inner_product(tangent_vec_a, tangent_vec_a, matrix=metric)
        norm_b = self.metric.inner_product(tangent_vec_b, tangent_vec_b, matrix=metric)
        inner_ab = self.metric.inner_product(tangent_vec_a, tangent_vec_b, matrix=metric)

        normalization_factor = norm_a * norm_b - inner_ab ** 2

        result = torch.where(normalization_factor != 0, sectional / normalization_factor, torch.zeros_like(sectional))

        return result


    def ricci_tensor(self, base_point):
        r"""Compute Ricci curvature tensor at base_point.

        The Ricci curvature tensor :math:`\mathrm{Ric}_{ij}` is defined as:
        :math:`\mathrm{Ric}_{ij} = R_{ikj}^k`
        with Einstein notation.

        Adapted from [1. What is a Connection? — Geomstats latest documentation](https://geomstats.github.io/notebooks/02_foundations__connection_riemannian_metric.html).

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        ricci_tensor : array-like, shape=[..., dim, dim]
            ricci_tensor[...,i,j] = Ric_{ij}
            Ricci tensor curvature.
        """
        riemann_tensor = self.riemannian_curvature_tensor(base_point)
        ricci_tensor = torch.einsum("...ijkj -> ...ik", riemann_tensor)
        return ricci_tensor


    def scalar_curvature(self, base_point:torch.Tensor):
        # TODO: Smarter ways to parallelize
        gamma = self.connection.christoffels(base_point)
        gamma_derivative = self.christoffel_derivative(base_point=base_point)
        metric_matrix = self.metric.metric_matrix(base_point=base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix).float()
        def final_step(ga, ga_d, commy):
            term_1 = torch.einsum("...lmnl->...lmn", ga_d)
            term_2 = torch.einsum("...lmln->...lmn", ga_d)
            term_3 = torch.einsum("...smn,...lls->...lmn", ga, ga)
            term_4 = torch.einsum("...sml,...lns->...lmn", ga, ga)
            Rprime = term_1 - term_2 + term_3 - term_4
            S = torch.einsum("...mn,lmn->...", commy, Rprime)
            return S
        if len(base_point.size()) == 1:
            S = final_step(gamma, gamma_derivative, cometric_matrix)
        else:
            S = vmap(final_step)(gamma, gamma_derivative, cometric_matrix)
        return S

    def broken_scalar_curvature(self, base_point):
        r"""Compute scalar curvature at base_point.
        Implementation from GeomStats.
        - which has a negative sign messed up somewhere.
        
        In the literature scalar_curvature is noted S and writes:
        :math:`S = g^{ij} Ric_{ij}`,
        with Einstein notation, where we recognize the trace of the Ricci
        tensor according to the Riemannian metric :math:`g`.

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature : array-like, shape=[...,]
            Scalar curvature.
        """
        metric_matrix = self.metric.metric_matrix(base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix).float()
        ricci_tensor = self.ricci_tensor(base_point)
        print(cometric_matrix.dtype, ricci_tensor.dtype)
        return torch.einsum("...ij, ...ij -> ...", cometric_matrix, ricci_tensor)

    def generate_unit_vectors(self, n, base_point):
        """
        calculate polygon lengths using metric
        :param n: number of vectors
        :param base_point: the base point
        :return: array of norms
        """

        # the angles
        phi = torch.linspace(0., 2 * np.pi, n)

        # generate circular vector patch
        raw_vectors = torch.stack([torch.sin(phi), torch.cos(phi)])

        # metric at the point
        metric = self.metric.metric_matrix(base_point)

        # normalize vectors in pullback metric
        norm_vectors = self.metric.norm(raw_vectors, matrix=metric)

        norm_vectors = norm_vectors.unsqueeze(2).expand(*norm_vectors.shape, raw_vectors.shape[0])

        # reshape the raw vectors
        raw_vectors = raw_vectors.unsqueeze(2).expand(*raw_vectors.shape, base_point.shape[0])
        raw_vectors = torch.transpose(raw_vectors, dim0=0, dim1=2)

        # normalize the vector patches
        unit_vectors = torch.where(norm_vectors != 0, raw_vectors / norm_vectors, torch.zeros_like(raw_vectors))

        return unit_vectors, norm_vectors

    def belongs(self, point, atol=gs.atol):
        return

    def is_tangent(self, vector, base_point, atol=gs.atol):
        return

    def random_point(self, n_samples=1, bound=1.0):
        return

    def to_tangent(self, vector, base_point):
        return

