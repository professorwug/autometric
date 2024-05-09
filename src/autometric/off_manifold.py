# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/off-manifold-pullback.ipynb.

# %% auto 0
__all__ = ['grid', 'OffManifolderLinear', 'construct_ndgrid', 'construct_ndgrid_from_shape']

# %% ../../nbs/library/off-manifold-pullback.ipynb 7
import torch
class OffManifolderLinear():
    """
    Folds points off manifold into higher dimensions using random matrices.
    """
    def __init__(self,
                 X, # n x d points sampled from manifold (in latent space)
                 density_loss_function = None, # function that takes a batch of tensors as input, and outputs a scalar which is 0 on the manifold, and bigger further away.
                 folding_dim = 10,
                 density_k = 5,
                 density_tol = 0.1,
                 density_exponential = 4, 
                 # modify to pass in density_loss
                ):
        self.X = X
        self.device = X.device
        self.dim = X.shape[1]
        self.folding_dim = folding_dim
        self.density_k = density_k
        self.density_tol = density_tol
        self.density_exponential = density_exponential
        self.density_loss_function = density_loss_function
        
        self.preserve_matrix = torch.zeros(self.dim, self.folding_dim, dtype=torch.float).to(self.device)
        for i in range(self.dim):
            self.preserve_matrix[i,i] = 1.0

        self.random_matrix = torch.randn(self.dim, self.folding_dim).to(self.device)
        self.random_matrix[:self.dim, :self.dim] = torch.zeros(self.dim, self.dim).to(self.device)
        # self.random_layer = torch.nn.Linear(self.dim, self.folding_dim)

    def _1density_loss(self, a):
        # 0 for points on manifold within tolerance. Designed for a single point.
        dists = torch.linalg.norm(self.X - a, axis=1)
        print(dists.shape)
        smallest_k_dists, idxs = torch.topk(dists, self.density_k, largest=False) # return k smallest distances
        loss = torch.sum(
            torch.nn.functional.relu( smallest_k_dists - self.density_tol )
        )
        return loss
    def density_loss(self, points):
        if self.density_loss_function is not None:
            return self.density_loss_function(points)
        else:
            return torch.vmap(self._1density_loss)(points)

    def immersion(self, points):
        preserved_subspace = points @ self.preserve_matrix
        random_dirs = points @ self.random_matrix
        # random_dirs = self.random_layer(points)
        weighting_factor = torch.exp(self.density_loss(points)*self.density_exponential) - 1 # starts at 1; gets higher immediately.
        print(f"{preserved_subspace.shape} {random_dirs.shape} {weighting_factor.shape}")
        return preserved_subspace + random_dirs*weighting_factor[:,None]

    def pullback_metric(self, points):
        if not isinstance(points, torch.Tensor): points = torch.tensor(points, dtype=torch.float)
        jac = torch.func.jacrev(self.immersion, argnums = 0) #(points)
        def pullback_per_point(p):
            print(p)
            print(p.shape)
            J = jac(p[None,:])
            J = torch.squeeze(J)
            print("shape J", J.shape)
            return J.T @ J
        return torch.vmap(pullback_per_point)(points)
    

# %% ../../nbs/library/off-manifold-pullback.ipynb 32
# construct 2d grid
import numpy as np
def construct_ndgrid(*args):
    # Construct an ndgrid of points
    ndgrid = np.meshgrid(*args, indexing='ij')
    points = np.vstack(list(map(np.ravel, ndgrid))).T
    return points
def construct_ndgrid_from_shape(dim, points_per_dim):
    # Construct an ndgrid of points
    ranges = [np.arange(start=-1,stop=1,step=2/points_per_dim) for _ in range(dim)]
    points = construct_ndgrid(*ranges)
    # move the element closest to the origin to the front
    distances_to_origin = (points**2).sum(-1)
    sorting_idxs = np.argsort(distances_to_origin, )
    points = points[sorting_idxs]
    return points
grid = construct_ndgrid_from_shape(2,100)
