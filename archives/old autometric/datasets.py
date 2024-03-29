# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/0d Datasets.ipynb.

# %% auto 0
__all__ = ['ToyManifold', 'PointcloudDataset', 'PointcloudWithDistancesDataset', 'dataloader_from_pointcloud_with_distances',
           'train_and_testloader_from_pointcloud_with_distances', 'nd_saddle', 'plot_3d_vector_field',
           'sphere_with_normals']

# %% ../nbs/0d Datasets.ipynb 4
from diffusion_curvature.random_surfaces import rejection_sample_from_surface
from .metrics import PullbackMetric
from .connections import LeviCivitaConnection
from .manifolds import RiemannianManifold
import sympy as sym
import numpy as np
import sympytorch

class ToyManifold:
    def __init__(
        self,
        F, # parameterization of manifold, as a sympy matrix of size $N \times 1$
        variable_bounds, 
        num_points = 2000, # num points to sample
    ):
        self.F = F
        self.param_list = [str(f) for f in list(F.free_symbols)]
        self.intrinsic_dimension = len(self.param_list)
        self.variable_bounds = variable_bounds
        self.X = self.sample(num_points)
        # compute metric information
        self.compute_immersion()
        self.metric = PullbackMetric(self.intrinsic_dimension, self.immersion)
        self.connection = LeviCivitaConnection(self.intrinsic_dimension, self.metric)
        self.manifold = RiemannianManifold(self.intrinsic_dimension, (1,1), metric = self.metric, connection = self.connection)

    def sample(
            self, num_points
    ):
        return rejection_sample_from_surface(
            F = self.F,
            n_points = num_points,
            bounds = self.variable_bounds,
        )
    
    def compute_immersion(self):
        # turns sympy extression into a pytorch function
        list_F = [item for sublist in self.F.tolist() for item in sublist]
        self.pytorch_function = sympytorch.SymPyModule(expressions = list_F)  
        # convert into a generic pytorch function that takes uniform samples in [0,1] and converts them to the right stuff.
        self.immersion = lambda x : self.pytorch_function(
            **{param: (x[i]*(self.variable_bounds[1] - self.variable_bounds[0]) - self.variable_bounds[0]) for i, param in enumerate(self.param_list)}
            )

# %% ../nbs/0d Datasets.ipynb 23
import torch

class PointcloudDataset(torch.utils.data.Dataset):
    def __init__(self, pointcloud):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        
    def __len__(self):
        return len(self.pointcloud)
    
    def __getitem__(self, idx):
        return self.pointcloud[idx]

class PointcloudWithDistancesDataset(torch.utils.data.Dataset):
    def __init__(self, pointcloud, distances, batch_size = 64):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.pointcloud)
    
    def __getitem__(self, idx):
        batch_idxs = torch.randperm(len(self.pointcloud))[:self.batch_size]
        batch = {}
        batch['x'] = self.pointcloud[batch_idxs]
        batch['d'] = self.distances[batch_idxs][:,batch_idxs]
        return batch

# %% ../nbs/0d Datasets.ipynb 24
def dataloader_from_pointcloud_with_distances(pointcloud, distances, batch_size = 64):
    dataset = PointcloudWithDistancesDataset(pointcloud, distances, batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
    return dataloader

# %% ../nbs/0d Datasets.ipynb 25
def train_and_testloader_from_pointcloud_with_distances(
    pointcloud, distances, batch_size = 64, train_test_split = 0.8
):
    X = pointcloud
    D = distances
    split_idx = int(len(X)*train_test_split)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_idx,:split_idx]
    D_test = D[split_idx:,split_idx:]
    trainloader = dataloader_from_pointcloud_with_distances(X_train, D_train, batch_size)
    testloader = dataloader_from_pointcloud_with_distances(X_test, D_test, batch_size)
    return trainloader, testloader

# %% ../nbs/0d Datasets.ipynb 27
from .n0d1_branching_datasets import *

# %% ../nbs/0d Datasets.ipynb 28
from diffusion_curvature.datasets import *
from diffusion_curvature.utils import plot_3d

# %% ../nbs/0d Datasets.ipynb 29
import sympy as sp
import numpy as np
from diffusion_curvature.random_surfaces import rejection_sample_from_surface, scalar_curvature_at_origin
def nd_saddle(n_samples=1000, intrinsic_dim = 2, verbose=False, intensity=1, return_normal_vectors = False):
    d = intrinsic_dim
    vars = sp.symbols('x0:%d' % d)
    saddle = sp.Matrix([*vars])
    for i in range(d,d+1):
        saddle = saddle.row_insert(i, sp.Matrix([intensity*sum([(-1)**j * vars[j]**2 for j in range(d)])]))
    # if verbose: print(saddle)
    # k = scalar_curvature_at_origin(saddle)
    # if return_normal_vector:
    points = rejection_sample_from_surface(saddle, n_samples)
    if not return_normal_vectors:
        return points
    else:
        normal_vecs = np.empty(points.shape)
        directional_derivatives = sp.Matrix([[sp.diff(saddle,v) for v in vars]])
        directional_derivatives_np = sp.lambdify(vars,directional_derivatives,'numpy')
        for i in range(len(points)):
            p = points[i]
            tangent_vecs = directional_derivatives_np(*p[:-1]).T
            # Use SVD to find orthogonal vectors
            U, s, V = np.linalg.svd(tangent_vecs, compute_uv = True)
            # Extract the orthogonal vectors
            orthogonal_vector = V[-1]  # Get the last row of V
            normal_vecs[i] = np.squeeze(orthogonal_vector / np.linalg.norm(orthogonal_vector))
        return points, normal_vecs

# %% ../nbs/0d Datasets.ipynb 31
import numpy as np
import plotly.graph_objects as go
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls

def plot_3d_vector_field(X, *vector_fields, names=None, arrow_length=0.5, upload_to_information_superhighway = False, username = "", api_key = "", filename = ""):
    """
    Create a 3D quiver plot with multiple vector fields.

    Args:
        X (list of tuples or arrays): Collection of points in 3D space.
        *vector_fields: Variable number of vector fields (lists of tuples or arrays).
        arrow_length (float): Scaling factor for arrow lengths.

    Returns:
        None
    """
    if names is None:
        names = [f"Vector Field {i}" for i in range(len(vector_fields))]
    
    fig = go.Figure()

    # Convert to NumPy array for vectorized operations
    X = np.array(X)

    # Generate a list of colors
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'brown', 'pink', 'grey', 'cyan']
    if len(vector_fields) > len(colors):
        # Generate more colors if needed
        additional_colors = np.random.choice(colors, size=(len(vector_fields) - len(colors)))
        colors.extend(additional_colors)

    # Function to add arrows (vectors)
    def add_arrows(X, V, color, name):
        V = np.array(V) * arrow_length
        end_points = X + V
    
        # Arrows (cones)
        fig.add_trace(go.Cone(
            x=end_points[:, 0],
            y=end_points[:, 1],
            z=end_points[:, 2],
            u=V[:, 0],
            v=V[:, 1],
            w=V[:, 2],
            sizemode='absolute',
            sizeref=0.1,
            showscale=False,
            colorscale=[[0, color], [1, color]],
            cmin=0,
            cmax=1,
            name=name,
            legendgroup=name,
            showlegend=True  # Set to True to show in legend
        ))
    
        # Lines (arrow shafts)
        for start, end in zip(X, end_points):
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(width=3, color=color),
                showlegend=False  # Set to False to avoid duplicate legend entries
            ))


    # Markers for points
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color='black'),
        name='Points',
        showlegend=False
    ))

    # Add arrows for each vector field
    for i, vector_field in enumerate(vector_fields):
        add_arrows(X, vector_field, colors[i % len(colors)], names[i])

    # Set axis labels
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ), title=filename)

    # Show the plot
    fig.show()
    if upload_to_information_superhighway:
        url = py.plot(fig, filename = filename, auto_open=False)
        print("Your plot is now live at ",url)


# %% ../nbs/0d Datasets.ipynb 34
def sphere_with_normals(
    n_points
):
    X, ks = sphere(n_points)
    N = X
    return X, N
