# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/1d Embedding Analysis.ipynb.

# %% auto 0
__all__ = ['determinants_of_encoder_pullback', 'trace_of_encoder_pullback', 'rank_of_encoder_pullback',
           'spectral_entropy_of_matrix', 'spectral_entropy_of_encoder_pullback', 'evals_of_encoder_pullback',
           'smallest_eigenvector', 'normal_vectors_of_encoder_pullback', 'visualize_encoder_pullback_metrics',
           'visualize_encoder_pullback_metrics_in_ambient_space', 'plot_indicatrices',
           'indicatrix_volume_variance_metric', 'frequency_of_volume_variance']

# %% ../nbs/1d Embedding Analysis.ipynb 6
from .metrics import PullbackMetric
import numpy as np
import torch
def determinants_of_encoder_pullback(model, dataloader):
    # returns the determinants of the metric matrices for each point in the dataset
    Metric = PullbackMetric(model.input_dim, model.encoder)
    Gs = Metric.metric_matrix(dataloader.dataset.pointcloud).detach().cpu().numpy()
    dets = [np.linalg.det(G) for G in Gs]
    return np.array(dets)

# %% ../nbs/1d Embedding Analysis.ipynb 7
from .metrics import PullbackMetric
import numpy as np
import torch
def trace_of_encoder_pullback(model, dataloader):
    # returns the determinants of the metric matrices for each point in the dataset
    Metric = PullbackMetric(model.input_dim, model.encoder)
    Gs = Metric.metric_matrix(dataloader.dataset.pointcloud).detach().cpu().numpy()
    dets = [np.sum(np.linalg.eigvals(G)) for G in Gs]
    return np.array(dets)

# %% ../nbs/1d Embedding Analysis.ipynb 8
from .metrics import PullbackMetric
import numpy as np
import torch
def rank_of_encoder_pullback(model, dataloader, eps=1e-10):
    # returns the determinants of the metric matrices for each point in the dataset
    Metric = PullbackMetric(model.input_dim, model.encoder)
    Gs = Metric.metric_matrix(dataloader.dataset.pointcloud).detach().cpu().numpy()
    ranks = [np.sum((np.linalg.eigvals(G)>eps).astype(int)) for G in Gs]
    return np.array(ranks)

# %% ../nbs/1d Embedding Analysis.ipynb 9
from .metrics import PullbackMetric
import numpy as np
import torch
def spectral_entropy_of_matrix(A):
    # returns the spectral entropy of a matrix
    # A is a numpy array
    eigvals = np.linalg.eigvals(A)
    eigvals = eigvals[eigvals > 0]
    eigvals /= eigvals.sum()
    return -np.sum(eigvals * np.log(eigvals))

def spectral_entropy_of_encoder_pullback(model, dataloader):
    # returns the determinants of the metric matrices for each point in the dataset
    Metric = PullbackMetric(model.input_dim, model.encoder)
    Gs = Metric.metric_matrix(dataloader.dataset.pointcloud).detach().cpu().numpy()
    entropies = [spectral_entropy_of_matrix(G) for G in Gs]
    return np.array(entropies)

# %% ../nbs/1d Embedding Analysis.ipynb 10
def evals_of_encoder_pullback(model, dataloader):
    # returns the determinants of the metric matrices for each point in the dataset
    Metric = PullbackMetric(model.input_dim, model.encoder)
    Gs = Metric.metric_matrix(dataloader.dataset.pointcloud).detach().cpu().numpy()
    e = [np.sort(np.linalg.eigvals(G))[::-1] for G in Gs]
    return np.vstack(e)

# %% ../nbs/1d Embedding Analysis.ipynb 11
def smallest_eigenvector(matrix):
    """
    Find the eigenvector associated with the smallest eigenvalue of a square matrix.

    Args:
        matrix (np.array): A square numpy array representing a matrix.

    Returns:
        np.array: The eigenvector associated with the smallest eigenvalue.
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the index of the smallest eigenvalue
    min_index = np.argmin(eigenvalues)

    # Return the corresponding eigenvector
    return eigenvectors[:, min_index]
def normal_vectors_of_encoder_pullback(model, dataloader):
    # returns the determinants of the metric matrices for each point in the dataset
    Metric = PullbackMetric(model.input_dim, model.encoder)
    Gs = Metric.metric_matrix(dataloader.dataset.pointcloud).detach().cpu().numpy()
    e = [smallest_eigenvector(G) for G in Gs]
    return np.vstack(e)

# %% ../nbs/1d Embedding Analysis.ipynb 13
import matplotlib.pyplot as plt
import numpy
from .utils import *
from mpl_toolkits.mplot3d import Axes3D

def visualize_encoder_pullback_metrics(model, dataloader, title):
    X = model.encoder(dataloader.dataset.pointcloud).cpu().detach().numpy()
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    spectral_entropy = spectral_entropy_of_encoder_pullback(model,dataloader)
    axs[0,0].scatter(X[:,0],X[:,1],c=spectral_entropy)
    axs[0,0].set_title("Spectral Entropy")

    trace = trace_of_encoder_pullback(model,dataloader)
    axs[0,1].scatter(X[:,0],X[:,1],c=trace)
    axs[0,1].set_title("Trace")

    rank = rank_of_encoder_pullback(model,dataloader)
    axs[0,2].scatter(X[:,0],X[:,1],c=rank)
    axs[0,2].set_title("Rank")
    
    evals = evals_of_encoder_pullback(model, dataloader)
    for i in range(3):
        axs[1,i].scatter(X[:,0],X[:,1],c=evals[:,i])
        axs[1,i].set_title(f"{printnum(i)} Eigenvalue")
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# %% ../nbs/1d Embedding Analysis.ipynb 14
import matplotlib.pyplot as plt
import numpy
from .utils import *
from mpl_toolkits.mplot3d import Axes3D

def visualize_encoder_pullback_metrics_in_ambient_space(model, dataloader, title):
    X = model.encoder(dataloader.dataset.pointcloud).cpu().detach().numpy()
    D = dataloader.dataset.pointcloud.cpu().detach().numpy()
    figure = plt.figure()

    ax = figure.add_subplot(231, projection='3d')
    spectral_entropy = spectral_entropy_of_encoder_pullback(model,dataloader)
    ax.scatter(D[:,0],D[:,1],D[:,2],c=spectral_entropy)
    ax.set_title("Spectral Entropy")

    ax = figure.add_subplot(232, projection='3d')
    trace = trace_of_encoder_pullback(model,dataloader)
    ax.scatter(D[:,0],D[:,1],D[:,2],c=trace)
    ax.set_title("Trace")

    ax = figure.add_subplot(233, projection='3d')
    rank = rank_of_encoder_pullback(model,dataloader)
    ax.scatter(D[:,0],D[:,1],D[:,2],c=rank)
    ax.set_title("Rank")
    
    evals = evals_of_encoder_pullback(model, dataloader)
    for i in range(3):
        ax = figure.add_subplot(230+i+4, projection='3d')
        ax.scatter(D[:,0],D[:,1],D[:,2],c=evals[:,i])
        ax.set_title(f"{printnum(i)} Eigenvalue")
    
    figure.suptitle(title)
    plt.show()

# %% ../nbs/1d Embedding Analysis.ipynb 17
from .util import *
from .connections import LeviCivitaConnection
from .metrics import PullbackMetric
from .manifolds import RiemannianManifold
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def plot_indicatrices(
                    model,
                    dataloader,
                    grid="convex_hull",
                    device="cpu",
                    num_steps=20,
                    num_gon=50,
                    output_path=None,
                    writer=None,
                    latent_activations=None,
                    model_name="GeomReg",
                    dataset_name="MNIST",
                    labels=None,
                    cmap="tab10",
                    just_on_data = False,
                    scaling_factor = 1/4
                    ):
    if latent_activations is None:
        pointcloud = dataloader.dataset.pointcloud
        try:
            latent_activations = model.encoder(dataloader.dataset.pointcloud).cpu().detach()
        except AttributeError:
            latent_activations = model.encode(dataloader.dataset.pointcloud).cpu().detach()
    if labels is None:
        labels = torch.zeros(pointcloud.shape[0])

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(labels.shape[0], generator=generator)
    
    latent_activations = latent_activations[perm]
    labels = labels[perm]

    coordinates_on_data = get_coordinates(torch.squeeze(latent_activations),
                                          grid="on_data",
                                          num_steps=num_steps,
                                          coords0=None,
                                          dataset_name=dataset_name,
                                          model_name=model_name).to(device)

    coordinates_off_data = get_coordinates(torch.squeeze(latent_activations),
                                           grid="off_data",
                                           num_steps=num_steps,
                                           coords0=None,
                                           dataset_name=dataset_name,
                                           model_name=model_name).to(device)

    if just_on_data:
        coordinates = coordinates_on_data
    else:
        coordinates = torch.vstack([coordinates_on_data, coordinates_off_data])

    # calculate grid step sizes
    x_min = torch.min(latent_activations[:, 0]).item()
    x_max = torch.max(latent_activations[:, 0]).item()
    y_min = torch.min(latent_activations[:, 1]).item()
    y_max = torch.max(latent_activations[:, 1]).item()

    num_steps_x = num_steps
    num_steps_y = int((y_max - y_min) / (x_max - x_min) * num_steps_x)

    step_size_x = (x_max - x_min) / (num_steps_x)
    step_size_y = (y_max - y_min) / (num_steps_y)
    stepsize = min(step_size_x, step_size_y)

    # # find initial coordinate
    # if coords0 is not None:
    #     coords0_index = None
    #     for i, row in enumerate(coordinates.cpu()):
    #         if torch.all(row.eq(coords0)):
    #             coords0_index = i
    coords0_index = 0
    coords0 = latent_activations[0]

    # initialize diffgeo objects
    # TODO: Equip for arbitrary dimensions
    pbm = PullbackMetric(2, model.decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    # generate vector patches at grid points, normed in pullback metric
    try:
        vector_patches, norm_vectors = rm.generate_unit_vectors(num_gon, coordinates)
    except RuntimeError:
        vector_patches, norm_vectors = rm.generate_unit_vectors(num_gon, coordinates.unsqueeze(-1).unsqueeze(-1))

    vector_patches = vector_patches.to(device)

    vector_norms = torch.linalg.norm(vector_patches.reshape
                                     (-1, 2), dim=1)
    max_vector_norm = torch.min(vector_norms[torch.nonzero(vector_norms)])

    normed_vector_patches = vector_patches / max_vector_norm * stepsize * scaling_factor  # / 3
    anchored_vector_patches = coordinates.unsqueeze(1).expand(*normed_vector_patches.shape) + normed_vector_patches

    # create polygons
    polygons = [Polygon(tuple(vector.tolist()), closed=True) for vector in anchored_vector_patches]

    polygons_on_data = polygons[:coordinates_on_data.shape[0]]
    polygons_off_data = polygons[coordinates_on_data.shape[0]:]


    if coords0 is not None:
        polygon0 = polygons.pop(coords0_index)

    """
    Plotting
    """
    
    latent_activations = latent_activations.detach().cpu()

    # plot blobs
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.)
    plt.margins(0.01, 0.01)

    ax.scatter(latent_activations[:, 0],
               latent_activations[:, 1],
               c=labels,
               cmap=cmap,
               **get_sc_kwargs())

    p_on_data = PatchCollection(polygons_on_data)
    p_off_data = PatchCollection(polygons_off_data)
    # p2 = PatchCollection(polygons2)

    # p_off_data.set_edgecolor([0 / 255, 0 / 255, 0 / 255, 0.2])
    # if model_name == "Vanilla" and dataset_name == "Zilionis":
    #    p_off_data.set_facecolor([0 / 255, 0 / 255, 0 / 255, 0.0])
    # else:
    #    p_off_data.set_facecolor([0 / 255, 0 / 255, 0 / 255, 0.2])

    p_on_data.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])
    p_off_data.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])

    if coords0 is not None:
        polygon0.set_color([255 / 255, 0 / 255, 0 / 255, 0.2])
        ax.add_patch(polygon0)

    ax.add_collection(p_off_data)
    ax.add_collection(p_on_data)
    ax.set_aspect("equal")
    ax.axis("off")
    # fig.suptitle(f"Indicatrices")

    if output_path is not None:
        plt.savefig(output_path, **get_saving_kwargs())

    plt.show()

    if writer is not None:
        writer.add_figure("indicatrix", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()

# %% ../nbs/1d Embedding Analysis.ipynb 24
def indicatrix_volume_variance_metric(
    model,
    dataloader,
):
    try:
        pointcloud = dataloader.dataset.pointcloud
    except AttributeError:
        pointcloud = dataloader.dataset.X
    try:
        latent_activations = model.encoder(pointcloud).cpu().detach()
    except AttributeError:
        latent_activations = model.encode(pointcloud).cpu().detach()
    
    # set up manifold
    pbm = PullbackMetric(2, model.decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)
    # calculate the logarithm of the generalized jacobian determinant
    log_dets = rm.metric_logdet(base_point=latent_activations)
    # replace nan values with a small number
    #EPSILON = 1e-9
    #torch.nan_to_num(log_dets, nan=EPSILON, posinf=EPSILON, neginf=EPSILON)
    torch.nan_to_num(log_dets, nan=1., posinf=1., neginf=1.) # ?? TODO Investigate replacement of eps by 1
    # calculate the variance of the logarithm of the generalized jacobian determinant
    raw_loss = torch.var(log_dets)
    return raw_loss

# %% ../nbs/1d Embedding Analysis.ipynb 27
from diffusion_curvature.kernels import gaussian_kernel
import pygsp

def frequency_of_volume_variance(
    model,
    dataloader,
    k=5, # k-nn graph
    alpha=1, # degree of anisotropic density normalization
):
    try:
        pointcloud = dataloader.dataset.pointcloud
    except AttributeError:
        pointcloud = dataloader.dataset.X
    try:
        latent_activations = model.encoder(pointcloud).cpu().detach()
    except AttributeError:
        latent_activations = model.encode(pointcloud).cpu().detach()
    
    # construct a graph out of these latent_activations
    W = gaussian_kernel(
        latent_activations.numpy(),
        kernel_type='adaptive',
        k = k,
        anisotropic_density_normalization = alpha,
    )
    G = pygsp.graphs.Graph(W)
    
    # set up manifold
    pbm = PullbackMetric(2, model.decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)
    # calculate the logarithm of the generalized jacobian determinant
    log_dets = rm.metric_logdet(base_point=latent_activations)
    # replace nan values with a small number
    EPSILON = 1e-9
    torch.nan_to_num(log_dets, nan=EPSILON, posinf=EPSILON, neginf=EPSILON)
    log_dets = log_dets.detach().cpu().numpy()

    # compute quadric form of laplacian
    quadric_form = log_dets @ G.L @ log_dets.T
    return quadric_form
