# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/2a autoencoder.ipynb.

# %% auto 0
__all__ = ['AutoencoderModel', 'View', 'Print', 'BoxAutoEncoder', 'ConvolutionalAutoEncoder', 'LinearAE',
           'VanillaAutoencoderModel', 'DerrickTheAutoencoder', 'DistanceMatchingAutoencoder']

# %% ../nbs/2a autoencoder.ipynb 1
"""
THIS FILE WAS TAKEN FROM https://github.com/BorgwardtLab/topological-autoencoders
"""


"""Base class for autoencoder models."""

import abc
from typing import Dict, Tuple
import torch.nn as nn
class AutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for autoencoders."""

    # pylint: disable=W0221
    @abc.abstractmethod
    def forward(self, x) -> Tuple[float, Dict[str, float]]:
        """Compute loss for model.

        Args:
            x: Tensor with data

        Returns:
            Tuple[loss, dict(loss_component_name -> loss_component)]

        """

    @abc.abstractmethod
    def encode(self, x):
        """Compute latent representation."""

    @abc.abstractmethod
    def decode(self, z):
        """Compute reconstruction."""

    def immersion(self, x):
        """The immersion defined by the autoencoder (or more precisely, the decoder)."""
        return self.decoder(x)


# %% ../nbs/2a autoencoder.ipynb 2
"""
THIS FILE WAS PARTLY TAKEN FROM https://github.com/BorgwardtLab/topological-autoencoders
"""


"""Submodules used by models."""


# Hush the linter: Warning W0221 corresponds to a mismatch between parent class
# method signature and the child class
# pylint: disable=W0221

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from collections import OrderedDict

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Print(nn.Module):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def forward(self, x):
        print(self.name, x.size())
        return x


class BoxAutoEncoder(AutoencoderModel):
    """100-100-100-2-100-100-100."""

    def __init__(self, input_dims=(1, 28, 28), **kwargs):
        super().__init__()
        self.latent_dim = 2
        n_input_dims = np.prod(input_dims)
        self.input_dim = n_input_dims.item()
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            # View((-1, n_input_dims)),
            nn.ELU(),
            nn.Linear(self.input_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.latent_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=self.input_dim),
            # View((-1,) + tuple(input_dims)),
        )

        self.reconst_error = nn.MSELoss()

        self.register_hook()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        x = x.view((-1, self.input_dim))

        x = self.encoder(x)
        return x

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        z = self.decoder(z)
        z = z.view((-1, *self.input_dims))
        return z

    def forward_(self, x):
        x = x.view((-1, self.input_dim))
        x = self.encoder(x)

        z = self.decoder(x)
        z = z.view((-1, *self.input_dims))

        return z

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}

    def register_hook(self):
        self.encoder.register_forward_hook(self.get_activation())

    def get_activation(self):
        """
        :return: activations at layer model
        """

        def hook(model, input, output):
            self.latent_activations = output
            self.latent_activations_detached = output.detach()

        return hook

    def load(self, path):
        dict = torch.load(path)

        # edit keys, since they are created from higher TopoAE class
        new_dict = OrderedDict([])

        for key in dict.keys():
            arr = key.split(".")
            if len(arr) == 1 or arr[-1] not in ["weight", "bias"]:
                continue
                # del dict[key]
            # elif arr[-1] not in ["weight", "bias"]:
            #    del dict[key]
            else:
                if arr[0] == "autoencoder":
                    new_key = ".".join(arr[1:])
                else:
                    new_key = key

                new_dict.update({new_key: dict[key]})

                # dict = OrderedDict([(new_key, v) if k == key else (k, v) for k, v in d.items()])
                # new_dict.update({new_key: dict[key]})

        # self.load_state_dict(torch.load(path))
        self.load_state_dict(new_dict)

        self.eval()


class ConvolutionalAutoEncoder(AutoencoderModel):
    """Convolutional autoencoder for MNIST and FashionMNIST"""

    def __init__(self, input_dims=(1, 28, 28), **kwargs):
        super().__init__()
        self.latent_dim = 2
        n_input_dims = np.prod(input_dims)
        self.input_dim = n_input_dims.item()
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            # b, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ELU(),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.Conv2d(8, 2, 2, stride=1, padding=0),  # b, 2, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 8, 2, stride=1),  # b, 8, 2, 2
            nn.ELU(),
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ELU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        self.reconst_error = nn.MSELoss()

        self.register_hook()

    def immersion(self, x):
        return self.decoder(x).view(-1, 784)

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward_(self, x):
        x = self.encoder(x)
        z = self.decoder(x)

        return z

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}

    def register_hook(self):
        self.encoder.register_forward_hook(self.get_activation())

    def get_activation(self):
        """
        :return: activations at layer model
        """

        def hook(model, input, output):
            self.latent_activations = output
            self.latent_activations_detached = output.detach()

        return hook

    def load(self, path):
        dict = torch.load(path)

        # edit keys, since they are created from higher TopoAE class
        new_dict = OrderedDict([])

        for key in dict.keys():
            arr = key.split(".")
            if len(arr) == 1 or arr[-1] not in ["weight", "bias"]:
                continue
                # del dict[key]
            # elif arr[-1] not in ["weight", "bias"]:
            #    del dict[key]
            else:
                if arr[0] == "autoencoder":
                    new_key = ".".join(arr[1:])
                else:
                    new_key = key

                new_dict.update({new_key: dict[key]})

                # dict = OrderedDict([(new_key, v) if k == key else (k, v) for k, v in d.items()])
                # new_dict.update({new_key: dict[key]})

        # self.load_state_dict(torch.load(path))
        self.load_state_dict(new_dict)

        self.eval()


class LinearAE(AutoencoderModel):
    """input dim - 2 - input dim."""

    def __init__(self, input_dims=(1, 28, 28)):
        super().__init__()
        self.input_dims = input_dims
        n_input_dims = np.prod(input_dims)
        self.encoder = nn.Sequential(
            View((-1, n_input_dims)),
            nn.Linear(n_input_dims, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, n_input_dims),
            View((-1,) + tuple(input_dims)),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


# %% ../nbs/2a autoencoder.ipynb 4
"""
THIS FILE WAS PARTLY TAKEN FROM https://github.com/BorgwardtLab/topological-autoencoders
"""
"""Vanilla models."""

class VanillaAutoencoderModel(AutoencoderModel):
    def __init__(self, autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None):
        super().__init__()

        ae_kwargs = ae_kwargs if ae_kwargs else {}
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)

        self.with_geom_loss = False

        if self.with_geom_loss:
            self.determinant_criterion = DeterminantLoss(model=self.autoencoder)

    def forward(self, x):
        # Use reconstruction loss of autoencoder

        if self.with_geom_loss:
            ae_loss, ae_loss_comp = self.autoencoder(x)

            det_loss = self.determinant_criterion().detach_()

            loss = ae_loss
            loss_components = {
                'loss.autoencoder': ae_loss,
                'loss.geom_error': det_loss
            }

            loss_components.update(ae_loss_comp)
            return (
                loss,
                loss_components
            )
        else:
            return self.autoencoder(x)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)


# %% ../nbs/2a autoencoder.ipynb 6
import torch
import torch.nn as nn
import pytorch_lightning as pl

class DerrickTheAutoencoder(pl.LightningModule):
    """
    Derrick doesn't have many friends, so he spends his time encoding data into the specified latent dimension using tanh activations.
    Fortunately, he's friendly with PyTorch Lightning; he gets struck occasionally, but that's made up by his lickety-split training.
    """
    def __init__(self, input_dim, intrinsic_dimension, learning_rate=1e-3):
        super().__init__()
        self.intrinsic_dimension = intrinsic_dimension
        self.lr = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=intrinsic_dimension)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=intrinsic_dimension, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def immersion(self, x):
        """The immersion defined by the autoencoder (or more precisely, the decoder)."""
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('test_loss', loss)
        return loss
    
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# %% ../nbs/2a autoencoder.ipynb 8
import torch
import torch.nn as nn
import pytorch_lightning as pl

class DistanceMatchingAutoencoder(pl.LightningModule):
    """
    What if you already know what your latent space should look like, but want to learn a differentiable mapping into it?
    Enter the DistanceMatchingAutoencoder, or DISMA for short. In addition to a mean squared error loss, it also penalizes the difference between the pairwise distances of the embedding and the supplied ground truth.

    Each minibatch from the dataloader is assumed to have the following keys:
    - `x`: the input data
    - `d`: the ground truth pairwise distances
    """
    def __init__(self, input_dim, intrinsic_dimension, learning_rate=1e-3, reconstruction_weight=1, distance_weight=1):
        super().__init__()
        self.intrinsic_dimension = intrinsic_dimension
        self.lr = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.distance_weight = distance_weight
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=intrinsic_dimension)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=intrinsic_dimension, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def immersion(self, x):
        """The immersion defined by the autoencoder (or more precisely, the decoder)."""
        return self.decoder(x)

    def distance_loss(self, x_embedded, ground_truth_distances):
        embedding_distances = torch.cdist(x_embedded, x_embedded)
        prepared_embedded = torch.log(embedding_distances + torch.eye(embedding_distances.shape[0]))        
        prepared_truth = torch.log(ground_truth_distances + torch.eye(ground_truth_distances.shape[0]))
        return nn.MSELoss()(prepared_embedded, prepared_truth)

    def step(self, batch, batch_idx):
        x = batch['x']
        d = batch['d']
        x_embedded = self.encoder(x)
        x_hat = self.decoder(x_embedded)
        recon_loss = nn.MSELoss()(x_hat, x)
        dist_loss = self.distance_loss(x_embedded, d)
        loss = self.reconstruction_weight * recon_loss + self.distance_weight * dist_loss
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

