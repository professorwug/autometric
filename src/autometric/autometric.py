# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/string-geodesics.ipynb.

# %% auto 0
__all__ = ['StringGeodesic']

# %% ../../nbs/library/string-geodesics.ipynb 4
import torch
from torch import nn
import lightning as pl

class StringGeodesic(pl.LightningModule):
    def __init__(self, 
                 metric, # a metric object.
                 start, # starting point
                 end, # ending point
                 dim: int = 2, # dimension of the space in which the metric lives.
                 num_beads: int = 1000,
                 flexibility: int = 10,
                 step_size = 1e-3,
                 ):
        super().__init__()
        
        self.dim = dim
        self.num_beads = num_beads
        self.metric = metric
        self.flexibility = flexibility
        self.step_size = step_size
        
        self.beads = torch.vstack(
            [torch.lerp(start, end, t) for t in torch.linspace(0,1,steps=num_beads)]
        ).float()
        self.metric_per_bead = self.metric.metric_matrix(self.beads).float()

        # Create the network layers
        self.Force = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )
    
    def progressive_lengths(self, beads, metric_per_bead):
        # Computes length between each consecutive bead
        lengths = torch.zeros(self.num_beads)
        for i in range(1, len(beads)):
            lengths[i]  = lengths[i-1] + torch.sqrt(self.metric.inner_product(beads[i] -  beads[i-1], beads[i] -  beads[i-1], matrix = metric_per_bead[i]))
        return lengths
    
    def apply_force(self, beads, forces, lengths):
        # applies the force given to each bead, modulating by lengths
        distances_from_end = torch.sqrt(lengths * (torch.sum(lengths) - lengths)) 
        force_modulator = torch.sigmoid(self.flexibility*distances_from_end)*2 - 1
        beads_after_wind = self.step_size*forces*force_modulator + beads
        return beads_after_wind
        
    def forward(self):
        # get distances at start
        self.metric_per_bead = self.metric.metric_matrix(self.beads).float()
        beginning_distances = self.progressive_lengths(self.beads, self.metric_per_bead)
        # calculate force assigned to each bead
        force = self.force(self.beads)
        # apply the force to get new beads
        self.beads = self.apply_force(self.beads, force, beginning_distances)
        # compute new distances
        end_distances = self.progressive_lengths(self.beads)
        return end_distances[-1]
