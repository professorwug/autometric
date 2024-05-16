# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/geodesics.ipynb.

# %% auto 0
__all__ = ['DjikstraGeodesic']

# %% ../../nbs/library/geodesics.ipynb 4
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from fastcore.all import *

class DjikstraGeodesic:
    def __init__(self, 
                 X, # data points
                 k = 10,
                 ):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        G = nx.Graph()
        # add nodes to G
        for i in range(len(X)):
            G.add_node(i)
        # Add edges between each point and its k-nearest neighbors.
        # This looks inefficient - but the inner loop only goes over the k nearest neighbors.
        for i in range(len(X)):
            for j in range(1, k+1):  # start from 1 to avoid self-loop (i.e., point itself)
                G.add_edge(i, indices[i][j], weight = torch.linalg.norm(X[i] - X[indices[i][j]]))
                G.add_edge(indices[i][j], i, weight = torch.linalg.norm(X[i] - X[indices[i][j]]))
        self.G = G
        self.X = torch.as_tensor(X)
        
    def pairwise_geodesic(self, 
                          a, 
                          b, 
                          ts, # ignored. For compatibility with other geodesic functions.
                          ):
        # Builds a nearest neighbor graph out of self.X_ground_truth. Then uses the djikstra algorithm to find the shortest path between a and b.
        # Assumes and and b are in the ground truth data.
        
        # get the indices of the closest points in X_ground_truth
        a_idx = int(torch.argmin(torch.linalg.norm(self.X - a, dim=1), dim=0))
        b_idx = int(torch.argmin(torch.linalg.norm(self.X - b, dim=1), dim=0))
        
        path = nx.shortest_path(self.G, a_idx, b_idx, weight = "weight")
        length = nx.shortest_path_length(self.G, a_idx, b_idx, weight = "weight")
        
        g = self.X[path]
        return g, length
    
    
    def geodesics(self, start_points, end_points, ts):
        """
        Takes start, endpoint pairs in ambient space, and list of times. Returns geodesics and lengths.
        """
        # test if start and end points are tensors
        if isinstance(start_points, np.ndarray):
            start_points = torch.as_tensor(start_points)
        if isinstance(end_points, np.ndarray):
            end_points = torch.as_tensor(end_points)
        if isinstance(ts, np.ndarray):
            ts = torch.as_tensor(ts)
        
        # test if start and end points are among the previously sampled points
        # for each point, find the closest point in the sampled points. If it exceeds a threshold of 1e-3, then raise an error.
        distances_to_sampled_points = torch.cdist(torch.cat([start_points, end_points], dim=0), self.X)
        corresponding_idxs = torch.argmin(distances_to_sampled_points, dim=1)
        
        for i in range(len(start_points)):
            # closest_idx = torch.argmin(distances_to_sampled_points[i], dim=0)
            closest_value = distances_to_sampled_points[i][corresponding_idxs[i]]
            if closest_value > 1e-3:
                raise ValueError(f"Start and end points must be among the previously sampled points. Min dist to manifold is {closest_value}")
            
        # convert the start and end points to the corresponding points in the ground truth data
        start_points = self.X[corresponding_idxs[:len(start_points)]]
        end_points = self.X[corresponding_idxs[len(start_points):]]
            
        gs = []
        lengths = []
        
        for i in range(len(start_points)):
            g, l = self.pairwise_geodesic(start_points[i], end_points[i], ts)
            # convert g to double
            g = g.double()
            gs.append(g)
            lengths.append(l)
            
        # make conversion safe
        lengths = torch.tensor(lengths)
        gs = [g.cpu().detach() for g in gs]
        lengths = lengths.cpu().detach()
        return gs, lengths
    
