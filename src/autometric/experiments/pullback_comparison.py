# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/experiments/1-comparing-pullback-types.ipynb.

# %% auto 0
__all__ = ['PullbackCurvatureComparisonDataset']

# %% ../../../nbs/experiments/1-comparing-pullback-types.ipynb 6
from ..self_evaluating_datasets import SelfEvaluatingDataset, metric
from ..datasets import Torus, Ellipsoid, Saddle
import torch

class PullbackCurvatureComparisonDataset(SelfEvaluatingDataset):
    def __init__(self, num_points = 3000):
        datalist = [
            Torus(num_points = num_points),
            Ellipsoid(num_points = num_points),
            Saddle(num_points = num_points),
        ]
        names = ["Torus", "Ellipsoid", "Saddle"]
        result_names = ["Curvature", "Metric Determinant"]
        super().__init__(datalist, names, result_names)
        self.MSE = torch.nn.MSELoss()
    
    def get_item(self,idx):
        X = self.DS[idx].obj.X
        return X
    
    def get_truth(self, result_name, idx):
        DS = self.DS[idx].obj
        match result_name:
            case "Curvature":
                return DS.ks.detach().numpy()
            case "Metric Determinant":
                return DS.manifold.metric_det(DS.intrinsic_coords).detach().numpy()
            case _:
                raise NotImplementedError(f"No such result {result_name}")
            
    def compute(self, metric, result_name, method_name, filter = None):
        # Overwrite this class with your logic. It implements the computation of a single metric for a single method
        d = {}
        for i, dsname in enumerate(self.names):
            d[dsname] = metric(self.labels[result_name][method_name][i], self.labels[result_name]['ground truth'][i])
        if filter is None: # average dataset values
            return np.mean([d[dsname] for dsname in self.names])
        elif filter == "Everything":
            return d
        elif filter in self.names:
            return d[filter]
        else:
            raise NotImplementedError("Invalid filter")
 
    @metric
    def dataset_mse(self, a, b):
        return np.sum(np.square(a - b))
    
