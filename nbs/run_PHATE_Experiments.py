from diffusion_curvature.datasets import sphere, torus
from autometric.n0d2_datasets import make_swiss_roll, generate_sine_wave_dataset
from autometric.n3b7_phate_embedding_experiments import run_PHATE_embedding_experiment
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PHATE Embedding Experiment')
parser.add_argument('--datasetname', type=str, default='swiss_roll', help='Name of the dataset')
parser.add_argument('--reconstruction_weight', type=float, default=1, help='Weight for reconstruction loss')
parser.add_argument('--dist_loss_fn', type=str, default='coordinatewise', help='Distance loss function')
args = parser.parse_args()

datasetname = args.datasetname
reconstruction_weight = args.reconstruction_weight
dist_loss_fn = args.dist_loss_fn

distance_weight = 1

run_PHATE_embedding_experiment(
    datasetname,
    distance_weight=distance_weight,
    reconstruction_weight=reconstruction_weight,
    dist_loss_fn=dist_loss_fn,
    savepath='../data_phate_experiments/'
)