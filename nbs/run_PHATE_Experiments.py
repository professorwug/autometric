from diffusion_curvature.datasets import sphere, torus
from autometric.n0d2_datasets import make_swiss_roll, generate_sine_wave_dataset
from autometric.n3b7_phate_embedding_experiments import run_PHATE_embedding_experiment
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PHATE Embedding Experiment')
parser.add_argument('--datasetname', type=str, default='swiss_roll', help='Name of the dataset')
parser.add_argument('--reconstruction_weight', type=float, default=1, help='Weight for reconstruction loss')
parser.add_argument('--include_pretraining', action='store_true', help='Whether to include pretraining')
parser.add_argument('--coordinatewise', action='store_true', help='Whether to use coordinatewise loss')
parser.add_argument('--savepath', type=str, default='../data_phate_flex', help='Path to save data')
args = parser.parse_args()

datasetname = args.datasetname
reconstruction_weight = args.reconstruction_weight
include_pretraining = args.include_pretraining
coordinatewise = args.coordinatewise
savepath = args.savepath

distance_weight = 1

run_PHATE_embedding_experiment(
    datasetname,
    distance_weight = distance_weight,
    reconstruction_weight = reconstruction_weight,
    coordinatewise = coordinatewise,
    include_pretraining = include_pretraining,
    savepath = savepath,
    max_epochs = 50,
)