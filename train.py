import torch 
import time
import numpy as np

from torch.optim import Adam 
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, dense_to_sparse

DATA_ROOT = '/mnt/raid0_24TB/datasets/torch_geo_datasets'

torch.set_num_threads(16)

LATENT = 16
HIDDEN = 32

def run_GAE(data, epochs=200):
    from models.ARGAE import train_ARGAE
    train_ARGAE(
        data, epochs=epochs, K=0,
        latent_dim=LATENT, hidden_dim=HIDDEN
    )

def run_ARGAE(data, epochs=200):
    from models.ARGAE import train_ARGAE
    train_ARGAE(
        data, epochs=epochs,
        latent_dim=LATENT, hidden_dim=HIDDEN
    )

def run_ProGAN_rw(data, epochs=200):
    from models.ProGAN import train_ProGAN
    from models.utils import sample_triplets_from_rw
    train_ProGAN(
        data, epochs=epochs, sample=sample_triplets_from_rw,
        embed_dim=LATENT, hidden_dim=HIDDEN
    )

def run_ProGAN(data, epochs=200):
    from models.ProGAN import train_ProGAN
    train_ProGAN(
        data, epochs=epochs, embed_dim=LATENT, hidden_dim=HIDDEN
    )

def run_MyGAE(data, epochs=200):
    from models.MyGAE import train_MyGAE
    train_MyGAE(
        data, epochs=epochs, embed_dim=LATENT, hidden_dim=HIDDEN,
        K=256
    )

def run_RLGAN(data, epochs=200):
    from models.RLGAN import train_RLGAN
    train_RLGAN(
        data, epochs=epochs, embed_dim=LATENT, hidden_dim=HIDDEN
    )


if __name__ == '__main__':
    from models.utils import pca
    data = Planetoid(DATA_ROOT, 'Cora').data
    run_RLGAN(data)