import torch 
import time
import numpy as np
import load_graphs as lg 

from torch.optim import Adam 
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

torch.set_num_threads(16)

def run_ARGAE(data, epochs=200):
    from models.ARGAE import train_ARGAE
    train_ARGAE(data, epochs=epochs)

def run_ProGAN(data, epochs=200):
    from models.ProGAN import train_ProGAN
    train_ProGAN(data, epochs=epochs)
    

if __name__ == '__main__':
    data = lg.load_cora()
    run_ProGAN(data)