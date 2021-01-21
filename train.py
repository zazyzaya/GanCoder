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
    return train_ARGAE(
        data, epochs=epochs, K=0,
        embed_dim=LATENT, hidden_dim=HIDDEN, lr=0.01
    )

def run_ARGAE(data, epochs=200):
    from models.ARGAE import train_ARGAE
    return train_ARGAE(
        data, epochs=epochs,
        embed_dim=LATENT, hidden_dim=HIDDEN
    )

def run_ProGAN(data, epochs=200):
    from models.ProGAN import train_ProGAN
    return train_ProGAN(
        data, epochs=epochs, embed_dim=LATENT, hidden_dim=HIDDEN
    )

def run_RLGAN(data, epochs=200):
    from models.RLGAN import train_RLGAN
    return train_RLGAN(
        data, epochs=epochs, embed_dim=LATENT, hidden_dim=HIDDEN
    )

def test_all(data, fname, trials=5):
    import json 

    tests = {
        'ARGAE': run_ARGAE,
        'GAE': run_GAE,
        'RLGAN': run_RLGAN
    }

    scores = {}
    for label, t in tests.items():
        st = time.time()
        test_scores = [t(data) for _ in range(trials)]
        runtime = time.time() - st

        auc, ap = zip(*test_scores)
        scores[label] = {'AUC': auc, 'AP': ap, 'Avg Runtime': runtime/trials}

    print(json.dumps(scores, indent=4))
    with open('results_' + fname + '.json', 'w+') as f:
        f.write(json.dumps(scores, indent=4))

    return scores

from models.utils import pca
from torch_geometric.utils import add_remaining_self_loops
if __name__ == '__main__':
    data = Planetoid(DATA_ROOT, 'Cora').data
    data.x = pca(data.x)
    test_all(data, 'cora', trials=5)

    data = Planetoid(DATA_ROOT, 'Citeseer').data
    data.x = pca(data.x)
    test_all(data, 'citeseer', trials=5)