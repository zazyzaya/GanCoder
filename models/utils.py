import torch 
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def get_score(tp_edges, neg, recon_adj, resize_neg=True):
    # Select an equal number of random false edges
    n = tp_edges.size(1)
    
    if resize_neg:
        neg = neg[:, torch.randperm(neg.size(1))[:n]]

    pscore = recon_adj[tp_edges[0], tp_edges[1]]
    nscore = recon_adj[neg[0], neg[1]]
    score = torch.cat([pscore, nscore]).numpy()

    labels = np.zeros(n*2, dtype=np.long)
    labels[:n] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return auc, ap

from sklearn.decomposition import PCA
def pca(X, dim=256):
    n_components = min(dim, X.size()[1])
    decomp = PCA(n_components=n_components, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))