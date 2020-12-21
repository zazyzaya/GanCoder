import torch 
import numpy as np

from torch import nn
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


'''
    Impliments Minibatch Discrimination to avoid same-looking output
    Shamelessly stolen from https://gist.github.com/t-ae/
'''
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=64, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # Outputs Batch x Out*Kernel 
        matrices = x.mm(self.T.view(self.in_features, -1))

        # Transforms to Batch x Out x Kernel
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        # Now we quickly find distance from each X to every other
        # X by viewing it as a 1 x Batch x Out x Kernel mat and a
        #                      Batch x 1 x Out x Kernel mat 
        # That way the difference along the kernel dimension is 
        # equivilant to the dist from x to every other sample
        M = matrices.unsqueeze(0)  
        M_T = M.permute(1, 0, 2, 3) 

        # Simple distance formula
        norm = torch.abs(M - M_T).sum(3)  # Batch x Batch x Out
        expnorm = torch.exp(-norm)
        
        # Add all distances together, and remove self distance (minus 1)
        o_b = (expnorm.sum(0) - 1)   # Batch x Out 
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x