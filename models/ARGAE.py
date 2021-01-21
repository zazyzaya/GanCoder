import torch
import copy 

from .GAE import GAE
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

'''
Trying to impliment Adversarially Regularized GAE from
Pan et al., 2018 -- IJCAI

The generator is just a GAE constructed in the same way as
the Kipf and Welling (V)GAE
'''
class ARGAEDiscriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim=32):
        super(ARGAEDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.feat_dim = feat_dim
        self.loss = nn.BCELoss()

        self.tl = 1.0
        self.fl = 0.0

    '''
    Allows either tensor of Zs to be passed in, or an int
    denoting the batch size of noise data to process 
    '''
    def forward(self, data):
        if type(data) == torch.Tensor:
            return self.layers(data)

        # Paper samples "true" values from normal distro
        x = Variable(
            torch.empty(data, self.feat_dim).normal_()
        )

        '''
        dist = torch.distributions.exponential.Exponential(1.0)
        x = Variable(dist.sample((data, self.feat_dim)))
        '''

        return self.layers(x)

    def loss_fn(self, inputs, tl):
        val = self.tl if tl else self.fl 
        labels = Variable(
            torch.full((inputs.size(0), 1), val)
        )

        return self.loss(inputs, labels)

'''
Note: with higher embedding & hidden dimensions (128 and 256 resp.), 
regular GAE (set K=0) outperforms ARGAE, by like 10% 
'''
def train_ARGAE(data, epochs=200, lr=0.0005, K=10, SE_VAL=25,
                embed_dim=16, hidden_dim=32):
    from torch.optim import Adam 
    from torch_geometric.utils import to_dense_adj, dense_to_sparse
    from .utils import get_score, edge_tvt_split, fast_negative_sampling

    nn = data.x.size(0)
    ne = data.edge_index.size(1)

    G = GAE(data.x.size(1), embed_dim=embed_dim, hidden_dim=hidden_dim)
    D = ARGAEDiscriminator(G.embed_dim, hidden_dim=hidden_dim)
    
    tr, val, te = edge_tvt_split(data)
    tv = torch.logical_or(tr, val)

    adj_tr = to_dense_adj(data.edge_index[:, tr], max_num_nodes=nn)[0]
    neg_val = fast_negative_sampling(data.edge_index[:, tv], val.sum().item())

    g_opt = Adam(G.parameters(), lr=lr)#, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=lr)#, betas=(0.5, 0.999))
    
    best = (0, None)
    stop_early = 0

    for e in range(epochs):
        d_loss = float('nan')
        neg_tr = fast_negative_sampling(data.edge_index[:, tr], tr.sum().item())

        for _ in range(K):
            # Train disc 
            G.eval() 

            d_opt.zero_grad()
            d_loss_t = D.loss_fn(D(nn), True)
            d_loss_f = D.loss_fn(
                D(
                    G(data.x, data.edge_index[:, tr])
                ), False
            )

            d_loss_t.backward()
            d_loss_f.backward()
            d_opt.step()

            d_loss = d_loss_t.mean().item() + d_loss_f.mean().item()
            d_loss /= 2

            # Train encoder to confuse D 
            g_loss_d = D.loss_fn(
                D(
                    G(data.x, data.edge_index[:, tr])
                ), True
            )
            g_loss_d.backward()
            g_opt.step()

        # Train encoder 
        G.train() 
        g_opt.zero_grad()
        g_loss = G.loss_fn(
            G(data.x, data.edge_index[:, tr]), 
            data.edge_index[:, tr],
            neg_tr
        )
        g_loss.backward()
        g_opt.step()

        g_loss = g_loss.item()

        # Check AUC  
        with torch.no_grad():
            G.eval()
            auc, ap = get_score(
                data.edge_index[:, val], 
                neg_val, 
                G(data.x, data.edge_index[:, tr]).detach()
            )

        print("[%d] D Loss: %0.3f  G Loss: %0.4f  |  AUC: %0.3f AP: %0.3f" %
            (
                e, d_loss, g_loss, 
                auc, ap
            )
        )

        # Early stopping to prevent over-fitting
        vscore = (auc+ap).mean()
        if vscore > best[0]:
            best = (vscore, copy.deepcopy(G)) 
            stop_early = 0
        else:
            stop_early += 1
            if stop_early == SE_VAL:
                print("Early stopping")
                break
    
    # Final score and cleanup
    del adj_tr
    del neg_val 

    G = best[1]
    neg = fast_negative_sampling(data.edge_index, te.sum().item())

    G.eval()
    auc, ap = get_score(
        data.edge_index[:, te],
        neg, 
        G(data.x, data.edge_index[:, tr]).detach()
    )

    print("Final AUC: %0.4f  AP: %0.4f" % (auc, ap))
    return auc, ap