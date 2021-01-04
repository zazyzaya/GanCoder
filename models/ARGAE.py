import torch
import copy 

from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

'''
Trying to impliment Adversarially Regularized GAE from
Pan et al., 2018 -- IJCAI
'''
class ARGAEGenerator(nn.Module):
    def __init__(self, feat_dim, latent_dim=16, hidden_dim=32):
        super(ARGAEGenerator, self).__init__()

        self.c1 = GCNConv(feat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, latent_dim)
        self.sig = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        return self.c2(x, ei, edge_weight=ew)

    def recon(self, x, ei, ew=None):
        z = self.forward(x, ei, ew=ew)
        return self.sig(torch.mm(z,z.T))

    def loss_fn(self, inputs, target):
        adj = self.sig(torch.mm(inputs, inputs.T))
        return self.loss(adj, target)

class ARGAEDiscriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64):
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
def train_ARGAE(data, epochs=200, lr=0.001, K=5, SE_VAL=25,
                latent_dim=16, hidden_dim=32):
    from .utils import get_score
    from torch.optim import Adam 
    from torch_geometric.utils import to_dense_adj, dense_to_sparse

    nn = data.x.size(0)
    ne = data.edge_index.size(1)

    G = ARGAEGenerator(data.x.size(1), latent_dim=latent_dim, hidden_dim=hidden_dim)
    D = ARGAEDiscriminator(G.latent_dim, hidden_dim=hidden_dim)
    
    rnd = torch.randperm(ne)
    tr, val, te = rnd[:int(ne*0.85)], rnd[int(ne*0.85):int(ne*0.90)], rnd[int(ne*0.9):]
    tv = torch.cat([tr,val])

    adj_tr = to_dense_adj(data.edge_index[:, tr], max_num_nodes=nn)[0]
    neg_val = ~(
        to_dense_adj(
            data.edge_index[:, tv],
            max_num_nodes = nn
        ).bool()
    ).long()[0]

    neg_val = dense_to_sparse(neg_val)[0]

    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    best = (0, None)
    stop_early = 0

    for e in range(epochs):
        d_loss = float('nan')
        for _ in range(K):
            # Train disc 
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
        g_opt.zero_grad()
        g_loss = G.loss_fn(
            G(data.x, data.edge_index[:, tr]), adj_tr
        )
        g_loss.backward()
        g_opt.step()

        g_loss = g_loss.item()

        # Check AUC  
        auc, ap = get_score(
            data.edge_index[:, val], 
            neg_val, 
            G.recon(data.x, data.edge_index[:, tv]).detach()
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

    neg = dense_to_sparse(
        ~(
            to_dense_adj(
                data.edge_index
            ).bool()
        ).long()[0]
    )[0]

    auc, ap = get_score(
        data.edge_index[:, te],
        neg, 
        G.recon(data.x, data.edge_index).detach()
    )

    print("Final AUC: %0.4f  AP: %0.4f" % (auc, ap))
    return G, D