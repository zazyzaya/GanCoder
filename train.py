import torch 
import time
import numpy as np
import load_graphs as lg 

from torch.optim import Adam 
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
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

def train_ARGAE(data, epochs=200, lr=0.001, K=3):
    from ARGAE import ARGAEDiscriminator as Disc 
    from ARGAE import ARGAEGenerator as Gen 

    nn = data.x.size(0)
    ne = data.edge_index.size(1)

    G = Gen(data.x.size(1))
    D = Disc(G.latent_dim)
    
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

    for e in range(epochs):
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
    
    # Final score and cleanup
    del adj_tr
    del neg_val 

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

if __name__ == '__main__':
    data = lg.load_cora()
    train_ARGAE(data)