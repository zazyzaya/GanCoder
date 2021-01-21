import torch 

from torch import nn
from torch.autograd import Variable

'''
Using an adapted version of KD GAN 
https://arxiv.org/pdf/1711.04071.pdf

Which allows for indexing node representations the Discriminator/encoder
uses by taking the "realness" score the generated x and the real x produce
and using them as rewards s.t. backprop is possible over the probability
distro it uses

The generator is a simple NN that to output prob distro of nodes that are
convincing (but not real) neighbors of a given input node
'''
class RLGenerator(nn.Module):
    def __init__(self, num_nodes, embed_dim, hidden_dim=32):
        super(RLGenerator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_nodes),
            #nn.Dropout(0.25)
        )

    def forward(self, x, batch_size=1):
        x = self.net(x)

        # Get indices to pass to discriminator
        probs = torch.softmax(Variable(x), dim=1)
        sample = torch.multinomial(probs, num_samples=batch_size)

        # Get log probability to pass to loss 
        log_probs = torch.log_softmax(x, dim=1)
        log_probs = log_probs.gather(1, sample)

        if batch_size == 1:
            log_probs = log_probs.squeeze(-1)
            sample = sample.squeeze(-1)

        return sample, log_probs

    '''
    Takes tensor of rewards (constants, w no attached grad)
    and back props r * log G(x' | x)
    '''
    def loss_fn(self, log_probs, rewards):
        loss = -torch.mean(Variable(torch.relu(rewards)) * log_probs)
        return loss 


'''
Not terribly important what this is, as long as its some sort of 
node autoencoder capible of finding X1X2^T approaching 1 if neighbors
0 else 

For this example, just using a GCN GAE to see if it's more successful than
a regular one (recycled from MyGAE code)
'''
from torch_geometric.nn import GCNConv
from .GAE import GAE
from .utils import decode 
class RLDiscriminator(GAE):
    def __init__(self, feat_dim, embed_dim, hidden_dim=16):
        super(RLDiscriminator, self).__init__(
            feat_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )

    def loss_fn(self, z, pos, neg):
        if neg != None:
            return super().loss_fn(z, pos, neg)

        # If no negative samples, then this is Generator getting loss
        # and we need the scores individually, not averaged
        return -torch.log(self.decode(z, pos)+1e-6)


import copy 
from torch.optim import Adam 
from .utils import (fast_negative_sampling, get_score, 
                    edge_tvt_split, in_order_negative_sampling)

'''
Cora: AUC 0.9779, AP 0.9780
'''
def train_RLGAN(data, epochs=200, lr=0.01, SE_VAL=25, embed_dim=16, 
                hidden_dim=32, K=10):
    
    G = RLGenerator(data.x.size(0), embed_dim, hidden_dim=hidden_dim)
    D = RLDiscriminator(data.x.size(1), embed_dim, hidden_dim=hidden_dim)

    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    tr, te, va = edge_tvt_split(data)  
    val_neg = fast_negative_sampling(data.edge_index[:, va], va.sum().item())

    src = lambda mask : data.edge_index[:, mask][0, :]
    dst = lambda mask : data.edge_index[:, mask][1, :]

    best = (0, None)
    se  = 0
    g_loss = torch.tensor([0.])

    for e in range(epochs):
        D.train()
        G.eval()

        d_opt.zero_grad()
        g_opt.zero_grad()
        
        # Get embeddings and samples
        z = D(data.x, data.edge_index[:, tr])
        
        # Generate an equal number of neg edges
        gneg, log_prob = G(Variable(z)[src(tr)])
        tneg = in_order_negative_sampling(data.edge_index[:, tr], src(tr))

        # Calculate Discriminator loss & update
        d_loss = D.loss_fn(
            D(data.x, data.edge_index[:, tr]),
            data.edge_index[:, tr],
            torch.cat([
                torch.stack([src(tr), gneg]),
                torch.stack([src(tr), tneg])
            ], dim=1)
        )
        
        d_loss.backward()
        d_opt.step()

        # Calculate Generator loss & update
        for _ in range(K):
            G.train()
            D.eval()

            z = D(data.x, data.edge_index[:, tr])
            neg, log_prob = G(Variable(z)[src(tr)])
            g_opt.zero_grad()

            reward = D.loss_fn(
                z, torch.stack([src(tr), neg]), None
            )
            reward[dst(tr) == neg] = 0 # Punish for using actual samples

            g_loss = G.loss_fn(log_prob, reward)
            g_loss.backward()
            g_opt.step()

        # Get validation data
        with torch.no_grad():
            D.eval()
            z = D(data.x, data.edge_index[:, tr])

            auc, ap = get_score(
                data.edge_index[:, va], 
                val_neg, 
                z
            )

        print(
            "[%d] D Loss: %0.4f  G Loss: %0.4f | Val: AUC %0.3f, AP %0.3f"
            % (e, d_loss.item(), g_loss.item(), auc, ap)
        )

        avg_score = (auc+ap)/2
        if avg_score > best[0]:
            best = (avg_score, copy.deepcopy(D))
            se = 0
        else:
            se += 1
            if se == SE_VAL:
                print("Early stopping!")
                break

    del val_neg
    neg = fast_negative_sampling(data.edge_index, te.sum().item())
    D = best[1]

    with torch.no_grad():
        D.eval()
        z = D(data.x, data.edge_index[:, tr])
        auc, ap = get_score(
            data.edge_index[:,te], 
            neg, 
            z
        )

    print(
        "\nFinal score: AUC %0.4f, AP %0.4f"
        % (auc, ap)
    )

    return auc, ap