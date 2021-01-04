import torch 
import copy 

from torch import nn 
from torch.autograd import Variable
from .utils import sample_triplets

class ProGANGenerator(nn.Module):
    def __init__(self, output_dim, latent_dim=64, hidden_dim=512):
        super(ProGANGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.linear = nn.Sequential(
            nn.Linear(latent_dim*2, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()

    def generate_triplet(self, batch_size):
        sim = Variable(
            torch.empty((batch_size*2, self.latent_dim*2)).normal_()
        ) 

        # Make sure zi and zj share the first half of their latent var
        zi, zj = sim[:batch_size, :], sim[batch_size:, :]
        zj[:, :self.latent_dim] = zi[:, :self.latent_dim]

        zk = Variable(
            torch.empty((batch_size, self.latent_dim*2)).normal_()
        )

        return zi, zj, zk

    '''
    Returns generated triplet <zi, zj, zk> s.t. sim(zi, zj) > sim(zi, zk)
    '''
    def forward(self, batch_size):
        zi, zj, zk = self.generate_triplet(batch_size)
        return self.linear(zi), self.linear(zj), self.linear(zk)

    '''
    D(zi,zj,zk) returns 
        B*3 x 1 tensor of how real the nodes look, 
        B x 1 tensor of how similar zi,zj are
        B x 1 tensor of how dissimilar zi,zk are
    '''
    def loss(self, realness, sim, dis):
        r_loss = self.loss_fn(
            realness, 
            Variable( torch.full((realness.size(0), 1), 1.0) )
        )

        s_loss = self.loss_fn(
            sim, 
            Variable( torch.full((sim.size(0), 1), 1.0) )
        )

        d_loss = self.loss_fn(
            dis, 
            Variable( torch.zeros((dis.size(0), 1)) )
        )

        return r_loss+s_loss+d_loss


class ProGANDiscriminator(nn.Module):
    def __init__(self, input_dim, embed_dim=100, hidden_dim=512):
        super(ProGANDiscriminator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.realness = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LeakyReLU(0.2)
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, zi, zj, zk):
        # Pass everything through first layer
        zi = self.linear(zi)
        zj = self.linear(zj)
        zk = self.linear(zk)
        
        # Branch out into specialized layers
        real = torch.cat([
            self.realness(zi),
            self.realness(zj),
            self.realness(zk)
        ], dim=0)

        zi = self.embedding(zi)
        zj = self.embedding(zj)
        zk = self.embedding(zk)

        # Reshape to find zi[n] zj[n]^t in one go w torch fanciness
        # Thank you stack overflow:  
        # https://stackoverflow.com/questions/65330884
        zi = zi.reshape(zi.size(0), 1, zi.size(1))
        zj = zj.reshape(zj.size(0), zj.size(1), 1)
        zk = zk.reshape(zk.size(0), zk.size(1), 1)

        sim = torch.sigmoid(torch.matmul(zi, zj)).squeeze(1)
        dis = torch.sigmoid(torch.matmul(zi, zk)).squeeze(1)

        return real, sim, dis

    def embed(self, x):
        x = self.linear(x)
        return self.embedding(x)

    
    '''
    In one step, calculate how accurately the discriminator/encoder
        1.) Can determine if nodes are real
        2.) Can determine if fake nodes are (dis)similar
        3.) Can determine if real nodes are (dis)similar
    '''
    def loss(self, realness, sim, dis, is_real=False):
        rl = 1.0 if is_real else 0.0
        
        r_loss = self.loss_fn(
            realness, 
            Variable( torch.full((realness.size(0), 1), rl) )
        )

        s_loss = self.loss_fn(
            sim, 
            Variable( torch.full((sim.size(0), 1), 1.0) )
        )

        d_loss = self.loss_fn(
            dis, 
            Variable( torch.zeros((dis.size(0), 1)) )
        )

        return r_loss + s_loss + d_loss 

def train_ProGAN(data, epochs=200, lr=0.001, SE_VAL=float('inf'),
                embed_dim=100, PCA_dim=128, sample=sample_triplets,
                hidden_dim=512):
    from .utils import get_score, pca
    from torch.optim import Adam 
    from torch_geometric.utils import to_dense_adj, dense_to_sparse

    if PCA_dim:
        data.x = pca(data.x, dim=PCA_dim)

    G = ProGANGenerator(data.x.size(1), hidden_dim=hidden_dim)
    D = ProGANDiscriminator(G.output_dim, embed_dim=embed_dim, hidden_dim=hidden_dim)

    nn = data.x.size(0)
    adj = to_dense_adj(data.edge_index, max_num_nodes=nn)[0]
    neg = ~(
        to_dense_adj(
            data.edge_index,
            max_num_nodes = nn
        ).bool()
    ).long()[0]

    neg = dense_to_sparse(neg)[0]

    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    best = (0, None)
    stop_early = 0

    for e in range(epochs):
        d_loss = float('nan')
        
        # Train disc 
        d_opt.zero_grad()
        zi, zj, zk = G(nn)
        real, sim, dis = D(zi, zj, zk)
        f_loss = D.loss(real, sim, dis, is_real=False)

        xi, xj, xk = sample(adj, data.x, data.edge_index)
        real, sim, dis = D(xi, xj, xk)
        t_loss = D.loss(real, sim, dis, is_real=True)

        f_loss.backward()
        t_loss.backward()
        d_opt.step()

        d_loss = f_loss.mean().item() + t_loss.mean().item()
        d_loss /= 2

        # Train generator
        g_opt.zero_grad()
        zi, zj, zk = G(nn)
        real, sim, dis = D(zi, zj, zk)
        g_loss = D.loss(real, sim, dis, is_real=True)

        g_loss.backward()
        g_opt.step()

        g_loss = g_loss.item()

        # Check AUC  
        recon = D.embed(data.x).detach()
        recon = torch.mm(recon, recon.T)
        auc, ap = get_score(
            data.edge_index, 
            neg, 
            recon
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
            best = (vscore, copy.deepcopy(D)) 
            stop_early = 0
        else:
            stop_early += 1
            if stop_early == SE_VAL:
                print("Early stopping")
                break

    return G, D