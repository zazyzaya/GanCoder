# GanCoders
This package contains implimentations of various GAN-based graph autoencoders to verify/further test the results resported in each model's related paper. It also serves as a little playground for me to mess with building my own. In this readme, I'll post the odd note about each model so I have some means of looking back and accurately evaluating each one for my own purposes. 

<br/>

#### General observations
The most significant outcome from playing with these various autoencoders that I've noticed is this: the reconstruction loss function for (V)GAEs used by Kipf and Welling (and by extention, used by default by the uninitiated) is bad. Now, I'm not the first person to notice this. The PyTorch version of (V)GAE uses the better loss function I'm about to describe, but it's worth pointing out, and pointing out right here at the top of the page. 

The function in question is this:

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%5Cmathcal%7BL%7D_R%20%3D%20%5Cmathbb%7BE%7D_%7Bq%28%5Cmathbf%7BZ%7D%7C%5Cmathbf%7BX%2CA%7D%29%29%7D%5Cbig%5B%5Clog%20p%28%5Cmathbf%7BA%7CZ%7D%29%5Cbig%5D)

This requires not only reconstructing *the entire adjacency matrix* which fills quite a bit of memory, but then backpropogating the whole thing. It's absurd. We really don't care a lot of the time how close obviously distant nodes are going to be when their link score is tested. If the model is general enough, it'll just know without having to be tested on it every time. 

To avoid this absurdly large time and space requirement, we can instead create two edge lists, 2x|E| tensors of source to destination tuples of true positives (e.g. edges that we know exist in the training set) and an equally sized list of true negatives (randomly selected non-edges from the training set). This is much, much faster. Now, if we have matrix **Z** where each row denotes a node's encoding, we index it to get stacks of source and destination encodings, **S** and **D**. The new loss function is now 

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%5Cmathcal%7BL%7D_R%20%3D%20-%5Clog%5Cbigg%28%20%5Csum_%7Bi%3D0%7D%5Ed%20%5Cmathbf%7B%28S%20%5Codot%20D%29_i%7D%20%5Cbigg%29)

for true positive edges, and 

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%5Cmathcal%7BL%7D_R%20%3D%20-%5Clog%5Cbigg%281%20-%20%5Csum_%7Bi%3D0%7D%5Ed%20%5Cmathbf%7B%28S%20%5Codot%20D%29_i%7D%20%5Cbigg%29)

for true negative edges. Where *d* represents the number of dimensions. Note that this is just the dot product of each edges embeddings, and PyTorch can evaluate this equation wicked fast. (in code it's just one line: `-torch.log((Z[src] * Z[dst]).sum(dim=1))`)

Doing this boosts the GAE's AUC from a measly ~0.92 (what's reported in the paper) to a spicy ~0.97 (what the GAE in this repo can achieve with the same settings).[<sup>1</sup>](#fn1)


#### [Adversarially Regularized Graph Autoencoder (ARGAE)](https://www.ijcai.org/Proceedings/2018/0362.pdf)
In this model, the generator is the encoder itself, and the "True" samples are random noise sampled from the Gaussian distribution. In this way, the graph autoencoder (GAE) learns to generate embeddings that seem likely to be from the normal distribution. 

It seems very similar to a normal variational GAE (VGAE)--like a method of ensuring node embeddings stay inside the normal distribution without having to take the KL loss at each step. However, the authors of the paper also impliment this on a VGAE. I didn't bother doing this, as their reported results were mostly the same as, or worse than the adversarial GAE. 

In checking their results, I can confirm, I got the same numbers on the Cora and Citeseer datasets. The ARGAE does indeed work better than a normal GAE... when using low dimensional embeddings. 

See, I wanted to see how their method stacked up against [node2vec](https://dl.acm.org/doi/pdf/10.1145/2939672.2939754), which they didn't report.[<sup>2</sup>](#fn2) And node2vec used 128-dimensional vectors for embedding, so I tried cranking ARGAE to that same level. It only performed moderately better, if not worse. However, when removing the restriction to generate embeddings in the normal distribution (i.e. turning off the adversarial part of the model), regular GAE with 128-dimensional vectors was hitting AUC scores higher than 0.999. So, yeah. May be better to just use the simpler model sometimes. It's hard to beat plain old GAE. 

I think it doesn't work well for higher dimensions because it's less important for the vectors to be normally distributed, as with higher dimensional embeddings, they have more space to encode more information, which likely won't be in the normal distribution. But at low dimensions, it only encodes the graph information which is distributed normally.[<sup>3</sup>](#fn3) So maybe it's that the really important info is generally distributed normally, and the fine details aren't? But the authors don't really explain why they do their method, just that it's effective. I dunno. 

Interesting model though. 

#### [ProGAN: Network Embedding via Proximity Generative Adversarial Network](https://dl.acm.org/doi/pdf/10.1145/3292500.3330866)
This model is interesting, and is sort of what I was thinking of trying to build in the first place.[<sup>4</sup>](#fn4) Basically, the discriminator looks at a set of triplets &#12296;x, y, z&#12297; where the similarity between x and y should be high (ideally 1), and the similarity between x and z should be low (ideally 0). Similarity is calculated via inner product, and as a result, finding a good metric for similarity will also give you a nice node embedding. 

I know what you're thinking: "but zazyzaya, that's not very adversarial of that network". Well that's where you're wrong, bucko. The discriminator also has a branch that looks at all thre elements of the triplet, and decides if they're part of the original set of nodes or not. There you go, adversarial network.[<sup>5</sup>](#fn5) 

The generator is what really kicks this bad boy into high freakin gear. Okay, it's not that exciting, but I thought it was cool. So, the generator wants to build up a 3-tuple to give to the discriminator, but it needs some way of deciding if it's turning the static into a similar node, or a different node. The way the authors do this is by dividing the dimension of that static in two, such that the first half of the static that's turned into x is also the first half of the static that's turned into y. That way, the two nodes have similar latent input, so--hopefully--they'll have similar latent output! Cool, right? 

Anyway, the authors only tested this on node classification, which makes sense, since it only really works on graphs that have node features. But I am interested entirely in link prediction, which they didn't test for. If you want to do something right, you've got to do it yourself. 

Unfortunately, they didn't test it for a reason. See, it's kind of crap with link prediction. It can only hit AUCs of low 80s if it's very very lucky. It makes sense, there's no message passing going on, so any information about neighborhoods it has to totally infer from node features, and the loss function. It's not great at this. So not great in fact, I didn't even bother giving it a full test run on all of the data sets. It's just not very good at this task. 

#### [KBGAN: Adversarial Learning for Knowledge Graph Embeddings](https://arxiv.org/pdf/1711.04071.pdf)
At first, I really liked this architecture. It solves a big problem in autoencoding: with so many more negative samples than positives, how do you decide which one's to train with? The generator, given a list of source nodes outputs a probability distribution of what good destination nodes would be to fool the decoder into thinking they're neighbors. 

And honestly? It gets really good at this! Too good, in fact. It gets so good at this that it starts picking neighbors that later end up in the validation and test set. It starts picking nodes that actually are neighbors (if you don't account for this in the training). Things get ugly. As a result, output is very often worse than if you didn't use it to sample negative nodes at all. Oh well. Still gets a nice 0.95 AUC though. 

<br/>

#### Footnotes

<a name=fn1></a><sub>1. Note that you can verify this yourself by setting the parameter `use_recon_loss` to `True` in the GAE constructor. If one does this, the results are much more reflective of those reported in the original paper, and subsequent papers which build from it. I'm kind of surprised no one has thought to use the better loss function in future papers. I mean, it's the default way it's calculated in lots of libraries, it must have happened at some point, but if you were to use any of these algorithms with the better loss function, they would run better than what's reported for GAE simply because they use a more optimized GAE. Dunno. Weird stuff, man.</sub>

<a name=fn2></a><sub>2. I don't think this was intentional to hide anything. Node2Vec was closest to LINE in their paper, which ARGAE beat handilly. I think node2vec hadn't been published at the time ARGAE was being written, or it was just before they submitted it or something.</sub>

<a name=fn3></a><sub>3. I suspect this has something to do with "the law of large numbers", where if you mix enough distributions together, it just becomes Gaussian. But I'm really not sure.</sub>

<a name=fn4></a><sub>4. Why is it that as soon as you think of a cool, new, idea, 15 different researchers already published in KDD about it? Oh well.</sub>

<a name=fn5></a><sub>5. Okay, I admit, this network also isn't very adversarial. You could probably do this without generating fake samples, and it would work just as well.</sub>
