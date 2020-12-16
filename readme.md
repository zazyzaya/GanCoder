# GanCoders
This package contains implimentations of various GAN-based graph autoencoders to verify/further test the results resported in each model's related paper. It also serves as a little playground for me to mess with building my own. In this readme, I'll post the odd note about each model so I have some means of looking back and accurately evaluating each one for my own purposes. 

<br/>

#### [Adversarially Regularized Graph Autoencoder (ARGAE)](https://www.ijcai.org/Proceedings/2018/0362.pdf)
In this model, the generator is the encoder itself, and the "True" samples are random noise sampled from the Gaussian distribution. In this way, the graph autoencoder (GAE) learns to generate embeddings that seem likely to be from the normal distribution. 

It seems very similar to a normal variational GAE (VGAE)--like a method of ensuring node embeddings stay inside the normal distribution without having to take the KL loss at each step. However, the authors of the paper also impliment this on a VGAE. I didn't bother doing this, as their reported results were mostly the same as, or worse than the adversarial GAE. 

In checking their results, I can confirm, I got the same numbers on the Cora and Citeseer datasets. The ARGAE does indeed work better than a normal GAE... when using low dimensional embeddings. 

See, I wanted to see how their method stacked up against [node2vec](https://dl.acm.org/doi/pdf/10.1145/2939672.2939754), which they didn't report[<sup>1</sup>](#fn1). And node2vec used 128-dimensional vectors for embedding, so I tried cranking ARGAE to that same level. It only performed moderately better, if not worse. However, when removing the restriction to generate embeddings in the normal distribution (i.e. turning off the adversarial part of the model), regular GAE with 128-dimensional vectors was hitting AUC scores higher than 0.999. So, yeah. May be better to just use the simpler model sometimes. It's hard to beat plain old GAE. 

I think it doesn't work well for higher dimensions because it's less important for the vectors to be normally distributed, as with higher dimensional embeddings, they have more space to encode more information, which likely won't be in the normal distribution. But at low dimensions, it only encodes the graph information which is distributed normally[<sup>2</sup>](#fn2). So maybe it's that the really important info is generally distributed normally, and the fine details aren't? But the authors don't really explain why they do their method, just that it's effective. I dunno. 

Interesting model though. 

#### [ProGAN: Network Embedding via Proximity Generative Adversarial Network](https://dl.acm.org/doi/pdf/10.1145/3292500.3330866)
This model is interesting, and is sort of what I was thinking of trying to build in the first place[<sup>3</sup>](#fn3). Basically, the discriminator looks at a set of triplets &#12296;x, y, z&#12297; where the similarity between x and y should be high (ideally 1), and the similarity between x and z should be low (ideally 0). Similarity is calculated via inner product, and as a result, finding a good metric for similarity will also give you a nice node embedding. 

I know what you're thinking: "but zazyzaya, that's not very adversarial of that network". Well that's where you're wrong, bucko. The discriminator also has a branch that looks at all thre elements of the triplet, and decides if they're part of the original set of nodes or not. There you go, adversarial network[<sup>4</sup>](#fn4). 

The generator is what really kicks this bad boy into high freakin gear. Okay, it's not that exciting, but I thought it was cool. So, the generator wants to build up a 3-tuple to give to the discriminator, but it needs some way of deciding if it's turning the static into a similar node, or a different node. The way the authors do this is by dividing the dimension of that static in two, such that the first half of the static that's turned into x is also the first half of the static that's turned into y. That way, the two nodes have similar latent input, so--hopefully--they'll have similar latent output! Cool, right? 

Anyway, the authors only tested this on node classification, which makes sense, since it only really works on graphs that have node features. But I am interested entirely in link prediction, which they didn't test for. If you want to do something right, you've got to do it yourself. Well... maybe not entirely right, as you'll see. 

I'm not totally sure how I should have gone about building a train test validate partition on the data. See, the positive samples we give the discriminator are based entirely on what nodes do or don't have edges in the data. If we pull out some edges for testing and validation, it's just gonna learn that those edges don't exist. It's not super helpful. I don't know, maybe I should go back and try that, but it really didn't seem like a great idea. On the other hand, if nodes are pulled out, that wipes out entire rows of the adjacency matrix, and again, becomes a big mess to partition the edges. So, seeing as how the method is entirely unsupervised anyway, I just trained it on the whole dataset[<sup>5</sup>](#fn5). 

It did pretty well. As well, in fact, as the ARGAE. However, increasing the embedding dimensions didn't help much. On the same note, lowering didn't do all that much either. It's a solid architecture, it's just hard to test properly. I'm really not sure if it would have done as well if I could have found a better means of partitioning the input data. 

It's also worth noting, that unlike in the paper, I ran PCA on the node features (which is the only way the nodes are represented in the model) before giving them to the discriminator. This helped quite a bit, actually. It brought the AUC from a low 80 to about a 92, which is a win in my book. 

I really like this framework, I just wish there was something like it for graphs without features. 

<br/>

#### Footnotes

<a name=fn1></a><sub>1. I don't think this was intentional to hide anything. Node2Vec was closest to LINE in their paper, which ARGAE beat handilly. I think node2vec hadn't been published at the time ARGAE was being written, or it was just before they submitted it or something.</sub>

<a name=fn2></a><sub>2. I suspect this has something to do with "the law of large numbers", where if you mix enough distributions together, it just becomes Gaussian. But I'm really not sure.</sub>

<a name=fn3></a><sub>3. Why is it that as soon as you think of a cool, new, idea, 15 different researchers already published in KDD about it? Oh well.</sub>

<a name=fn4></a><sub>4. Okay, I admit, this network also isn't very adversarial. You could probably do this without generating fake samples, and it would work just as well.</sub>[<sup>6</sup>](#fn6)

<a name=fn5></a><sub>5. Please forgive me Professor Barbar&#225;. I know you said not to do that in my undergrad. I didn't forget. 

<a name=fn6></a><sub>6. Hmm. Come to think of it, I should probably do that right now instead of writing this goofy README with nested footnotes...</sub>