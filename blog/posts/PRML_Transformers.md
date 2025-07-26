


### Why? 


I've deicded, just now, a few minutes ago, that I'm going to dedicate some time every week to actually formally writing down the notes I take while reading papers to *formalize* my thoughts into something coherent rather than a chicken-skratch notebook from hell. I hope this will also put me into a better regime of actually reading papers which I should have read a long time ago. To start, this is one which I should've read a long time ago... *Attention Is All You Need*.


## I. Introductory Remarks

The grand idea behind the transformer, as hinted in the very first paragraph, is to go beyond a sequential computation by computing token interactions in parallel rather than step-by-step, which is limited (e.g. sequential computation). Although the attention mechanism had existed before this paper, every system that used attention still wrapped it around a standard recurrent network (e.g. an RNN or LTSM), so you'd pay attention but you'd also step through the sequence one time-step at a time. However, before this paper, very few cases had people try to use attention without recurrence at all—yet the authors propose that attention by itself is powerful enough to capture long-range connection.




## II. Self Attention

We should recall facts and intuitions about the self attention mechanism—you know, for the readers' sake and not mine. So, why does self attention *exist*? (This already existed if you have a platonic inclination of things — or if you're a Max Tegmark fan, this is all that exists and nothing else does.) Well, self attention is about the most intuitive idea that you can come across: the meaning of words (ahem, tokens) are determined by those that sit around them. The next representation of token $i$ should be a *mixture* of other token whose relevance depends on their content, e.g. "dog" should pull from "barks" even if it is 30 tokens away. In a sense, each token $t_i$ in a sequence will give a *query* $q_i$ that asks, "which other tokens $t_j$ are relevant to me?" Relevance is measured by a similar score $q_i^\intercal k_j$ with each token's *key* $k_j$. 

These query, key, and *value* are the essential things we need to know. To beat the drum again, each token emits a **query** ("what I need"), each token provides a **key** ("how to find me"), and every token carries a **value** ("what you get if you pick me"). A token should read a *mixture of values* weighted by how well its query matches others' keys. 

The standard formulation goes as follows. Hold on, my food just got here...

Let $X \in \mathbb R^{n \times d_{\text{mode}}}$ be the sequence, where rows are token embeddings—$n$ is the sequnece length (number of tokens are tokenization) and $d_{\text{mode}}$ is the model width $=$ dimensionality of each token's embedding/hidden state (e.g. 512 or 768). If $x_i \in \mathbb R^{d_{\text{model}}}$ is the vector for the $i$-th token, after adding positional information, then 
$$
X = \begin{bmatrix}
x_1^\intercal \\
x_2^\intercal \\
\cdots \\
x_n^\intercal
\end{bmatrix}^\intercal \in \mathbb R^{n \times d_{\text{model}}}
$$

To emphasize this setup, we'll carry an an example with us. Suppose we have the sentence "Time flies fast." and it tokenizes to 4 tokens. Here $n=4$ and pick $d_{\text{model}} = 6$. After the embeedding lookup and adding positional encoding you have $X = \mathbb R^{4 \times 6} = \begin{bmatrix} x_1^\intercal \\ x_2^\intercal \\ \cdots \\ x_n^\intercal \end{bmatrix}^\intercal$. Choose learned linear maps $W_Q \in \mathbb R^{d_{\text{model}} \times d_k}$, $W_K \in \mathbb R^{d_{\text{model}} \times d_k}$, and $W_V \in \mathbb R^{d_{\text{model}} \times d_v}$ — the subscripts $Q$, $K$, $V$, refer to query, key, value, respectively, as we mentioned a few paragraphs ago (same goes for the little subscripts of $d_\ast$). Define $Q = X W_Q $, $K = X W_K$, $V = X W_V$. 

In the context of our little tokens, $q_i = W_Q x_i$, $k_i = W_K x_i$, and $v_i = W_V x_i$. For simplicity, it's common to set $d_k = d_v$. These weights matrices are learnable parameters of the attention layer, and they allow the model to learn the optimal transformations for comparing and combining tokens. But to determine how much attention position $i$ should pay to position $j$, we need a similarity score between the query from position $i$, $q_i$, and the key from position $j$, $k_j$. A simple and effective choice for a similarity function is the dot product, 
$$\text{score} (q_i, k_j) = q_i \cdot k_j = q_i^\intercal k_j$$
This score, might be written $e_{ij}$, measures the compatibility or alignment between the query and the key. A large positive dot product would suggest that $k_j$ is highly relevant to the query $q_i$. But these raw scores can have arbitrary magnitudes, and to turn them into a well-behaved set of weights that sum to 1, i.e. a probability distribution, we apply the softmax function across all keys for a given query $q_i$: $$\alpha_{ij} = \text{softmax} (e_{i1}, e_{i2}, \ldots, e_{in}) = \frac{\exp(e_{ij})}{\sum_{\ell =1 }^n \exp (e_{i \ell})}$$ This $\alpha_{ij}$ is the *attention weight* representing the proportion of attention that the output at position $i$ pays to the input at position $j$. The final output for a position $i$, which we write $z_i$, is the weighted sum of all the value vectors $v_j$ in the sequence, using the attention weights $\alpha_{ij}$ we just computed $$ z_i = \sum_{j=1}^n \alpha_{ij} v_j$$
So this is a contextualized representation of $x_i$ having "looked" at all other $x_j$ and synthesized their information based on relevance via the combination weights determined dynamically by the query-key interactions.



Okay, good, that's the vector format of things, but to be more computationally efficient we must revert back to matrices; we stack the individual vectors into matrices. The matrix of all pairwise dot product scores is then simply $Q K^\intercal$, where the $(i,j)$-th entry of this matrix is $q_i^\intercal k_j$. Now I would be remissed to not mention the scaling factor of $\frac{1}{\sqrt{d_k}}$. As the dimension of the keys, $d_k$, increases, the magnitude of the dot products $q_i^\intercal k_k$ tends to grow. So, if we assume the components $q_i$ and $k_j$ are independent random variables with mean 0 and variance 1, their dot product has a mean of 0 and a variance of $d_k$. Large inputs into the softmax can push its gradients to be vanishingly small, damaging learning. To counteract this, the dot products are scaled down by the standard deviation of their distribution $\sqrt{d_k}$. 

The full self-attention formula is therefore written as 

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\intercal}{\sqrt{d_k}} \right) V$$

The output is a matrix $Z \in \mathbb R^{n \times d_v}$ where the $i$-th row is the contextualized vector $z_i$. 


Just to be clear (I believe in a concrete understanding rather than a facetious, mechanical implemenation-level knowledge of things), you form a score $S = QK^\intercal /\sqrt{d_k}$ if shape $n \times n$ where $S_{ij}$ measures how much token $i$ should pay attention to token $j$, and $\text{softmax} (S)$ normalizes each row into a probability distribution over the $n$ positions. The matrix $V$ is also $n \times d_v$ where row $j$ of $V$, say, $v_j$, is the "content vector" for position $j$. Then when you multiply the $n \times n$ attention weights by $V$, each output row $z_i = \sum_{j=1}^n \alpha_{ij} v_j$. Intuitively, $Q K^\intercal$ decides *how much* to look at each other position; $V$ is *what* information you actually pull in. 


### II.1 Multi-Head Attention

This is actually *new* in the paper, but it fits better in this recap section. Basically, a single attention mechanism might learn to focus on a particular type of relationship, e.g. syntactic dependencies $\rightarrow$ "The chicken didn't cross the road because it was too tired" is an example where attention may learn that "it" is more attached to "chicken" rather than "road" or "The". But to allow the model to jointly attend to information from different representation subspaces at different position, the authors introduce **Multi-Head Attention**. 

The idea is to run $h$ different attention mechanisms, or "heads", in parallel. Each head has its own set of learnable weight matrices $ \{W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \}_{i=1}^h$ and 

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)} )$$ 

Typically the dimensions are chosen such that $d_k = d_v = d_ {\text{model} }/h$—this keeps the total computation similar to a single head with full dimensionality. The ouputs of the $h$ heads are then concatenated and passed through a final layer projection, governed by a weight matrix $W_O \in \mathbb R^{h d_v \times d_{\text{model}} }$, to produce a final output to the layer.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

This structural choice gives each head the ability to specialize in capturing different types of relationships within the sequence, making it way more expressive and powerful. 


### III. Transformer Architecture
