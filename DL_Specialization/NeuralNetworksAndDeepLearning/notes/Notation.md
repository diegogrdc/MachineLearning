# Notation 

|Term| Description|
|---|---|
| $\vec{x}$ | Input Variables / Features |
| $y$ | Output Variable |
| $m$ | # of examples |
| $m_{train}$ | # of Training examples |
| $m_{test}$ | # of Test examples |
| $n_x$ | Size of input vector |
| $X$ | Matrix of size $(n_x, m)$, where each column represents an input vector $\vec{x}$ |
| $Y$ | Matrix of size $(1, m)$, where each column represents the output of an example $y$ |
|$?^{(i)}$| Superscript means that it refers to the $i$-th training example |
|$\alpha$| Learning rate |
|$?^{[i]}$| Superscript means that it refers to the $i$-th layer in a network |
|$a^{[i]}_j$| Activations of unit $j$ on layer $i$ |
|$a^{[0]}$| Inputs, activation of NN. Same as $x$ |
|$n^{[l]}$| Number of units in layer $l$ |
|$W^{[l]}$| Matrix of shape $(n^{[l]}, n^{[l - 1]})$. It stores all weight parameters between layer $l - 1$ and layer $l$ |
|$b^{[l]}$| Column vector of shape $(n^{[l]}, 1)$. It stores all bias parameters of layer $l$ |
|$A^{[l](i)}_j$| Matrix of shape $(n^{[l]}, m)$. Activation of unit $j$ in layer $l$ of training example $i$ |
|$L$| Number of layers in a NN |
|$g^{[l]}$| Activation function of layer $l$ |
