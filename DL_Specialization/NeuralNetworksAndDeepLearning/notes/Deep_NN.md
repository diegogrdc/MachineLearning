> Read Shallow NN first

# Deep Neural Network

![Deep NN Img](https://s7280.pcdn.co/wp-content/uploads/2020/07/Two-or-more-hidden-layers-comprise-a-Deep-Neural-Network.png)

Depending on the number of hidden layers, we call our models a bit differently:
- 0 hidden layers: Logistic Regression
- 1 hidden layer: Shallow NN
- 2 or more hidden layers: Deep NN

## Why Deep Networks Work

Deep networks are really powerful. Each layer can detect a pattern. For example, in image recognition, first layer can learn to detect edges. Next one starts detecting shapes from those edges, and in the following layers, more complex patterns. As we make our layer deeper, we are giving it the chance to find much more complex patterns.

![Features in Deep Network](https://static.packt-cdn.com/products/9781787124769/graphics/B05883_01_14-2.jpg)

In audio, we can see it as first layer computing low level audio, then phonemes, then words, and finally sentences.d 

## Forward Propagation

Remember $A[0] = X$. Then, for each layer (loop):

- $Z^{[l]} = W^{[l]} A^{[l - 1]} + b^{[l]}$
- $A^{[l]} = g^{[l]}(Z^{[l]})$

Each steps inputs $A^{[l - 1]}$ and outputs $A^{[l]}$, and cache $Z^{[l]}$ to calculate gradients

On shapes, we have:

- $shape(W^{[l]}) = (n^{[l]}, n^{[l - 1]})$
  - same for $dW^{[l]}$
- $shape(b^{[l]}) = (n^{[l]}, 1)$
  - same for $db^{[l]}$
- $shape(z^{[l]}) = (n^{[l]}, 1)$
  - same for $a^{[l]}$
- $shape(Z^{[l]}) = (n^{[l]}, m)$
  - same for $A^{[l]}$

## Backward Propagation

For each layer (loop):

- $dZ^{[l]} = dA^{[l]} * g^{[l]'}(Z^{[l]}) $ 
- $dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l - 1]T}$
- $db^{[l]} = \frac{1}{m} \sum dZ^{[l]} $
  - `np.sum(dZ[l], axis=1, keepdims=True)`
- $dA^{[l - 1]} = W^{[l]T} dZ^{[l]}$

Each steps inputs $dA^{[l]}, dZ^{[l]}$ and outputs $dA^{[l - 1]}, dW^{[l]}, db^{[l]}$ 

## Parameters vs Hyperparameters

Hyperparameters are parameters that control what will result in our parameters after running the algorithm. Parameters depend on hyperparameters. For example, changing number of iterations or number of hidden layers will affect the final choices of $W, b$.

Parameters:
- $W$
- $b$

Hyperparameters: 
- $\alpha$
- \# of iterations
- \# of hidden layers $L$
- \# of hidden units $n^{[l]}$
- Choice of activation function

Choosing hyperparameters is a very empirical process. We try out different values and try to find the one that works best.