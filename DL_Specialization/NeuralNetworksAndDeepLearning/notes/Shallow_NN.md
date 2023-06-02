# Shallow Neural Network

## Neural Network Representation

![NN representation](https://miro.medium.com/v2/resize:fit:782/1*CfdaqnNb6RHLzPJTt1UXjQ.png)

A Neural Network has multiple components:
- Input Layer
- Hidden Layer(s)
- Output Layer

Each layer have parameters associated with it. Each layer has some weights $w$ and a bias $b$. When counting number of layers, we don't count the input layer (input layer is called layer 0).

## Computing outputs

Each node has two steps of computations. First half computes $z = w^T \cdot x + b$, and the second half computes the activation $a = g(z)$. 

> Note: A logistic regression is equivalent to one NN output node. That's why a logistic regression is 1 layer NN (no hidden layers).

This equations can be vectorized to execute all layer operations without a for loop. 

Then, for each layer:

> $Z[l] = W^{[l]} A^{[l - 1]} + b^{[l]}$


> $A[l] = g(Z[l])$

Assuming we have matrixes with all examples stuck, we just need these steps for each layer to calculate the forward propagation for all examples.

## Activation Functions

We have used the sigmoid activation function, but sometimes other choices can work better. So we get to choose $g(z)$ for any node. Some choices of activation are:
- Sigmoid
  - $g(z) = \frac{1}{1+e^{-z}}$
  - Output ranges between 0 and 1
  - Usually only used for output function
- TanH
  - $g(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}$
  - Output ranges between -1 and 1
  - Usually better than sigmoid for hidden layers
- ReLU
  - $a = max(0, z)$
  - Faster gradient descent, as slopes are never that small. 
  - Default for hidden layers
- Leaky ReLU
  - $a = max(0.01z, z)$
- Linear
  - Might be used in outputs when we need numbers

### Derivatives of activation functions
- Sigmoid
  - $\frac{d}{dz} g(z) = g(z) (1 - g(z))$
- TanH
  - $\frac{d}{dz} g(z) = 1-(tanh(z))^2$
- ReLU
  - $\frac{d}{dz} g(z) = 0$ if $z < 0$ else $1$
- Leaky ReLU
  - $\frac{d}{dz} g(z) = 0.01$ if $z < 0$ else $1$

## Gradient Descent

To use gradient descent in Neural Networks:

- Repeat
  - Forward propagation
    - Compute predictions for each layer
    - $Z^{[l]} = W^{[l]} A^{[l - 1]} + b^{[l]}$
    - $A^{[l]} = g(Z^{[l]})$
  - Backward propagation
    - Calculate derivatives for each layer
    - $dZ^{[l]} = A - Y$ (last layer) 
    - $dZ^{[l]} = W^{[l + 1]} dZ^{[l + 1]} * g^{[l]'}(Z^{[l]}) $ (hidden layers)
    - $dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l - 1]T}$
    - $db^{[l]} = \frac{1}{m} \sum dZ^{[l]} $
  - Update parameters
    - Adust $W$ and $b$· with calculated derivatives


### Cost Function

> $J(W, B) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}, y)$

### Random Initialization

It is necessary to initialize our parameters randomly, as initializing them with 0 will not work. This is because all parameters will be symmetrical, and we will have the same parameters repeated, instead of multiple parameters that help us with computation. As a note, $b$ could be initialized as $0$.

> `np.random.randn((shape)) * 0.01`