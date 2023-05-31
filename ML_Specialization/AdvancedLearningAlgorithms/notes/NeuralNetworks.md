# Neural Networks

Algorithm trying to mimic the brain

![Neural Network Model](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0n1Qw6VlRwchLKasr6RDMg.png)


### Structure

Neural networks work with $layers$. Each layer has a certain quantity of nodes, usually chosen by the user and each node has its own small "model", composed of:
- Weights
- Bias
- Activation function
- Output

![Structure of a neuron](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qQPpdtR0r1APiEfTqN74aA.png)

The first layer (called input layer) receives data from user. The Following layers compute values from previous layer, with a weight for each value in the previous layer. The last layer is called the output layer, and it is usually a single node giving us an output probability.

Layers in between input and output layers are called hidden layers, and they compute/look for attributes or features (like feature engineering) automatically, looking for the ones that help the model achieve the best results.

Two main steps are used in neural networks, forward propagation and backward propagation.

## Forward propagation

When we evaluate the model, we call it $forward propagation$.
Given an input, we calculate activation for all nodes in the network, layer by layer, until we compute the values in the output layer

## Back propagation

To train our model, we use back propagation. This is done by calculating derivatives and adjusting our parameters, traversing the *computational graph* backwards to get the partial derivatives using the multiplication rule. Running this algorithm backwards helps calculate them efficiently, achieving a linear time complexity for calculations.

## "Adam" Algorithm

> Adam: Adaptive Moment estimation

To increase efficiency, this algorithm provides a dynamic learning rate $\alpha$, which increases if steps are too small, or reduced if steps are too big. It uses learning rate for each parameter, instead of a single global one. 

Formally:

- If parameter keeps moving in the same direction, increase $\alpha$
- If parameter keeps oscillating, reduce $\alpha$


## Activation functions

There are alternatives to our activation function that can make our network more powerful. Some activation functions are:

- Sigmoid
  - $g(z) = \frac{1}{1+e^{-z}}$ 
- ReLU 
  - $g(z) = max(0, z)$ 
- Linear Activation Function
  - $g(z) = z$
- Softmax 
- Others (TanH, Leaky ReLU)

How do we choose which one to use?

### Output Layer

Natural choice, depending on what is our target $y$.
- Binary Classification: Sigmoid
- Regression (+/-): Linear Activation
- Regression (0/+): ReLU

### Hidden Layers

- ReLU is most common choice
  - Faster to compute
  - Faster training


## Types of Layers
- Dense Layer
  - Each neuron output is a function of all activation outputs of the previous layer.
- Convolutional Layer
  - Each neuron looks at a certain part of the previous layer's input
  - This can speed up computations, and it is less prone to overfitting.

## Vectorization

Matrix multiplications help us vectorize Neural Networks and make them much more efficient.

