# Logistic Regression as a Neural Network

![Logistic Regression NN Img](https://carpentries-incubator.github.io/ml4bio-workshop/assets/logit_nodes.png)

Logistic regression is an algorithm of binary classification. This means all output values $y$ are either *0* or *1*. Now, we want to obtain a mapping function, that given $x$, it gets $\hat{y} = P(y = 1 | x)$, the probability that the training example $x$ has label $y = 1$.

The output is calculated by 

> $\hat{y} = \sigma(w^T \cdot X + b)$

Where:

> $\sigma(z) = \frac{1}{1+e^{-z}}$

So, logistic regression has to find the parameters $\vec{w}$ and $b$, so that $\hat{y}$ becomes a good estimate of $y = 1$, based in the training examples that we feed to it.

## Cost Function

To evaluate our model we can define the following loss function:

> $L(\hat{y}, y) = - ( y log(\hat{y}) + (1-y)log(1 - \hat{y})) $

And the following cost function:

> $J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})$

## Gradient Descent

Using gradient descent, we can improve our parameters $w, b$. The goal is to find $w, b$ that minimize $J(w, b)$. This is done by taking steps in the steepest downhill direction (gradient) iteratively. This works well on logistic regression, as the function $J(w, b)$ is convex, and will always have a global minimum.

Formally, on each iteration:

> $w = w - \alpha \frac{dJ(w, b)}{dw}$ 

> $b = b - \alpha \frac{dJ(w, b)}{db}$

This is done by two steps: forward propagation step, and then a backward propagation step.

### Computational graph

A computational graph is a directed graph where the nodes correspond to operations or variables. If we have this graph, we can forward propagate and evaluate an example to get $\hat{y}$ and the cost $J(w, b)$. Having the cost, we can do the backward propagation step to calculate derivatives and do gradient steps with our parameters (to get $dw$ and $db$).

### Derivatives

By using calculus, we can get the derivative formulas for our parameters. Remember $z = w^T \cdot X + b$:

> $\frac{dJ(w, b)}{dz} = a - y$

> $\frac{dJ(w, b)}{dw^{(i)}} = x^{(i)} dz$

> $\frac{dJ(w, b)}{db} = dz$

The above formulas work for a single example. For derivatives for $m$ examples, we get an average over the sum of all derivatives for each parameter

## Vectorization

> Vectorization: Art of getting rid of explicit for loops in code

When doing multiple for loops calculations, we can take a lot of time to execute those calculations. To improve efficiency, we can use *vectorization techniques*.

### Vectorized Dot Product 

When computing $w^T \cdot x + b$, instead of doing:

```
z = 0
for i in range(n_x):
    z += w[i] * x[i]
z += b
```

We vectorize like the following:

```
z = np.dot(w, x) + b
```

### Vectorized Logistic Regression Forward Prop

$Z$ will contain all $m$ forward propagation results given parameters $w$ and $b$. Z will have shape $(1, m)$

```
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
```

### Vectorized Logistic Regression Backward Prop

$dZ$ will contain all $m$ $dz$ values, with a shape of $(1, m)$. dW will also have a shape $(n_x,1$), each representing the gradient for each weight. Finally, $db$ is a scalar value with parameter $b$ gradient.
```
dZ = A - Y
dW = 1 / m * np.matmul(X, dZ.T)
db = 1 / m * np.sum(dZ)
```