# Supervised Learning Algorithms

Some notes on supervised learning algorithms

## Linear Regression

> Fitting a straight line/plane given some input data

![Regression graph example](https://storage.googleapis.com/lds-media/images/440px-Linear_regression.svg.width-1200.png)

Useful to visualize data as a table, columns representing features (inputs) or label (output)

### Linear Regression Function

Model needs a function to predict output ($\hat{y}$)

Given the parameters:
- Weights: $\vec{w}=[w_{1}, w_{2}, ..., w_{n}]$
- Bias: a number $b$ 
- Inputs/Features:  $\vec{x}=[x_{1}, x_{2}, ..., x_{n}]$

The function in linear regression is:

> $f_{\vec{w}, b}(\vec{x}) = w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n} + b$

Or in simpler terms:

> $f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$

Where "$\cdot$" is the dot product of two vectors

### Vectorization

> Vectorization: Process of transforming multiple operations acting on individual data elements, to a single operation that act concurrently on multiple data elements, speeding up execution and making code simpler

#### Example:

$\vec{w} \cdot \vec{x}$

Non Vectorized: 
```
f = 0
for j in range(0, n):
  f = f + w[j] * x[j]
f = f + b 
```
Vectorized:

```
f = np.dot(w, x) + b 
```
