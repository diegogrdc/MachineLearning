# Linear Regression

> Fitting a straight line/plane given some input data

![Regression graph example](https://storage.googleapis.com/lds-media/images/440px-Linear_regression.svg.width-1200.png)

Useful to visualize data as a table, columns representing features (inputs) or label (output)

## Linear Regression Function

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

## Vectorization

> Vectorization: Process of transforming multiple operations acting on individual data elements, to a single operation that act concurrently on multiple data elements, speeding up execution and making code simpler

### Example:

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

## Cost Function

We can use a function that calculates the average of distance between data points and our model.

> $J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{i} - y^{i})^2$

This cost function is convex and can help us reach the global minimum without issues.

## Gradient update

We update our parameters and do the gradient descent until we converge or get to max iteration limit. We use following formulas (based on the defined above cost function):

> $\frac{dJ(\vec{w}, b)}{dw^{(j)}} = \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(x^{(i)})-y^{(i)}) * x^{(i, j)}$

> $\frac{dJ(\vec{w}, b)}{db} = \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(x^{(i)})-y^{(i)})$

### Note

> Other possible cost functions (and therefore gradient derivatives) are also possible to use

## Regularized Linear Regression

To regularize linear regression, we can:

- Adjust cost function
  -  > $J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{i} - y^{i})^2 + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$
- Adjust gradient descent weight functions:
  - > $\frac{dJ(\vec{w}, b)}{dw^{(j)}} = \frac{1}{m} \sum_{i=1}^{m}[(f_{\vec{w},b}(x^{(i)})-y^{(i)}) * x^{(i, j)}] + \frac{\lambda}{m}*w_j$
- $b$ function stays the same, as we don't regularize it 