
# Logistic Regression

> Logistic Regression: Predict the probability of a binary (yes/no) event occurring, outputting a number between 0 and 1

Used in binary classification problems like:
- Is email spam? Yes/No
- Is tumor malignant? Yes/No

![Logistic Regression graph example](https://static.javatpoint.com/tutorial/machine-learning/images/logistic-regression-in-machine-learning13.png)

## Logistic Regression  Function
> $f_{\vec{w}, b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b)$

This model comes from:

- Model/Function
  - > $z = \vec{w} \cdot \vec{x} + b$
- Sigmoid Function
  - > $g(z) = \frac{1}{1+e^{-z}}$
- Outputs value between 0 and 1

## Decision boundary

> Decision boundary: Line where $z = \vec{w} \cdot \vec{x} + b = 0$

Logistic regression outputs a probability, so we need to pick a threshold to decide if a probability is 1 or 0. Usually this is 0.5 (heuristic), and that means all $g(z) \ge 0$ will be $1$. This translates to $f(\vec{x}) \ge 0$, and that is where the decision boundary equation comes from.

Usually a line, but can use polynomials to create more complex decision boundaries.

## Cost Function

The cost function will use a loss function, giving us:

> $J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m} L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)})$

Or, expanded (explained below):

> $J(\vec{w}, b) = -\frac{1}{m}\sum_{i=1}^{m} [y{(i)}*log(f_{\vec{w}, b}(x^{(i)})) + (1 - y^{(i)}) * log(1 - f_{\vec{w}, b}(x^{(i)}))]$

### Logistic loss function

This function penalizes far answers. The further the prediction is from the true target, higher the loss.

> $L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) =$
- > $-log(f_{\vec{w}, b}(x^{(i)}))$
  - If $y^{(i)} = 1$
- > $-log(1 - f_{\vec{w}, b}(x^{(i)}))$
  - If $y^{(i)} = 0$

Or, all in one equation:

- > $-y{(i)}*log(f_{\vec{w}, b}(x^{(i)})) - (1 - y^{(i)}) * log(1 - f_{\vec{w}, b}(x^{(i)}))$

This will make sure the loss function will be convex, then making gradient descent a useful tool to improve our model. 


## Gradient update

We update our parameters and do the gradient descent until we converge or get to max iteration limit. We use following formulas (based on the defined above cost function):

> $\frac{dJ(\vec{w}, b)}{dw^{(j)}} = \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(x^{(i)})-y^{(i)}) * x^{(i, j)}$

> $\frac{dJ(\vec{w}, b)}{db} = \frac{1}{m} \sum_{i=1}^{m}(f_{\vec{w},b}(x^{(i)})-y^{(i)})$

## Regularized Logistic Regression

To regularize logistic regression, we can:

- Adjust cost function
  -  > $J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m} L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$
- Adjust gradient descent weight functions:
  - > $\frac{dJ(\vec{w}, b)}{dw^{(j)}} = \frac{1}{m} \sum_{i=1}^{m}[(f_{\vec{w},b}(x^{(i)})-y^{(i)}) * x^{(i, j)}] + \frac{\lambda}{m}*w_j$
- $b$ function stays the same, as we don't regularize it 