# Gradient Descent 

Given:

- A model
  - Example: $f(\vec{x})$
- Params
  - Example: $\vec{w}, b$
- Cost Function
  - Example: $J(\vec{w}, b)$

Our goal to optimize the model is to minimize our *cost function* by choosing values for our *parameters*

To do this automatically, we use gradient descent.

> Gradient: Given a point in the model, direction of steepest descent from that point. 

> Gradient Descent: Doing many steps following the gradient until getting to a minimum in our *cost function*.

## Conceptual Steps

- Initialize our parameters into a random/arbitrary value.
- Keep changing/updating *parameters* to reduce the *cost function*.
- Stop when we settle at or near a minimum.

## Algorithm

To update our params, we can do the following:

- $p=p-\alpha \frac{d}{dp} J(p,b)$

Where:

- $p$: parameters 
- $\alpha$: learning rate
- $\frac{d}{dp} J(p,b)$: partial derivative of cost function (gradient)

Remember, we do this for each parameter simultaneously.

## Learning Rate

> Learning Rate: Parameter that determines step size on each update to parameters using the gradient

Choice of learning rate has a huge impact on efficiency of gradient descent. 
- If $\alpha$ is too small, we take really small steps, so descent will be really slow.
- If $\alpha$ is too large, we might overshoot the minimum and fail to converge.

Some things to consider:

- A fixed learning rate might leave us stuck in a local minimum.
- You can use a learning curve graph to check if it is converging.
  - If graph is flaky or increasing, alpha might be too big (or bug in code).
- Some good values to try are: $[..., 0.001, 0.01, 0.1, 1, ...]$


## Cost Functions 

> Cost function: Function that measures how well a model fits training data

Some possible cost functions are:

### Mean Squared Error (MSE)

#### Formula
$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{i} - y^{i})^2$

## Types of Gradient Descent

Some types of gradient descent are:

- "Batch" gradient descent: Each step of gradient descent uses all the training examples
- "Mini-batch" gradient descent
- Stochastic gradient descent

## Feature Scaling

When you have different features with really different ranges of values, it can slow down gradient descent.

If we rescale features to take in comparable range of values, gradient descent can be much faster.

To do it we can:

- Divide by maximum
  - $\frac{x_{i}}{max}$
- Mean Normalization
  - $\frac{x_{i}-\mu_{i}}{max-min}$
- Z-score normalization
  - $\frac{x_{i}-\mu_{i}}{\sigma_{i}}$

## Feature Engineering

> Feature Engineering: Using intuition to design new features by transforming or combining original features.