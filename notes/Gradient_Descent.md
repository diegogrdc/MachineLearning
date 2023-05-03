# Gradient Descent 

Given:

- A model
  - Example: $f(x)$
- Params
  - Example: $w, b$
- Cost Function
  - Example: $J(w, b)$

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

- $w=w-\alpha \frac{d}{dw} J(w,b)$

where:

- $w$: a parameter 
- $\alpha$: learning rate
- $\frac{d}{dw} J(w,b)$: partial derivative of cost function (gradient)

Remember, we do this for each parameter simultaneously.

## Learning Rate

> Learning Rate: Parameter that determines step size on each update to parameters using the gradient

Choice of learning rate has a huge impact on efficiency of gradient descent. 
- If $\alpha$ is too small, we take really small steps, so descent will be really slow.
- If $\alpha$ is too large, we might overshoot the minimum and fail to converge.

A fixed learning rate might leave us stuck in a local minimum.

## Cost Functions 

> Cost function: Function that measures how well a model fits training data

Some possible cost functions are:

### Mean Squared Error (MSE)

#### Formula
$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{i} - y^{i})^2$

## Types of Gradient Descent

- "Batch" gradient descent: Each step of gradient descent uses all the training examples
- "Mini-batch" gradient descent
- Stochastic gradient descent
