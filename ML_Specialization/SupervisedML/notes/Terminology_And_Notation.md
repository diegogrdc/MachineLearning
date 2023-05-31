# Terminology and Notation

## Terminology

Relevant ML Terms

|Term|Definiton|
|---|---|
|Machine Learning| Field of study that gives computers the ability to learn without being explicitly programmed.|
|Training Set| Labeled data used to train the model|
|Linear Regression| Fitting a straight line/plane given some input data|
|Logistic Regression| Predict the probability of a binary (yes/no) event occurring, outputting a number between 0 and 1|
|Gradient|Given a point in the model, direction of steepest descent from that point|
|Gradient Descent| Doing many steps following the gradient until getting to a minimum in our *cost function*|
|Vectorization| Process of transforming multiple operations acting on individual data elements, to a single operation that act concurrently on multiple data elements, speeding up execution and making code simpler|
|Cost function| Function that measures how well a model fits training data|
|Loss| Measure of the difference of a single example to its target value|
|Cost| Measure of the losses over the training set|
|Learning Rate| Parameter that determines step size on each update to parameters using the gradient|
|Feature Engineering| Using intuition to design new features by transforming or combining original features|
|Overfitting|Model that fits training data extremely well, but does not seem to generalize. It has high variance|
|Underfitting|Model does not fit the data very well. It has high bias|
|Generalization| Model ability to make good predictions even in not seen before data|
|Regularization|Make parameter values small to reduce overfitting|


## Notation 
|Term| Description|
|---|---|
| $x$ | Input Variable / Feature |
| $x_j$ | $j^{th}$ feature |
| $n$ | Number of Features |
| $\vec{x}^{(i)}$ | Features of $i^{th}$ training example |
| $x^{(i)}_{j}$ | value of feature $j$ in $i^{th}$ example |
| $y$ | Output / Target Variable |
| $\hat{y}$ | Prediction of the model for y |
| $m$ | Number of Training examples |
| $(x, y)$ | single training example |
| ($x^{i}, y^{i})$ | i<sup>th</sup> training example|
|$w$| parameter: weight|
|$b$| parameter: bias|
|$\alpha$| learning rate|
|$J(params)$| Cost function |
|$L(params)$| Loss function |