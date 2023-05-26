# Machine Learning Diagnostics

> Diagnostic: Test you run to gain insight into what is/is not working with a learning algorithm, to gain guidance into improving its performance

## Evaluating a Model

It is important to have a systematic way to evaluate a model's performance, so it is easier for us to measure when it is improving and what we are achieving. Some methods are:

### Train/Test Sets

We can split our dataset into train and test sets(e.g. 70/30, 80/20). We can train our model just feeding it data from the train set, and then we can evaluate it with unseen data from the test set. We will have a train error and test error, which we will try to balance, avoiding overfitting or underfitting of data.

## Model Selection

When looking into different models and unsure on which one to choose, we can use some methods to help us find the best one. 

### Train/Cross Validation/Test Sets

Split our data into train, cross validation and test sets (e.g. 60/20/10). We use the cross validation set to check the accuracy of different models. After splitting, we get the error for each set. We can choose the model that has the lowest error in the cross validation set. Finally, to report an estimate of a generalization error for the model, we use the test set. 

This is used to be *fair*. If we use the test set to make decisions, we can have an overly optimistic result. 

> Note: We can also use this to choose our regularization parameter $\lambda$.

## Bias and Variance

Usually models don't work as expected at the start. To improve them, we can look at the bias and variance. 

Remember:

- High variance (overfit)
  - A model has high variance when error in cross validation set is much higher than in the train set (train error might be low).
- High bias (underfit)
  - A model has high bias when the error in the rain set is high.
- Low bias and variance (just right)
  - Model has low bias and variance in both cross validation and train set.


![Bias/Variance](https://qph.cf2.quoracdn.net/main-qimg-f559dbd07509a2c986d10fcb13f3cb10-pjlq)


### Establishing a baseline

To define what is high and what is low, we need to establish a level of error that we can reasonably hope to get. We can do this by:
- Comparing to human level performance
- Competing algorithms performance
- Guess based on experience

## What to try?

If we identify some issues with our model, what do we do? 

If we have a high bias, we can:
- Try getting additional features
- Try adding polynomial features
- Decrease $\lambda$

If we have a high variance, we can:
- Get more training data
- Try smaller set of features
- Increase $\lambda$

## Neural Networks

NNs are usually low bias machines. If a NN has high bias, the solution is usually making a bigger network (more layers or more units). 

If it does not do well on the cross validation set, the solution would be to get more data.

The issues with this can be computational cost or not enough data available, so it is not always that simple.

> A larger NN will usually do better or at least as good as a smaller one, as long as regularization is chosen appropriately, but it is more costly computationally.

## Error Analysis

> Error Analysis: Manually examining wrong examples and categorize them in common traits

By doing this, we can get inspiration on what to try next, either add specific features, get more specific data, or even find errors not worth paying too much attention to. 

It is usually helpful for problems a human can understand, but difficult if a human cannot manually understand the output. 