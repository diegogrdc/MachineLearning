# Decision Trees

Decision trees are a supervised ML algorithm used to categorize or make predictions based on how a previous set of questions were answered. It mimics human thinking and it is easy to understand and interpret the results.

It is a classifier that uses categorical discrete inputs

It tends to work well on tabular structured data (Spreadsheets, Tables) but not so much on unstructured data (Audio, text, images). It is usually faster than NNs.

![Decision Tree Example](https://dimensionless.in/wp-content/uploads/2018/11/Picture1-1.png)



## Learning Process

For each node, we need to choose what feature to split on. That decision is usually made to maximize purity.

We also need to know when to stop splitting. This can be done with purity a threshold, a maximum depth, when improvements are below a threshold or when number of examples in a node are below a threshold

## Measuring purity

> Entropy: Metric that measures the impurity in a system.

The entropy formula is:
> $-\sum_{i=1}^N{P_i*log_2(P_i)}$

Where $P_i$ is the probability of choosing class $P_i$ and $N$ is the number of classes

## Choosing a split

> Information gain: Reduction of entropy

When building a decision tree, thee way we will decide which feature to split on, will be based on the one that gives us the most information gain.

To do this, we will take a weighted average between the entropy in the left and right sides (entropy times # of examples in node) and calculate how much it reduced the parent node's entropy. That is what we call *information gain*. Formally:

> Information gain = $H(p_1^{root}) - (w^{left}*H(p_1^{left})+w^{right}*H(p_1^{right}))$

where:
- $H(x)$ is the entropy of $x$
- $p_1$ is fraction of examples that have a positive label ($1$)
- $w$ is the fraction of total root examples that ended up in a certain side

## Process of Building the tree

We build decision trees by recursive splitting.

- Start with all examples at the root
- Calculate information gain for all possible features, and pick the highest
- Split dataset accordingly and create left and right branches
- Repeat splitting process until stopping criteria is met
  - Node is a 100% class (0 entropy)
  - Splitting would exceed max depth
  - Information gain is less than a threshold
  - \# Of examples is less than a threshold
- If we have a leaf node, we create a label node

## Prediction

When we want to predict the class of a new example, we just go down the tree. On each decision, we follow our example's information, until we reach a leaf node.


## Refinements 

### One hot encoding

If a categorical feature can take on *k* values, we create $k$ binary features (0 or 1 valued), so we can keep our tree being binary. Out of all new features, each example will have exactly one feature turned on, the *hot* feature. 

> Note: This can also be used for NN to transform all categorical data into binary features.

### Continuous values

When splitting a continuous value, we can sort values increasingly, and try all midpoints between values as possible thresholds. All values smaller than the threshold go left, and the rest goes to the right. We can get the information gain of this split, and compare it as any other feature when choosing the node feature. 


## Regression trees

If we want trees to predict a number instead of a category we can use regression trees.
Instead of having labels on the leafs, we can have an average of the examples that are part of the leaf tree. This average comes from the output values.

When a new example comes, the tree would predict the average of the leaf node it ended on.

When splitting, we try to reduce for a reduction of variance of *y* instead of reduction of entropy. 

## Tree Ensembles

An issue with a single tree is that it is really sensible at changes in the data. A solution to make the algorithm more robust is to build a lot of trees, called a *tree ensemble* 

When we get a new example, we get the labels on each tree, and we choose the one that was outputted the most.

To create tree ensembles, we can create different datasets using sampling with replacement. This will create multiple different, but similar training sets, giving us different trees that can help us have a more robust algorithm.

### Random Forest Algorithm

Given a training set of size $m$, we will create $B$ training sets of size $m$ using sampling with replacement. ($B$ can be chosen by the engineer, but typical choices around 64 to 128 trees). For each dataset, we train a decision tree.

However, when choosing a node feature, if $n$ features are available, we pick a random subset of size $k$ ($k < n$) features, and allow our algorithm to choose only from that subset of features. ($k$ is usually near $\sqrt{n}$). 

This small change makes us explore more possibilities and get a much more robust algorithm.


### Boosted Trees

A small modification to the random forest algorithm can make it work much better. 

When sampling, instead of picking from all examples uniformly, we make it more likely to choose from misclassified examples from previous trained trees.

### XGBoost

> XGBoost: Open Source implementation of Boosted Trees, with a fast efficient implementation and highly competitive algorithm. 

#### Classification:

```
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### Regression:
```
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```