> Recommender Systems:  Algorithm that segments customers based on their data patterns, and targets them with personalized suggestions.

# Collaborative Filtering

To explore this algorithm, we will use the example of a movie rating system. This system has $n_u$ users and $n_m$ movies. Each user has rated a certain subset of movies, giving it from 0 to 5 starts.

Collaborative filtering recommends items to a user, based on ratings of users who gave similar ratings than you.

Given this dataset, our recommender system is looking to recommend movies to each user, looking for the most likely movies the user would like, using the user's past ratings for other movies to make a choice.

## Predicting with features

Let's say we also have features for our dataset. In our example, this could be having multiple features representing movie genres (romance, action, etc), each with a percentage of how much that movie classifies in that genre. 

We can use a a similar algorithm to linear regression to do this. Given:

- $r(i,j)$ = $1$ if user has rated the movie and $0$ otherwise
- $y^{(i,j)}$ = rating given by user $j$ on movie $i$ (if defined)
- $w^{(j)}, b^{(j)}$ = parameters for user $j$
- $x^{(i)}$ = feature vector for movie $i$
- $m^{(j)}$ = No. of movies rated by user $j$

For user $j$ and movie $i$, we can predict rating with:

> $w^{(j)} \cdot x^{(i)} + b^{(j)}$

And we need to learn the parameters $w^{(j)}$ and $b^{(j)}$ for each user

### Cost Function

We want to minimize the following cost function (similar to mean squared error). Then, to minimize parameters for user $j$:

> $J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n}(w_k^{(j)})^2$ 

To learn parameters for all users (cost to learn parameters for all users)

> $J(\vec{w}, \vec{b}) = \sum_{j=1}^{n_u} J(w^{(j)}, b^{(j)})$

## Predicting with no features 

If we do not have any features, we can try to come up with some just by analyzing our data. We can assume we already have parameters $\vec{w}$ and $\vec{b}$ for our users.

### Cost function

For a single feature, we would try to minimize:

> $J(x^{(i)}) = \sum_{i:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n}(x_k^{(i)})^2$

To learn all features:

> $J(\vec{x}) = \sum_{i=1}^{n_m} J(x^{(i)})$

We already have a way to learn our parameters, and also a way to learn the features. If we put them together, we get:

> $J(\vec{w}, \vec{b}, \vec{x}) = \sum_{(i,j):r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2$

If we minimize parameters and features altogether, we actually get a working algorithm. Now, how do we minimize this function?

### Gradient Descent

We can use a similar algorithm to linear regression gradient descent, when in each iteration we update all the parameters $\vec{w}, \vec{b}, \vec{x}$.

This can be done because we have multiple users having information for the same number of movies (items). If we had just one user, then we would use a linear regression. That is why we call the algorithm *collaborative*, as we can use information from multiple users to learn insights on the data, and that in turn gives us insights on the users themselves.

## Binary Labels

Usually, instead of users giving a rating, we can have a user liking something or not, interacting with it or not, buying something or not, did user see the item at least 30 seconds, etc. This is a binary label, usually based on user behavior. 

We can generalize our algorithm to work with binary labels. 

### From regression to binary classification

For binary labels, we can predict the probability of $y^{(i,j)} = 1$. This would be given by:

> $f(x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})$

where:

> $g(z) = \frac{1}{1+e^{-z}}$

### Loss function

We would need a new loss function, used for binary labels $y^{(i, j)}$.

> $L(f(x), y^{(i, j)}) = - y^{(i, j)} log(f(x)) - (1 - y^{(i, j)}) log(1 - f(x))$

This can be used in the cost function to do the gradient descent algorithm.

## Mean Normalization

If some users have no data, without mean normalization, the results will always be 0 (or really small), due to normalization. To fix this, for each movie $i$ (or item), we calculate the mean $\mu_i$ and subtract it from the value $y^{(i, j)}$. This will mean $0$ will represent our mean, and users with no info will have mean predictions. Don't forget to add the term $\mu_i$ when predicting, to keep the output values consistent with the inputs.

Doing this optimization also makes the algorithm run faster (a little bit). We can also normalize columns, depending on the problem we are trying to solve.

## Finding Related Items

The features $x^{(i)}$ of item $i$ are quite hard to interpret. However, to find similar items, we can find item $k$ with an $x^{(k)}$ similar to $x^{(i)}$ with the squared distance:

> $\sum_{l=1}^{n} (x_l^{(k)} - x_l^{(i)})^2= ||x^{(k)} - x^{(i)}||^2$

The smallest distances represent the most similar items to item $i$.

## Limitations

- Not good at a cold start problem
  - How to rank new items with no ratings?
  - Show something reasonable to new users with few ratings?