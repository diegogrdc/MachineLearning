# Content Based Filtering

To explore this algorithm, we will use the example of a movie rating system. This system has $n_u$ users and $n_m$ movies. Each user has rated a certain subset of movies, giving it from 0 to 5 starts.

Content based filtering recommends items to a user, based on features of user and item to find a good match.

This means, users and items have features, and the algorithm tries to look for a match between them. That means, we are trying to find if a movie $i$ is a good match for a user $j$. For example:
- User features ($x_u^{(j)}$ for user $j$):
  - Age
  - Gender
  - Country
  - Movies watched
  - Average rating for genre
- Movie features ($x_m^{(i)}$ for movie $i$):
  - Year
  - Genre(s)
  - Reviews
  - Average rating

> Note: Categorical features can be one-hot encoded.

> Note: Number of features can be different between users and movies.

## Matching

To try to match a user and a movie, we can create two vector:

- $v_u^{(j)}$: Vector computed from $x_u^{(j)}$, giving us some general information on the user $j$, that we can match try to match with a movie. 
- $v_m^{(i)}$: Vector computed from $x_m^{(i)}$, giving us some general information on movie $i$, that we can match try to match with a user.

These two vectors need to have the same size. This is because, when we are trying to match a user and a movie, we can evaluate the dot product of both vectors, and use that result to decide if it is a good match.

## Deep Learning for Content Based Filtering

Given the input $x_u$, we could create a NN that outputs a layer of $k$ features. Similarly, we can create another NN for $x_m$ and output a layer of $k$ features. This outputs will be our $v_u$ and $v_m$, that we can evaluate and compare with each other. The internal structure of each NN is independent. We just need the output layer to have the same number of units. 

Formally, our prediction will be $v_u \cdot v_m$. 

We can create an architecture, where we have just one network, joining the two NN output layers by a dot product, having a single final output. 

### Cost Function

Cost function will be really similar to collaborative filtering. We will train both users and movies in the same network.

> $J = \sum_{(i,j):R(i,j)=1} (v_u^{(j)} \cdot v_m^{(i)} - y^{(i,j)})^2 + $ NN regularization term 

### Similarities

Having the vectors $v_m$, we could find two similar movies with $||x^{(k)} - x^{(i)}||^2$

## Large Datasets

Today, datasets have millions (or tens of millions) of data points. The algorithm described above is computationally expensive and might not be as efficient as needed, as each time a new user joins, or an item, we would need to run a lot of computations. 

### Retrieval & Ranking

#### Retrieval:

##### Process
- Generate large list of plausible item candidates
  - For example, for each 10 most recent movies watched, find 10 most similar movies to it
  - For most 3 viewed genres, get top 10 movies
  - Top 20 movies in the country
- Combine all info retrieved, removing duplicates or already watched. 

##### Things to look out for

As we retrieve more movies, we will have better performance in our algorithm, but it is less efficient (slower). We can experiment to optimize the trade off between the two.  

#### Ranking

##### Process
- Take list retrieved by previous step
- Rank using the learning model
- Display ranked items to user

### Cache

If we calculate $v_m$ constantly, an optimization would be tto cache the vector and use it when needed.

