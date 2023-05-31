> Unsupervised Learning: Dataset just has inputs ($\vec{x}$). This means, we don't have labels for the data. There is no *right answer*. Instead it tries to find some structure on the data. 

# Clustering

A clustering algorithm looks at data points, and finds points that are related or similar to each other. It tries to group data into *clusters*. One of the most common clustering algorithms is the *K-means* algorithm.

![Clustering Img](https://miro.medium.com/v2/resize:fit:561/0*ff7kw5DRQbs_uixR.jpg)

## Applications
- Grouping similar news
- Market segmentation
- DNA Data Analysis

## K-Means

The k-means algorithm mainly looks for centers of $K$ clusters (*cluster centroids*), starting them randomly, and then moving them while metrics are improving.

### Algorithm

- Randomly initialize $K$ cluster centroids
- Repeat:
  - For each point, it assigns each point to the closest cluster centroid
  - Recomputes each cluster centroid, moving it to the average of the data points
    - If a cluster does not have any associated points, we delete it
- Keep repeating the process until convergence (no more changes are occurring in the process).

### Optimization Objective

Given:
- $c^{(i)}$: index of cluster to which example $x^{(i)}$ is assigned to.
- $\mu_k$: cluster centroid of cluster $k$ 
- $\mu_{c^{(i)}}$: cluster centroid to which example $x^{(i)}$ has been assigned

Then we say the cost function of K-means is:

> $J(\vec{c}, \vec{\mu}) = \frac{1}{m} \sum_{i=1}^{m}{||x^{(i)} - \mu_{c^{(i)}}||^2}$ 

Or in words, the average squared distance between every training example, and the location of the cluster centroid assigned to it. It is called the *distortion function*. 

Our algorithm tries to minimize it in each step, and that is why it is guaranteed to converge. It can never go up.

### Initializing K-Means

We need to choose $K < m$

A way of initializing cluster centroids  is to pick $K$ random training examples and initialize the centroids on those points.

However, depending on the initial choice, we can end with different results (on local minima). Then, we need a way to avoid or try to mitigate this.

An option to do this is tu run the algorithm multiple times (we get different results due to initialization) and keep the one with the lowest cost at the end. This can be done from $50-1000$ times on average. 

### Choosing Number of Clusters

There is no *right answer* and it can be ambiguous. However, there are some techniques to automatically choose the number of clusters. Below, there are a couple of them.

We don't always choose the biggest $K$, as it always mostly reduces the cost to add more clusters, but it is more expensive computationally.

#### Elbow method

We run the algorithm with different number of clusters, and evaluate the cost of each one. If we graph the costs as $K$ increases, we can choose the $K$ where the function stops decreasing rapidly. It looks like an elbow on the graph, hence the name.

However, sometimes it is not really clear where the elbow is. There are much better options.

#### Tradeoff with final purpose

Often, you want to get clusters for some downstream purpose. It is recommended to evaluate K-means based on how well it performs on that later purpose. This is a tradeoff, depending on what you want, but you can manually decide what seems best.