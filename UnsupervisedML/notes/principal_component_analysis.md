# Principal Component Analysis

Algorithm commonly used for visualization. Specifically, if we have a dataset with lots of features, we can reduce the features to 2-3, so that we can plot it and analyze it. Mainly used for data visualization by data scientists.

![PCA Example](https://programmathically.com/wp-content/uploads/2021/08/pca_figure1-1024x1024.jpeg)

## Algorithm

- Preprocess features
  - Normalized to have zero mean
  - Feature scaling
- Choose an axis
  - *Principal component*: Maximum variance
    - Additional components would be 90 degree from principal component (perpendicular)
- Project examples onto the axis
  - The new coordinate will be the magnitude of the dot product of the length 1 vector of the chosen axis and the vector starting from the origin to the chosen point

## Applications
- Data visualization
  - Reduce to 2 - 3 dimensions

Less frequently used for (due to technology advances):
- Data compression
  - To reduce storage or transmission costs
- Speed up training of a supervised training model


## Scikit learn Code

- Optional preprocessing: Feature scaling

1. "Fit" the data to obtain 2 (or 3) new axes (principal components)
   - `fit`
   - Includes mean normalization
2. Optionally examine how much variance (info) is explained by each principal component
   - `explained_variance_ratio`
3. Transform (project) the data onto new axes
   - `transform`

### Example
```
X = np.array([[1, 1], [2, 1], [3, 2], [-1, -1], [-2, -1], [-3, -2]])
pca_1 = PCA(n_components=1)
pca_1.fit(X)
print(pca.explained_variance_ratio()) # 0.992
X_trans_1 = pca_1.transform(X)
X_reduced_1 = pca.inverse_transform(X_trans_1)
```






