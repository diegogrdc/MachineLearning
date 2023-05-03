# Intro to Machine Learning

> Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.

## ML Algorithms

Different types of algorithms classified in two main categories: supervised and unsupervised.

### Supervised Learning 

- Algorithm gets data in an (X,Y) pair form, having an input and an output which represents the correct answer. 

- It trains and learns from the "right" answers.

- After model learns from this data, it can take a not seen before input X, and try to predict the appropriate corresponding Y.

| Input(X) | Output(Y) | Application |
| --- | --- | --- |
| email | spam(0/1) | Spam filtering |
| audio | text transcript | speech recognition |
| Ad, user Info | Click? (0/1) | online Ads |
| Image, radar info | Position of other cars | Self Driving Car |

The most popular supervised learning algorithms are:

- Regression
  - Predict a number from infinitely many possible possible outputs.
  - Example: Price of house given size of house.
- Classification
  - Predict categories.
  - Example: Given a picture, guess if it is a dog or a cat.


### Unsupervised Learning

- Data that is not associated with labels, only input X, but not output Y.
- Algorithm looks to find patterns or structures data might have.

Some popular unsupervised learning algorithms are:
- Clustering
  - Tries to group similar data into clusters.
  - Example: If reading a news article, Google recommends similar news. 
- Anomaly Detection
  - Find unusual data points
- Dimensionality reduction
  - Compress data using fewer numbers

> Jupyter Notebooks: Helpful environment to code up and experiment with Machine Learning