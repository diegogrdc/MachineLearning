# Anomaly Detection

> Anomaly Detection: Unsupervised Learning algorithm that, given a dataset of normal events, it learns to detect if there is an anomalous event.

A technique used for anomaly detection is density estimation, which uses gaussian distribution.

![Anomaly Detection Img](https://www.researchgate.net/profile/Mustafa-Aljumaily-2/publication/321682378/figure/fig1/AS:569320483033088@1512747988945/Figure-1-anomaly-detection.png)

## Uses
- Fraud Detection
- Manufacturing checks
- Monitoring computers in data centers

## Gaussian Distribution (Normal)

To apply anomaly detection, we need to use the gaussian (normal) distribution.

![Gauss img](https://i.stack.imgur.com/oXbxu.jpg)

On a normal distribution, the probability of $x$ is determined by a Gaussian with a mean $\mu$, variance $\sigma^2$. The formula is:

> $p(x) = \frac{1}{\sqrt{2\pi}\sigma}*e^{\frac{-(x-\mu)^2}{2\sigma^2}}$

### Parameter estimation

To get the gaussian distribution for our data we have the following formulas:

> $\mu = \frac{1}{m}\sum_{i=1}^{m}{x^{(i)}}$

> $\sigma^2= \frac{1}{m}\sum_{i=1}^{m}({x^{(i)}-\mu)^2}$

## Density Estimation

Given a dataset, we build a model $p(x)$, which represents the probability of $x$ being seen in the dataset. When we get a new example, we compute its probability, and if $p(x_{test}) < \epsilon$ (where $\epsilon$ is our threshold), we call that example an anomaly.

## Algorithm

We are given a training set $\vec{x}$ of size $m$, and each example having $n$ features. We need to get a probability model $p(x)$. To model this, we do the following:

> $p(\vec{x}) = \prod_{j=1}^{n} {p(x_j; \mu_j, \sigma^2_j)}$

Having this, we can define the algorithm:

1. Choose $n$ features $x_i$ that you think might be indicative or anomalous examples.
2. Fit parameters $\mu$ and $\sigma^2$ for each of the $n$ features (can be vectorized).
3. Given a new example $x$, compute $p(x)$, with previous formulas. 
4. Classify as anomaly if $p(x) < \epsilon$ 

## Evaluating our algorithm

> Real number evaluation: Way of evaluation our learning algorithm, to know if our changes are making it better or worse.

To evaluate our algorithm, we need a small amount of labeled data (with anomalous examples). Having this, we can create a cross validation set, and if we have enough data, a test set too. Note that the training set remains unlabeled. Having this, we can evaluate our cross validation set and tune our parameters in a fair way. Also keep in mind that anomaly data sets are usually skewed, as there are much less anomalies than normal data points. To keep that in mind, we use metrics seen before, like precision, recall or F1-Score.

## Comparison with Supervised Learning

If we propose the use of labeled data, then why not use supervised learning? Let's compare the two. 

Anomaly Detection would be good when:
- Very small number of positive examples (0-20)
- Very large number of negative examples
- When there are many different *types* of anomalies.
  - When there are hard to learn a pattern on them or when really different anomalies might come be seen in the future.
- Examples like
  - Fraud
  - Manufacturing: Find unseen/new defects
  - Monitor data centers

Supervised Learning would be good when:
- Large number of positive and negative examples
- Similar anomalies
  - Supervised models try to fit the examples on positive examples
- Examples like
  - Spam
  - Manufacturing: Finding previous/common defects
  - Weather prediction

## Choosing Features

It is harder for unsupervised learning to learn which features to use or ignore, so it is an important part we have to take care of.

### Non-Gaussian Features

We can plot our data (`plt.hist(x)`). If it does not seem to be gaussian (normally distributed), we can transform it to make it more gaussian. This can be done with trial and error with logarithms, square roots or exponentiation.

### Error analysis

Try to look at where the algorithm is not doing well. Try to identify features that can make those errors more clear for the algorithm. For example, if an anomaly seems to be really similar to other data, find features that can make it easier to note that it is an anomaly.

### New Features

We can create new features by combining existing features. This sometimes gives us new insights on the data.