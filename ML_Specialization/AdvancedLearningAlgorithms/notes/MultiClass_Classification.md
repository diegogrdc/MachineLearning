# Multiclass Classification (Softmax)

> Multiclass Classification: Classification problem where we have a small number of possible discrete outputs, but more than two.

This an be achieved with an algorithm called softmax, and used as an activation function within neural networks for better performance. 

![Multiclass classification diagram](https://www.oreilly.com/api/v2/epubs/9781838823733/files/assets/8cb556f8-8f3b-434c-86bd-b142454f82c6.png)

## Formula

The softmax regression algorithm is a generalization of logistic regression.

For each class:

$z_i = \vec{w}_i \cdot \vec{x} + b_i$

$a_i = \frac{e^{z_i}}{\sum_{k=1}^{N}{e^{z_k}}} = P(y = i | \vec{x})$ 

Where $N$ is the number of classes

## Cost

$L(a_1, ..., a_N, y) = - log(a_j)$ if $y=j$

## Neural Network

Softmax output layer, with $N$ units.

```
...
Dense(units=N, activation='softmax')
...
model.compile(loss=SparseCategoricalCrossentropy())
...
model.fit(X, Y, epochs=100)
```


### More accuracy

Some calculations in tensorflow can be improved to be more precise by:
```
...
Dense(units=N, activation='linear')
...
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
...
model.fit(X, Y, epochs=100)
...
logits = model(X)
f_x = tf.nn.softmax(logits)
```

## Multi-label classification 

Associated with a single output, there can be multiple output classes.

It can be approached separately, with $N$ different Neural Networks, but it can also be done in a single Neural Network.

To do this, the output layer can have $N$ nodes with sigmoid activations representing the binary probability of each label.