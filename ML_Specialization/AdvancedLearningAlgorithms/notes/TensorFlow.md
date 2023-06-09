# Tensorflow

> Tensorflow: One of the leading frameworks to implement Deep Learning algorithms

## Data in TensorFlow

TensorFlow is a bit different than NumPy, so some inconsistencies can happen.

- Feature vector
  - `[[200.0, 17.0]]`
  - Tensorflow has the convention to use matrices instead of vectors, hence the double brackets.
  - This internally improves Tensorflow efficiency
- Internally represented as a $Tensor$
  - Type is `tf.Tensor`
  - Can cast to np using `.numpy()`

## Neural Network Example Tensorflow

```
# Set data
X = np.array([inputs])
Y = np.array(outputs)

# Create model with defined layers
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=1, activation='sigmoid')])


# Compile model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # adam optimizer
  loss=BinaryCrossentropy()) # loss function

# Train model (Back propagation) 
model.fit(X, Y, epochs=100)

# Forward prop
a_new = model.predict(x_new)
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

### Regularization

To apply regularization on Tensorflow:

```
Dense(units=25, activation='relu', kernel_regularizer=L2(lambda))
```

### Loss functions

- Binary Crossentropy
  - Logistic loss
  - Binary classification problems
- Mean Squared Error
  - Regression