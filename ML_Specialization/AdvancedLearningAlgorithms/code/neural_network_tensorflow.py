# Set data
X = np.array([inputs])
Y = np.array(outputs)

# Create model with defined layers
model = Sequential([
    Dense(units=25, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(units=15, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(units=1, activation='sigmoid')])


# Compile model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # adam optimizer
  loss=BinaryCrossentropy()) # loss function

# Train model (Back propagation) 
model.fit(X, Y, epochs=100)

# Predict
a_new = model.predict(x_new)