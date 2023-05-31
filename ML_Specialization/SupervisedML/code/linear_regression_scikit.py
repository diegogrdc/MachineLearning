# Imports 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor # Gradient Descent Regressor Model
from sklearn.preprocessing import StandardScaler # Feature scaling 
np.set_printoptions(precision=2)

# Generate dataset
X_train = np.array([[np.random.randint(300, 2501),
                    np.random.randint(1, 6),
                    np.random.randint(1, 3),
                    np.random.randint(0, 80),
                    ] for i in range(100)])

w_real = np.array([5, 2, 1, -.5])
b_real = 100

y_train = np.dot(X_train, w_real) + b_real
noise = np.random.normal(0,1,100) * 1000
y_train += noise
X_features = ['size(sqft)','bedrooms','floors','age']

# Scale (Normalize) data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)


# Create/Train Model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}")

# View Params
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")

# Predict 
y_pred = sgdr.predict(X_norm)

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# Plot results
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color="orange", label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
