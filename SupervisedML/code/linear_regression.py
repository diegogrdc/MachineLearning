# Imports
import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2) 

class LinearRegression:
    def __init__(self, X_train, y_train, w_init, b_init):
        # m = # of examples
        # n = number of features

        # shape(m, n) 
        self.X_train = X_train
        # shape(m,)
        self.y_train = y_train
        # shape(n,)
        self.w = w_init
        # shape(,)
        self.b = b_init

    # Eval given input given model parameters
    def predict(self, x, w, b): 
        """
        Single predict using linear regression
        
        Args:
        x (ndarray): Shape (n,) example with multiple features
        w (ndarray): Shape (n,) model parameters    
        b (scalar):  model parameter     
        
        Returns:
        p (scalar):  prediction
        """
        p = np.dot(x, w) + b
        return p

    # Compute cost of all data
    def compute_cost(self, X, y, w, b): 
        """
        compute cost
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        
        Returns:
        cost (scalar): cost
        """
        m = X.shape[0]
        cost = 0.0
        for i in range(m):
            yhat = self.predict(X[i], w, b)
            cost += (yhat - y[i]) ** 2
        cost = cost / (2 * m)
        return cost
    
    def compute_gradient(self, X, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        
        Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
        """
        m, n = X.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.0
        
        for i in range(m):
            err = self.predict(X[i], w, b) - y[i]
            for j in range(n):
                dj_dw[j] += err * X[i, j]
            dj_db += err

        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db
    
    def gradient_descent(self, X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
        """
        Performs batch gradient descent to learn w and b. Updates w and b by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
        X (ndarray (m,n))   : Data, m examples with n features
        y (ndarray (m,))    : target values
        w_in (ndarray (n,)) : initial model parameters  
        b_in (scalar)       : initial model parameter
        cost_function       : function to compute cost
        gradient_function   : function to compute the gradient
        alpha (float)       : Learning rate
        num_iters (int)     : number of iterations to run gradient descent
        
        Returns:
        w (ndarray (n,)) : Updated values of parameters 
        b (scalar)       : Updated value of parameter 
        """
    
        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w = copy.deepcopy(w_in)  #avoid modifying global w within function
        b = b_in
        
        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_dw,dj_db = gradient_function(X, y, w, b)

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
        
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion 
                J_history.append(cost_function(X, y, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
            
        return w, b, J_history #return final w,b and J history for graphing


    
if __name__ == "__main__":
    # Set up initial data
    X_train = np.array([
        [2104, 5, 1, 45], 
        [1416, 3, 2, 40], 
        [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    
    # initialize parameters
    initial_w = np.zeros((4,))
    initial_b = 0.

    # some gradient descent settings
    iterations = 100000
    alpha = 5.0e-7

    model = LinearRegression(X_train, y_train, initial_w, initial_b)

    # run gradient descent 
    w_final, b_final, J_hist = model.gradient_descent(X_train, y_train, initial_w, initial_b,
                                                        model.compute_cost, model.compute_gradient, 
                                                        alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    plt.show()

