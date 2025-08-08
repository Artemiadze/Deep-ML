import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-6) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(max_iter):
        # Compute predictions
        y_pred = np.dot(X, weights) + bias
        
        # Compute gradients for weights and bias
        error = y_pred - y
        grad_weights = (1/n_samples) * np.dot(X.T, error) + alpha * np.sign(weights)
        grad_bias = (1/n_samples) * np.sum(error)
        
        # Store old weights and bias for convergence check
        old_weights = weights.copy()
        old_bias = bias
        
        # Update weights and bias
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
        
        # Check for convergence with stricter tolerance
        if np.all(np.abs(weights - old_weights) < tol) and np.abs(bias - old_bias) < tol:
            break
    
    return weights, bias

X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])

alpha = 0.1
output = l1_regularization_gradient_descent(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000) 
print(output)