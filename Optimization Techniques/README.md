# 3. Optimization Techniques

# 3.1 Linear Regression Using Gradient Descent
Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

**Input**:
```python
X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
```

**Output**:
```python
np.array([0.1107, 0.9513])
```

**Reasoning:**
```The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.``` 

# 3.2 Implement Gradient Descent Variants with MSE Loss
In this problem, you need to implement a single function that can perform three variants of gradient descent Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini Batch Gradient Descent using Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use. Note: Do not shuffle the data.

**Input**:
```python
import numpy as np

# Sample data
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

# Parameters
learning_rate = 0.01
n_iterations = 1000
batch_size = 2

# Initialize weights
weights = np.zeros(X.shape[1])

# Test Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
# Test Stochastic Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
# Test Mini-Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch')
```

**Output**:
```python
[float,float]
[float, float]
[float, float]
```

**Reasoning:**
```The function should return the final weights after performing the specified variant of gradient descent.``` 

# 3.3 Implement Adam Optimization Algorithm

Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function adam_optimizer that updates the parameters of a given function using the Adam algorithm.

The function should take the following parameters:

- f: The objective function to be optimized
- grad: A function that computes the gradient of f
- x0: Initial parameter values
- learning_rate: The step size (default: 0.001)
- beta1: Exponential decay rate for the first moment estimates (default: 0.9)
- beta2: Exponential decay rate for the second moment estimates (default: 0.999)
- epsilon: A small constant for numerical stability (default: 1e-8)
- num_iterations: Number of iterations to run the optimizer (default: 1000)
- The function should return the optimized parameters.

**Input**:
```python
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)
```

**Output**:
```python
# Optimized parameters: [0.99000325 0.99000325]
```

**Reasoning:**
```The Adam optimizer updates the parameters to minimize the objective function. In this case, the objective function is the sum of squares of the parameters, and the optimizer finds the optimal values for the parameters.``` 

# 3.4 Implement Lasso Regression using Gradient Descent
In this problem, you need to implement the Lasso Regression algorithm using Gradient Descent. Lasso Regression (L1 Regularization) adds a penalty equal to the absolute value of the coefficients to the loss function. Your task is to update the weights and bias iteratively using the gradient of the loss function and the L1 penalty.

**Input**:
```python
import numpy as np

X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])

alpha = 0.1
weights, bias = l1_regularization_gradient_descent(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)
```

**Output**:
```python
(weights,bias)
(array([float, float]), float)
```

**Reasoning:**
```The Lasso Regression algorithm is used to optimize the weights and bias for the given data. The weights are adjusted to minimize the loss function with the L1 penalty.``` 