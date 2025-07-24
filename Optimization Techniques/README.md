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