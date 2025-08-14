# 4. Fundamentals of Neural Networks

## 4.1 Single Neuron
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.

**Input**:
```python
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
```

**Output**:
```python
([0.4626, 0.4134, 0.6682], 0.3349)
```

## 4.2 Implement ReLU Activation Function
Write a Python function `relu` that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.

**Input**:
```python
print(relu(0)) 
print(relu(1)) 
print(relu(-1))
```

**Output**:
```python
0
1
0
```

## 4.3 Leaky ReLU Activation Function
Write a Python function leaky_relu that implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function. The function should take a float z as input and an optional float alpha, with a default value of 0.01, as the slope for negative inputs. The function should return the value after applying the Leaky ReLU function.

```python
print(leaky_relu(0)) 
print(leaky_relu(1))
print(leaky_relu(-1)) 
print(leaky_relu(-2, alpha=0.1))
```

**Output**:
```python
0
1
-0.01
-0.2
```

## 4.4 Implement the PReLU Activation Function
Implement the PReLU (Parametric ReLU) activation function, a variant of the ReLU activation function that introduces a learnable parameter for negative inputs. Your task is to compute the PReLU activation value for a given input.

```python
prelu(-2.0, alpha=0.25)
```

**Output**:
```python
-0.5
```

## 4.5 Sigmoid Activation Function Understanding
Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.

```python
z = 0
```

**Output**:
```python
0.5
```

## 4.6 Softmax Activation Function Implementation
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

```python
scores = [1, 2, 3]
```

**Output**:
```python
[0.0900, 0.2447, 0.6652]
```

# 4.7 Implementation of Log Softmax Function
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.

```python
A = np.array([1, 2, 3])
print(log_softmax(A))
```

**Output**:
```python
array([-2.4076, -1.4076, -0.4076])
```