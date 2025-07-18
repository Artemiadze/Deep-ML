# 1. Linear Algebra

# 1.1 Matrix-Vector Dot Product
Write a Python function that computes the dot product of a matrix and a vector. The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. For example, an n x m matrix requires a vector of length m.

**Input:**
```python
a = [[1, 2], [2, 4]], b = [1, 2]
```

**Output:**
```python
[5, 10]
```

**Reasoning:**
```Row 1: (1 * 1) + (2 * 2) = 1 + 4 = 5; Row 2: (1 * 2) + (2 * 4) = 2 + 8 = 10```

# 1.2 Transpose of a Matrix
Write a Python function that computes the transpose of a given matrix.

**Input:**
```python
a = [[1,2,3],[4,5,6]]
```

**Output:**
```python
[[1,4],[2,5],[3,6]]
```

# 1.3 Dot Product Calculator
Write a Python function to calculate the dot product of two vectors. The function should take two 1D NumPy arrays as input and return the dot product as a single number.

**Input:**
```python
vec1 = np.array([1, 2, 3]), vec2 = np.array([4, 5, 6])
```

**Output:**
```python
32
```

**Reasoning:**
```The function calculates the dot product by multiplying corresponding elements of the two vectors and summing the results. For vec1 = [1, 2, 3] and vec2 = [4, 5, 6], the result is (1 * 4) + (2 * 5) + (3 * 6) = 32.```