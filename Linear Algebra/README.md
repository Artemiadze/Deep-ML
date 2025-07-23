# 1. Linear Algebra

for launching code use this type of command
```bash
python file_name.py
```

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

# 1.4 Scalar Multiplication of a Matrix
Write a Python function that multiplies a matrix by a scalar and returns the result.

**Input:**
```python
matrix = [[1, 2], [3, 4]], scalar = 2
```

**Output:**
```python
[[2, 4], [6, 8]]
```

**Reasoning:**
```Each element of the matrix is multiplied by the scalar.```

# 1.5 Calculate Cosine Similarity Between Vectors
**Task: Implement Cosine Similarity**
In this task, you need to implement a function cosine_similarity(v1, v2) that calculates the cosine similarity between two vectors. Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity.

**Input:**
- v1 and v2: Numpy arrays representing the input vectors.

**Output:**
- A float representing the cosine similarity, rounded to three decimal places.

**Constraints:**
- Both input vectors must have the same shape.
- Input vectors cannot be empty or have zero magnitude.

**Example**\
**Input:**
```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(cosine_similarity(v1, v2))
```

**Output:**
```python
1.0
```


# 1.6 Calculate Mean by Row or Column
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

**Input**:
```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
```

**Output**:
```python
[4.0, 5.0, 6.0]
```