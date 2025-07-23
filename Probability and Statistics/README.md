# 2. Probability and Statistics

# 2.1 Poisson Distribution Probability Calculator
Write a Python function to calculate the probability of observing exactly k events in a fixed interval using the Poisson distribution formula. The function should take k (number of events) and lam (mean rate of occurrences) as inputs and return the probability rounded to 5 decimal places.

**Input**:
```python
k = 3, lam = 5
```

**Output**:
```python
0.14037
```

**Reasoning:**
```The function calculates the probability for a given number of events occurring in a fixed interval, based on the mean rate of occurrences.``` 

# 2.2 Binomial Distribution Probability
Write a Python function to calculate the probability of achieving exactly k successes in n independent Bernoulli trials, each with probability p of success, using the Binomial distribution formula.

**Input**:
```python
n = 6, k = 2, p = 0.5
```

**Output**:
```python
0.23438
```

**Reasoning:**
```The function calculates the Binomial probability, the intermediate steps include calculating the binomial coefficient, raising p and (1-p) to the appropriate powers, and multiplying the results.``` 

# 2.3 Normal Distribution PDF Calculator
Write a Python function to calculate the probability density function (PDF) of the normal distribution for a given value, mean, and standard deviation. The function should use the mathematical formula of the normal distribution to return the PDF value rounded to 5 decimal places. 

**Input**:
```python
x = 16, mean = 15, std_dev = 2.04
```

**Output**:
```python
0.17342
```

**Reasoning:**
```The function computes the PDF using x = 16, mean = 15, and std_dev = 2.04.``` 

# 2.4 Descriptive Statistics Calculator
Write a Python function to calculate various descriptive statistics metrics for a given dataset. The function should take a list or NumPy array of numerical values and return a dictionary containing mean, median, mode, variance, standard deviation, percentiles (25th, 50th, 75th), and interquartile range (IQR).

**Input**:
```python
[10, 20, 30, 40, 50]
```

**Output**:
```python
{'mean': 30.0, 'median': 30.0, 'mode': 10, 'variance': 200.0, 'standard_deviation': 14.142135623730951, '25th_percentile': 20.0, '50th_percentile': 30.0, '75th_percentile': 40.0, 'interquartile_range': 20.0}
```

**Reasoning:**
```The dataset is processed to calculate all descriptive statistics. The mean is the average value, the median is the central value, the mode is the most frequent value, and variance and standard deviation measure the spread of data. Percentiles and IQR describe data distribution.``` 

# 2.5 Calculate Covariance Matrix
Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists. Additionally, provide test cases to verify the correctness of your implementation.

**Input**:
```python
[[1, 2, 3], [4, 5, 6]]
```

**Output**:
```python
[[1.0, 1.0], [1.0, 1.0]]
```

**Reasoning:**
```The covariance between the two features is calculated based on their deviations from the mean. For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix.``` 