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