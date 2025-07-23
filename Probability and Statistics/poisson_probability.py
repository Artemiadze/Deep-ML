import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	# Your code here
	val = (math.pow(lam, k) / math.factorial(k)) * math.exp(-lam) 
	return round(val,5)

print(poisson_probability(3, 5))
print(poisson_probability(0, 5))
print(poisson_probability(2, 10))
print(poisson_probability(1, 1))
print(poisson_probability(20, 20))