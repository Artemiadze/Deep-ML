import math

def binomial_probability(n, k, p):
    """
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
    # Your code here 
    probability = math.comb(n, k) * math.pow(p, k) * math.pow(1-p, n-k)
    return round(probability, 5)

print(binomial_probability(6, 2, 0.5))
print(binomial_probability(6, 4, 0.7))
print(binomial_probability(3, 3, 0.9))
print(binomial_probability(5, 0, 0.3))
print(binomial_probability(7, 2, 0.1))