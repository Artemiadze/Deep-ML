def prelu(x: float, alpha: float = 0.25) -> float:
	"""
	Implements the PReLU (Parametric ReLU) activation function.

	Args:
		x: Input value
		alpha: Slope parameter for negative values (default: 0.25)

	Returns:
		float: PReLU activation value
	"""
	# Your code here
	if x >= 0:
		return x
	return x * alpha

print(prelu(2.0))
print(prelu(0.0))
print(prelu(-2.0))
print(prelu(-2.0, alpha=0.1))
print(prelu(-2.0, alpha=1.0))