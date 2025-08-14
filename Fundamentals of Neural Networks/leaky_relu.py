def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	if z >= 0:
		return z
	return z* alpha

print(leaky_relu(0)) 
print(leaky_relu(1))
print(leaky_relu(-1)) 
print(leaky_relu(-2, alpha=0.1))