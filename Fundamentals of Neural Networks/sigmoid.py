import math

def sigmoid(z: float) -> float:
	#Your code here
	return round(1/ (1 + math.exp(-z)), 4)

print(sigmoid(0))
print(sigmoid(1))
print(sigmoid(-1))