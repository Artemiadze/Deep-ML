import math

def single_neuron_model(features: list[list[float]], labels: list[int],
                        weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    for x in features:
        # Сумма w*x + b
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        # Сигмоида
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))

    # MSE
    mse = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)

    return probabilities, mse

print(single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))
print(single_neuron_model([[1, 2], [2, 3], [3, 1]], [1, 0, 1], [0.5, -0.2], 0))