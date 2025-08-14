import math

def softmax(scores: list[float]) -> list[float]:
    # Вычитаем max для численной стабильности
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    probabilities = [round(e / total, 4) for e in exp_scores]
    return probabilities

print(softmax([1, 2, 3]))
print(softmax([1, 1, 1]))
print(softmax([-1, 0, 5]))