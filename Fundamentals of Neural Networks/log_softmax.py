import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores, dtype=float)
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    softmax_vals = exp_scores / np.sum(exp_scores)
    return np.round(np.log(softmax_vals), 4)

print(log_softmax([1, 2, 3]))
print(log_softmax([1, 1, 1]))
print(log_softmax([1, 1, .0000001]))