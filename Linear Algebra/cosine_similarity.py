import numpy as np
from math import sqrt

def cosine_similarity(v1, v2):
    sum_v1_v2 = sum(v1[i]*v2[i] for i in range(len(v1)))
    q_sum_v1 = sum(v1[i]*v1[i] for i in range(len(v1)))
    q_sum_v2 = sum(v2[i]*v2[i] for i in range(len(v2)))
    return round(sum_v1_v2/(sqrt(q_sum_v1) * sqrt(q_sum_v2)), 3)

print("Test 1")
v1 = np.array([1, 2, 3]) 
v2 = np.array([2, 4, 6]) 
print(cosine_similarity(v1, v2))

print("Test 2")
v1 = np.array([1, 2, 3])
v2 = np.array([-1, -2, -3])
print(cosine_similarity(v1, v2))

print("Test 3")
v1 = np.array([1, 0, 7]) 
v2 = np.array([0, 1, 3]) 
print(cosine_similarity(v1, v2))
