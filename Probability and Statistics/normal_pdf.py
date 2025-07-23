import math

def normal_pdf(x, mean, std_dev):
    val = (1 / (std_dev * math.sqrt(2 * math.pi))) * \
          math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return round(val, 5)

print(normal_pdf(0, 0, 1))
print(normal_pdf(16, 15, 2.04))
print(normal_pdf(1, 0, 0.5))