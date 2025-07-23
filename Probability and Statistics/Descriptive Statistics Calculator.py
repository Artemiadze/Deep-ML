import numpy as np 

def descriptive_statistics(data):
    # Your code here
    data = np.array(data)
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]

    stats_dict = {
        "mean": np.mean(data),
        "median": np.median(data),
        "mode": mode,
        "variance": np.round(np.var(data),4),
        "standard_deviation": np.round(np.std(data),4),
        "25th_percentile": np.percentile(data, 25),
        "50th_percentile": np.percentile(data, 50),
        "75th_percentile": np.percentile(data, 75),
        "interquartile_range": np.percentile(data, 75) - np.percentile(data, 25)
    }
    return stats_dict


print('descriptive_statistics([10, 20, 30, 40, 50])')
print(descriptive_statistics([10, 20, 30, 40, 50]))

print('descriptive_statistics([1, 2, 2, 3, 4, 4, 4, 5])')
print(descriptive_statistics([1, 2, 2, 3, 4, 4, 4, 5]))