from typing import List

def calculate_covariance_matrix(vectors: List[List[float]], ddof: int = 1) -> List[List[float]]:
    """
    Calculate the covariance matrix for a set of features.

    Parameters
    ----------
    vectors : list of list of float
        Each inner list is one feature (variable) containing observations in order.
        Shape: (n_features, n_observations).
    ddof : int, default=1
        Delta degrees of freedom. Use 1 for sample covariance (unbiased),
        0 for population covariance.

    Returns
    -------
    cov_matrix : list of list of float
        Symmetric covariance matrix (n_features x n_features).

    Raises
    ------
    ValueError
        If input is empty, lengths differ, or n_observations - ddof <= 0.
    """
    if not vectors:
        raise ValueError("Input 'vectors' is empty.")

    n_features = len(vectors)
    n_obs = len(vectors[0])
    
    for v in vectors:
        if len(v) != n_obs:
            raise ValueError("All feature vectors must have the same number of observations.")

    if n_obs - ddof <= 0:
        raise ValueError(f"Not enough observations ({n_obs}) for ddof={ddof}.")

    means = [sum(v) / n_obs for v in vectors]

    cov_matrix = [[0.0] * n_features for _ in range(n_features)]

    for i in range(n_features):
        for j in range(i, n_features):
            cov_ij = 0.0
            vi = vectors[i]
            vj = vectors[j]
            mi = means[i]
            mj = means[j]
            for k in range(n_obs):
                cov_ij += (vi[k] - mi) * (vj[k] - mj)
            cov_ij /= (n_obs - ddof)
            cov_matrix[i][j] = cov_ij
            cov_matrix[j][i] = cov_ij  # symmetry

    return cov_matrix

print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))
print(calculate_covariance_matrix([[1, 5, 6], [2, 3, 4], [7, 8, 9]]))