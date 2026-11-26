import numpy as np


def simulate_wealth(shares, values, N=10_000, seed=42, within_variation=True):
    """Simulate an economy of N adults and assign each adult a net wealth.

    Args:
        shares (array): age-group shares (probabilities), one per age group, summing to one
        values (2D array): wealth percentiles, age groups in rows and P1-P99 in columns
        N (int): number of simulated adults
        seed (int): seed for the random number generator
        within_variation (bool): if False, everybody gets the median (P50) of their age group;
                                 if True, everybody gets the percentile of their drawn rank

    Returns:
        age (array): age-group index (0, 1, ...) for each adult
        rank (array): drawn rank from 1 to 99 for each adult
        wealth (array): simulated net wealth for each adult, DKK
    """

    # a. check inputs
    shares = np.asarray(shares, dtype=float)
    assert np.isclose(shares.sum(), 1.0), 'age shares must sum to one'
    assert values.shape == (shares.size, 99), 'values must have one row per age group and 99 columns'

    rng = np.random.default_rng(seed)

    # b. draw an age group for each adult
    u = rng.uniform(0, 1, size=N)
    age = np.zeros(N, dtype=int)

    lower = 0.0
    for j, s in enumerate(shares):
        upper = lower + s
        age[(u >= lower) & (u < upper)] = j
        lower = upper

    # c. draw a rank from 1 to 99 for each adult (after the ages)
    rank = rng.integers(1, 100, size=N)

    # d. assign wealth
    wealth = np.zeros(N)
    for i in range(N):
        if within_variation:
            wealth[i] = values[age[i], rank[i] - 1]    # P1 is column 0
        else:
            wealth[i] = values[age[i], 49]             # P50 is column 49

    return age, rank, wealth