import numpy as np

def newton_divided_differences(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    n = len(xs)

    table = np.zeros((n, n), dtype=float)
    table[:, 0] = ys

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (xs[i+j] - xs[i])

    coef = table[0, :].copy()
    return coef, table

def newton_eval(xs, coef, xq):
    xs = np.array(xs, dtype=float)
    coef = np.array(coef, dtype=float)
    n = len(xs)

    result = coef[0]
    prod = 1.0
    for k in range(1, n):
        prod *= (xq - xs[k-1])
        result += coef[k] * prod
    return float(result)
