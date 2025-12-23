import numpy as np

def lagrange_interpolate(xs, ys, xq):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    n = len(xs)
    steps = []

    total = 0.0
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if i != j:
                Li *= (xq - xs[j]) / (xs[i] - xs[j])
        contrib = ys[i] * Li
        total += contrib
        steps.append({
            "i": i, "x_i": xs[i], "y_i": ys[i], "L_i(xq)": Li, "aporte": contrib
        })

    return total, steps
