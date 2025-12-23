import numpy as np
from src.common import norm_inf

def jacobi(A, b, x0=None, tol=1e-10, nmax=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)

    D = np.diag(A)
    if np.any(np.abs(D) < 1e-15):
        raise ValueError("Jacobi requiere diagonal no nula.")

    R = A - np.diagflat(D)

    steps = []
    for k in range(1, nmax + 1):
        x_new = (b - R @ x) / D
        err = norm_inf(x_new - x)
        steps.append({"k": k, "x": x.copy(), "x_new": x_new.copy(), "error_inf": err})
        x = x_new
        if err < tol:
            return x, steps

    return x, steps
