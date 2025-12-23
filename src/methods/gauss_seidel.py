import numpy as np
from src.common import norm_inf

def gauss_seidel(A, b, x0=None, tol=1e-10, nmax=100, w=1.0):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)

    steps = []
    for k in range(1, nmax + 1):
        x_old = x.copy()

        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError("Gauss-Seidel requiere diagonal no nula.")

            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x_i_new = (b[i] - s1 - s2) / A[i, i]

            # Relajación (SOR)
            x[i] = w * x_i_new + (1 - w) * x_old[i]

        err = norm_inf(x - x_old)
        steps.append({"k": k, "x": x.copy(), "error_inf": err})

        if err < tol:
            return x, steps

    return x, steps
