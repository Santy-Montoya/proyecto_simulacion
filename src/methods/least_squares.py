import numpy as np

def poly_least_squares_fit(xs, ys, degree):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    m = int(degree)

    # Matriz de Vandermonde: [1, x, x^2, ..., x^m]
    V = np.vander(xs, N=m+1, increasing=True)

    # Ecuaciones normales: (V^T V) c = V^T y
    A = V.T @ V
    b = V.T @ ys
    coef = np.linalg.solve(A, b)

    yhat = V @ coef
    resid = ys - yhat
    sse = float(np.sum(resid**2))

    steps = []
    steps.append({"A_shape": A.shape, "b_shape": b.shape, "SSE": sse})

    info = {"SSE": sse, "steps": steps}
    return coef, info

def poly_eval(coef, x):
    coef = np.array(coef, dtype=float)
    x = float(x)
    powers = np.array([x**k for k in range(len(coef))], dtype=float)
    return float(np.dot(coef, powers))
