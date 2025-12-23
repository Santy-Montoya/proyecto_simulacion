import numpy as np

def poisson_1d_dirichlet(f, a, b, ua, ub, n_interior):
    """
    Resuelve -u''(x) = f(x), u(a)=ua, u(b)=ub
    Con diferencias finitas de 2do orden.
    n_interior = cantidad de nodos interiores.
    """
    a = float(a); b = float(b)
    n = int(n_interior)

    h = (b - a) / (n + 1)
    xs = np.array([a + i*h for i in range(n + 2)], dtype=float)

    # Construir A (n x n)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 2.0
        if i - 1 >= 0:
            A[i, i-1] = -1.0
        if i + 1 < n:
            A[i, i+1] = -1.0

    # rhs
    rhs = np.zeros(n, dtype=float)
    for i in range(n):
        xi = xs[i+1]
        rhs[i] = (h**2) * f(xi)

    # Ajuste por condiciones de frontera
    rhs[0] += ua
    rhs[-1] += ub

    # Resolver
    u_interior = np.linalg.solve(A, rhs)

    us = np.zeros(n + 2, dtype=float)
    us[0] = ua
    us[-1] = ub
    us[1:-1] = u_interior

    return xs, us, A, rhs
