import numpy as np

def newton_system_2d(F, G, x0, y0, tol=1e-10, nmax=30, h=1e-6):
    def jacobian(x, y):
        # Derivadas parciales numéricas (centrales)
        Fx = (F(x + h, y) - F(x - h, y)) / (2*h)
        Fy = (F(x, y + h) - F(x, y - h)) / (2*h)
        Gx = (G(x + h, y) - G(x - h, y)) / (2*h)
        Gy = (G(x, y + h) - G(x, y - h)) / (2*h)
        return np.array([[Fx, Fy],
                         [Gx, Gy]], dtype=float)

    x, y = float(x0), float(y0)
    steps = []
    for k in range(1, nmax + 1):
        J = jacobian(x, y)
        b = -np.array([F(x, y), G(x, y)], dtype=float)

        if abs(np.linalg.det(J)) < 1e-14:
            raise ValueError("Jacobiano casi singular: no se puede invertir.")

        delta = np.linalg.solve(J, b)
        x_new = x + delta[0]
        y_new = y + delta[1]
        err = float(np.linalg.norm(delta, ord=np.inf))

        steps.append({
            "k": k,
            "x": x, "y": y,
            "F": -b[0], "G": -b[1],
            "dx": delta[0], "dy": delta[1],
            "error_inf": err
        })

        x, y = x_new, y_new
        if err < tol:
            return (x, y), steps

    return (x, y), steps
