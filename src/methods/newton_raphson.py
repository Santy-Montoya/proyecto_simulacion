def newton_raphson(f, x0, tol=1e-8, nmax=30, h=1e-6):
    def deriv(x):
        return (f(x + h) - f(x - h)) / (2.0 * h)

    x = float(x0)
    steps = []
    for k in range(1, nmax + 1):
        fx = f(x)
        dfx = deriv(x)
        if abs(dfx) < 1e-15:
            raise ValueError("Derivada ~ 0, Newton-Raphson falla (divide por cero).")

        x_new = x - fx / dfx
        err = abs(x_new - x)

        steps.append({
            "k": k, "x": x, "f(x)": fx, "f'(x)": dfx,
            "x_new": x_new, "error": err
        })

        x = x_new
        if err < tol or abs(fx) < tol:
            return x, steps

    return x, steps
