def bisection(f, a, b, tol=1e-6, nmax=50):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Bisección requiere f(a)*f(b) < 0")

    steps = []
    c = None
    for k in range(1, nmax + 1):
        c = (a + b) / 2.0
        fc = f(c)
        err = abs(b - a) / 2.0

        steps.append({
            "k": k, "a": a, "b": b, "c": c,
            "f(a)": fa, "f(b)": fb, "f(c)": fc,
            "error_est": err
        })

        if abs(fc) < tol or err < tol:
            return c, steps

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, steps
