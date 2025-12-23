import streamlit as st
import numpy as np
import pandas as pd

from src.ui_helpers import section_title, show_steps_table, parse_points_text, parse_matrix_text, parse_vector_text
from src.safe_eval import SafeEvaluator, SafeEvalError

from src.methods.bisection import bisection
from src.methods.newton_raphson import newton_raphson
from src.methods.newton_systems import newton_system_2d
from src.methods.lagrange import lagrange_interpolate
from src.methods.newton_interpolation import newton_divided_differences, newton_eval
from src.methods.jacobi import jacobi
from src.methods.gauss_seidel import gauss_seidel
from src.methods.finite_differences import poisson_1d_dirichlet
from src.methods.least_squares import poly_least_squares_fit, poly_eval


st.set_page_config(
    page_title="Simulación y Computación Numérica",
    page_icon="🧮",
    layout="wide"
)

st.markdown("""
<style>
.big-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtle {
    color: #6b7280;
    margin-top: 0rem;
}
.card {
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(120,120,120,0.2);
    background: rgba(255,255,255,0.03);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧮 Suite Interactiva - Simulación y Computación Numérica</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Selecciona un método, ingresa datos y obtén solución + paso a paso + tablas.</div>', unsafe_allow_html=True)

safe = SafeEvaluator()

with st.sidebar:
    st.header("⚙️ Panel de control")
    metodo = st.selectbox(
        "Método",
        [
            "Bisección",
            "Newton-Raphson",
            "Newton no lineales (Sistema 2D)",
            "Jacobi",
            "Gauss-Seidel",
            "Diferencias Finitas (Poisson 1D)",
            "Lagrange (Interpolación)",
            "Interpolación de Newton (Divididas)",
            "Mínimos Cuadrados (Polinomio)"
        ]
    )
    st.divider()
    st.caption("Tip: usa funciones tipo `x**3 - x - 1`, `sin(x)`, `exp(x)`.")

colA, colB = st.columns([1.15, 0.85], gap="large")

# =========================
# BISECCIÓN
# =========================
if metodo == "Bisección":
    with colA:
        section_title("🔎 Método de Bisección", "Encuentra una raíz en [a,b] verificando cambio de signo.")
        expr = st.text_input("f(x) =", value="x**3 - x - 1")
        a = st.number_input("a", value=1.0)
        b = st.number_input("b", value=2.0)
        tol = st.number_input("Tolerancia", value=1e-6, format="%.10f")
        nmax = st.number_input("Iteraciones máximas", value=50, step=1)

        if st.button("🚀 Resolver", type="primary"):
            try:
                f = lambda x: safe.eval_expr(expr, x=x)
                root, steps = bisection(f, a, b, tol=tol, nmax=int(nmax))
                st.success(f"✅ Raíz aproximada: {root:.10f}")
                show_steps_table(steps)

            except SafeEvalError as e:
                st.error(f"Error en f(x): {e}")
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Condición")
        st.write("Debe cumplirse: **f(a)·f(b) < 0**")
        st.subheader("🧾 Fórmula")
        st.latex(r"c=\frac{a+b}{2}")
        st.write("Se actualiza el intervalo según el signo de f(c).")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# NEWTON-RAPHSON
# =========================
elif metodo == "Newton-Raphson":
    with colA:
        section_title("⚡ Newton-Raphson", "Raíz usando derivada numérica (no necesitas f'(x)).")
        expr = st.text_input("f(x) =", value="x**3 - x - 1")
        x0 = st.number_input("x0 (inicial)", value=1.5)
        tol = st.number_input("Tolerancia", value=1e-8, format="%.10f")
        nmax = st.number_input("Iteraciones máximas", value=30, step=1)
        h = st.number_input("h (derivada numérica)", value=1e-6, format="%.10f")

        if st.button("🚀 Resolver", type="primary"):
            try:
                f = lambda x: safe.eval_expr(expr, x=x)
                root, steps = newton_raphson(f, x0, tol=tol, nmax=int(nmax), h=h)
                st.success(f"✅ Raíz aproximada: {root:.12f}")
                show_steps_table(steps)
            except SafeEvalError as e:
                st.error(f"Error en f(x): {e}")
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Fórmula")
        st.latex(r"x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}")
        st.write("Aquí usamos derivada numérica central:")
        st.latex(r"f'(x)\approx \frac{f(x+h)-f(x-h)}{2h}")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# NEWTON SISTEMAS 2D
# =========================
elif metodo == "Newton no lineales (Sistema 2D)":
    with colA:
        section_title("🧩 Newton para sistemas no lineales (2 variables)", "Resuelve F(x,y)=0 y G(x,y)=0 con Jacobiano numérico.")
        fx = st.text_input("F(x,y) =", value="x**2 + y**2 - 4")
        gy = st.text_input("G(x,y) =", value="x - y")
        x0 = st.number_input("x0", value=1.0)
        y0 = st.number_input("y0", value=1.0)
        tol = st.number_input("Tolerancia", value=1e-10, format="%.12f")
        nmax = st.number_input("Iteraciones máximas", value=30, step=1)
        h = st.number_input("h (Jacobiano numérico)", value=1e-6, format="%.10f")

        if st.button("🚀 Resolver", type="primary"):
            try:
                F = lambda x, y: safe.eval_expr(fx, x=x, y=y)
                G = lambda x, y: safe.eval_expr(gy, x=x, y=y)
                (xr, yr), steps = newton_system_2d(F, G, x0, y0, tol=tol, nmax=int(nmax), h=h)
                st.success(f"✅ Solución aproximada: x={xr:.12f}, y={yr:.12f}")
                show_steps_table(steps)
            except SafeEvalError as e:
                st.error(f"Error en funciones: {e}")
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Idea")
        st.write("Se resuelve un sistema lineal en cada iteración:")
        st.latex(r"J(x_k,y_k)\,\Delta = -\begin{bmatrix}F(x_k,y_k)\\G(x_k,y_k)\end{bmatrix}")
        st.latex(r"\begin{bmatrix}x_{k+1}\\y_{k+1}\end{bmatrix}=\begin{bmatrix}x_k\\y_k\end{bmatrix}+\Delta")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# JACOBI
# =========================
elif metodo == "Jacobi":
    with colA:
        section_title("🧮 Jacobi", "Resuelve Ax=b iterando (requiere dominancia diagonal idealmente).")
        A_txt = st.text_area("Matriz A (filas separadas por enter, columnas por espacio)", value="10 -1 2\n-1 11 -1\n2 -1 10", height=120)
        b_txt = st.text_input("Vector b (separado por espacio)", value="6 25 -11")
        x0_txt = st.text_input("x0 (opcional, vacío = ceros)", value="")
        tol = st.number_input("Tolerancia", value=1e-10, format="%.12f")
        nmax = st.number_input("Iteraciones máximas", value=100, step=1)

        if st.button("🚀 Resolver", type="primary"):
            try:
                A = parse_matrix_text(A_txt)
                b = parse_vector_text(b_txt)
                x0 = parse_vector_text(x0_txt) if x0_txt.strip() else np.zeros_like(b)
                x, steps = jacobi(A, b, x0=x0, tol=tol, nmax=int(nmax))
                st.success("✅ Solución aproximada:")
                st.write(x)
                show_steps_table(steps)
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Fórmula")
        st.latex(r"x_i^{(k+1)}=\frac{1}{a_{ii}}\left(b_i-\sum_{j\ne i}a_{ij}x_j^{(k)}\right)")
        st.write("Usa norma infinito del error para detener.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# GAUSS-SEIDEL
# =========================
elif metodo == "Gauss-Seidel":
    with colA:
        section_title("🧮 Gauss–Seidel", "Como Jacobi, pero actualiza en caliente y suele converger más rápido.")
        A_txt = st.text_area("Matriz A (filas enter, columnas espacio)", value="4 1 2\n3 5 1\n1 1 3", height=120)
        b_txt = st.text_input("Vector b", value="4 7 3")
        x0_txt = st.text_input("x0 (opcional, vacío = ceros)", value="")
        tol = st.number_input("Tolerancia", value=1e-10, format="%.12f")
        nmax = st.number_input("Iteraciones máximas", value=100, step=1)
        w = st.number_input("Factor de relajación ω (1 = normal)", value=1.0, format="%.6f")

        if st.button("🚀 Resolver", type="primary"):
            try:
                A = parse_matrix_text(A_txt)
                b = parse_vector_text(b_txt)
                x0 = parse_vector_text(x0_txt) if x0_txt.strip() else np.zeros_like(b)
                x, steps = gauss_seidel(A, b, x0=x0, tol=tol, nmax=int(nmax), w=w)
                st.success("✅ Solución aproximada:")
                st.write(x)
                show_steps_table(steps)
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Fórmula")
        st.latex(r"x_i^{(k+1)}=\frac{1}{a_{ii}}\left(b_i-\sum_{j<i}a_{ij}x_j^{(k+1)}-\sum_{j>i}a_{ij}x_j^{(k)}\right)")
        st.write("Incluye ω (SOR): x ← ω*x_new + (1-ω)*x_old")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DIFERENCIAS FINITAS (Poisson 1D)
# =========================
elif metodo == "Diferencias Finitas (Poisson 1D)":
    with colA:
        section_title("📐 Diferencias Finitas (Poisson 1D)", "Resuelve: -u''(x)=f(x), u(a)=ua, u(b)=ub")
        expr = st.text_input("f(x) =", value="1")  # -u''=1 => u''=-1
        a = st.number_input("a", value=0.0)
        b = st.number_input("b", value=1.0)
        ua = st.number_input("u(a)", value=0.0)
        ub = st.number_input("u(b)", value=0.0)
        n = st.number_input("n (puntos interiores)", value=10, step=1)

        if st.button("🚀 Resolver", type="primary"):
            try:
                f = lambda x: safe.eval_expr(expr, x=x)
                xs, us, A, rhs = poisson_1d_dirichlet(f, a, b, ua, ub, int(n))
                st.success("✅ Solución aproximada u(x) en nodos:")
                df = pd.DataFrame({"x": xs, "u(x)": us})
                st.dataframe(df, use_container_width=True)

                st.subheader("Sistema lineal (A u = rhs)")
                st.write("A (tridiagonal):")
                st.dataframe(pd.DataFrame(A), use_container_width=True)
                st.write("rhs:")
                st.write(rhs)

            except SafeEvalError as e:
                st.error(f"Error en f(x): {e}")
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Discretización")
        st.latex(r"-u''(x_i)\approx -\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}=f(x_i)")
        st.write("Esto produce una matriz tridiagonal fácil de resolver.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LAGRANGE
# =========================
elif metodo == "Lagrange (Interpolación)":
    with colA:
        section_title("📌 Interpolación de Lagrange", "Dado (xi, yi), estima f(xq).")
        pts_txt = st.text_area("Puntos (uno por línea): x y", value="0 1\n1 3\n2 2", height=140)
        xq = st.number_input("x a evaluar", value=1.5)

        if st.button("🚀 Interpolar", type="primary"):
            try:
                xs, ys = parse_points_text(pts_txt)
                yq, steps = lagrange_interpolate(xs, ys, xq)
                st.success(f"✅ P({xq}) = {yq:.12f}")
                show_steps_table(steps)
            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Fórmula")
        st.latex(r"P(x)=\sum_{i=0}^{n-1}y_i\prod_{j\ne i}\frac{x-x_j}{x_i-x_j}")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# NEWTON INTERPOLATION
# =========================
elif metodo == "Interpolación de Newton (Divididas)":
    with colA:
        section_title("📌 Interpolación de Newton", "Construye polinomio con diferencias divididas y evalúa.")
        pts_txt = st.text_area("Puntos (uno por línea): x y", value="0 1\n1 3\n2 2\n3 5", height=160)
        xq = st.number_input("x a evaluar", value=2.5)

        if st.button("🚀 Interpolar", type="primary"):
            try:
                xs, ys = parse_points_text(pts_txt)
                coef, table = newton_divided_differences(xs, ys)
                yq = newton_eval(xs, coef, xq)
                st.success(f"✅ P({xq}) = {yq:.12f}")

                st.subheader("Tabla de diferencias divididas")
                st.dataframe(pd.DataFrame(table), use_container_width=True)

                st.subheader("Coeficientes (primera fila)")
                st.write(coef)

            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Forma de Newton")
        st.latex(r"P(x)=a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\cdots")
        st.write("Los a_k salen de la tabla (primera fila).")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MINIMOS CUADRADOS
# =========================
elif metodo == "Mínimos Cuadrados (Polinomio)":
    with colA:
        section_title("📉 Mínimos Cuadrados (ajuste polinomial)", "Ajusta un polinomio de grado m a datos (xi, yi).")
        pts_txt = st.text_area("Puntos (uno por línea): x y", value="0 1\n1 2\n2 2\n3 3\n4 5", height=160)
        grado = st.number_input("Grado del polinomio", value=2, step=1)
        xq = st.number_input("Evaluar en x =", value=2.5)

        if st.button("🚀 Ajustar", type="primary"):
            try:
                xs, ys = parse_points_text(pts_txt)
                coef, info = poly_least_squares_fit(xs, ys, int(grado))
                yq = poly_eval(coef, xq)

                st.success("✅ Ajuste listo")
                st.write("Coeficientes (de menor a mayor grado):")
                st.write(coef)
                st.info(f"Error SSE (suma de cuadrados): {info['SSE']:.12f}")
                st.write(f"P({xq}) = {yq:.12f}")

                st.subheader("Detalles")
                st.dataframe(pd.DataFrame(info["steps"]), use_container_width=True)

            except Exception as e:
                st.error(str(e))

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Objetivo")
        st.latex(r"\min_{\beta}\sum_{i=1}^n (y_i-P(x_i))^2")
        st.write("Se resuelve con ecuaciones normales usando álgebra lineal.")
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("Hecho para: Simulación y Computación Numérica • Python + Streamlit • Modular y ampliable.")
