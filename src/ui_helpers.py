import numpy as np
import pandas as pd
import streamlit as st

def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

def show_steps_table(steps):
    if steps is None or len(steps) == 0:
        st.info("No hay pasos para mostrar.")
        return
    df = pd.DataFrame(steps)
    st.dataframe(df, use_container_width=True)

def parse_points_text(text: str):
    xs, ys = [], []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("Cada línea debe tener exactamente 2 valores: x y")
        x, y = float(parts[0]), float(parts[1])
        xs.append(x); ys.append(y)
    if len(xs) < 2:
        raise ValueError("Necesitas al menos 2 puntos.")
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def parse_matrix_text(text: str):
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        row = [float(x) for x in line.replace(",", " ").split()]
        rows.append(row)
    A = np.array(rows, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser cuadrada (n x n).")
    return A

def parse_vector_text(text: str):
    if not text.strip():
        return np.array([], dtype=float)
    vals = [float(x) for x in text.replace(",", " ").split()]
    return np.array(vals, dtype=float)
