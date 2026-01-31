import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# Configuración general
# ======================
st.set_page_config(
    page_title="Generador de Datos + EDA",
    layout="wide"
)

st.title("🧪 Generador Universal de Datos + Dashboard EDA")
st.caption("Crea cualquier dataset y explóralo visualmente")

# ======================
# Sidebar – Generador
# ======================
st.sidebar.header("⚙️ Configuración del Dataset")

n_rows = st.sidebar.slider("Número de filas", 50, 5000, 500)
n_num = st.sidebar.slider("Variables numéricas", 1, 10, 3)
n_cat = st.sidebar.slider("Variables categóricas", 0, 5, 1)
dist_type = st.sidebar.selectbox(
    "Distribución numérica",
    ["Normal", "Uniforme"]
)

generate = st.sidebar.button("🚀 Generar Dataset")

# ======================
# Generación de datos
# ======================
@st.cache_data
def generate_data(n_rows, n_num, n_cat, dist_type):
    data = {}

    # Numéricas
    for i in range(n_num):
        if dist_type == "Normal":
            data[f"num_{i+1}"] = np.random.normal(
                loc=np.random.randint(10, 100),
                scale=np.random.randint(5, 20),
                size=n_rows
            )
        else:
            data[f"num_{i+1}"] = np.random.uniform(
                low=0,
                high=np.random.randint(50, 200),
                size=n_rows
            )

    # Categóricas
    for i in range(n_cat):
        categories = [f"C{i+1}_{j}" for j in range(1, 6)]
        data[f"cat_{i+1}"] = np.random.choice(categories, n_rows)

    return pd.DataFrame(data)

if not generate:
    st.info("⬅️ Configura el dataset y haz clic en **Generar Dataset**")
    st.stop()

df = generate_data(n_rows, n_num, n_cat, dist_type)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

# ======================
# KPIs
# ======================
st.subheader("📌 Resumen del Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", df.shape[0])
c2.metric("Columnas", df.shape[1])
c3.metric("Numéricas", len(numeric_cols))
c4.metric("Categóricas", len(categorical_cols))

st.divider()

# ======================
# Vista general
# ======================
with st.expander("📄 Vista previa del dataset", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

# ======================
# Análisis Cualitativo
# ======================
if categorical_cols:
    st.subheader("🧩 Análisis Cualitativo")

    cat_col = st.selectbox(
        "Variable categórica",
        categorical_cols
    )

    freq = df[cat_col].value_counts().reset_index()
    freq.columns = ["Categoría", "Frecuencia"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(freq, use_container_width=True)

    with col2:
        fig = px.bar(
            freq,
            x="Categoría",
            y="Frecuencia",
            title=f"Distribución de {cat_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================
# Análisis Cuantitativo
# ======================
st.subheader("📐 Análisis Cuantitativo")

num_col = st.selectbox(
    "Variable numérica",
    numeric_cols
)

stats = df[num_col].describe().to_frame("Valor")
st.dataframe(stats, use_container_width=True)

# ======================
# Análisis Gráfico
# ======================
st.subheader("📊 Análisis Gráfico Interactivo")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x=num_col,
        nbins=30,
        marginal="box",
        title=f"Histograma de {num_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df,
        y=num_col,
        title=f"Boxplot de {num_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================
# Scatter y correlación
# ======================
if len(numeric_cols) > 1:
    st.divider()
    st.subheader("🔗 Relación entre variables")

    x_var = st.selectbox("Variable X", numeric_cols, index=0)
    y_var = st.selectbox("Variable Y", numeric_cols, index=1)

    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        trendline="ols",
        title=f"{x_var} vs {y_var}"
    )
    st.plotly_chart(fig, use_container_width=True)

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)
