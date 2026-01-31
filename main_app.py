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
    page_title="EDA Dashboard Dinámico",
    layout="wide"
)

st.title("📊 Dashboard EDA Dinámico")
st.caption("Explora cualquier dataset de forma interactiva y segura")

# ======================
# Sidebar – Carga de datos
# ======================
st.sidebar.header("📂 Datos")

uploaded_file = st.sidebar.file_uploader(
    "Cargar CSV o Excel",
    type=["csv", "xlsx"]
)

@st.cache_data
def safe_load(file):
    try:
        if file.name.endswith(".csv"):
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="latin-1")
        else:
            df = pd.read_excel(file)

        df = df.dropna(axis=1, how="all")
        return df, None
    except Exception as e:
        return None, str(e)

if uploaded_file is None:
    st.info("⬅️ Carga un archivo para comenzar")
    st.stop()

df, error = safe_load(uploaded_file)
if error:
    st.error("Error al cargar el archivo")
    st.code(error)
    st.stop()

# ======================
# Sidebar – Controles dinámicos
# ======================
st.sidebar.header("🎛️ Controles")

max_rows = len(df)
n_rows = st.sidebar.slider(
    "Cantidad de muestras a analizar",
    min_value=10,
    max_value=max_rows,
    value=min(500, max_rows),
    step=10
)

sample_mode = st.sidebar.radio(
    "Modo de muestreo",
    ["Primeras filas", "Aleatorio"]
)

if sample_mode == "Aleatorio":
    df_view = df.sample(n_rows, random_state=42)
else:
    df_view = df.head(n_rows)

numeric_cols = df_view.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_view.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# ======================
# Sidebar – Opciones de vista
# ======================
st.sidebar.header("🧭 Vistas")

show_table = st.sidebar.checkbox("Mostrar tabla de datos", True)
show_stats = st.sidebar.checkbox("Mostrar estadística descriptiva", True)
show_graphs = st.sidebar.checkbox("Mostrar gráficos", True)
show_corr = st.sidebar.checkbox("Mostrar correlaciones", False)

view_mode = st.sidebar.selectbox(
    "Modo de visualización",
    ["Compacto", "Detallado"]
)

# ======================
# KPIs
# ======================
st.subheader("📌 Resumen")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas analizadas", df_view.shape[0])
c2.metric("Columnas", df_view.shape[1])
c3.metric("Numéricas", len(numeric_cols))
c4.metric("Categóricas", len(categorical_cols))

st.divider()

# ======================
# Vista de tabla
# ======================
if show_table:
    with st.expander("📄 Datos", expanded=view_mode == "Detallado"):
        st.dataframe(df_view, use_container_width=True)

# ======================
# Estadística descriptiva
# ======================
if show_stats and numeric_cols:
    with st.expander("📐 Estadística descriptiva", expanded=view_mode == "Detallado"):
        st.dataframe(
            df_view[numeric_cols].describe().T,
            use_container_width=True
        )

# ======================
# Gráficos dinámicos
# ======================
if show_graphs and numeric_cols:
    st.subheader("📊 Gráficos interactivos")

    selected_nums = st.multiselect(
        "Selecciona variables numéricas",
        numeric_cols,
        default=numeric_cols[:1]
    )

    for col in selected_nums:
        fig = px.histogram(
            df_view,
            x=col,
            nbins=30,
            marginal="box",
            title=f"Distribución de {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================
# Scatter dinámico
# ======================
if show_graphs and len(numeric_cols) > 1:
    st.subheader("🔗 Relación entre variables")

    x_var = st.selectbox("Eje X", numeric_cols, index=0)
    y_var = st.selectbox("Eje Y", numeric_cols, index=1)

    fig = px.scatter(
        df_view,
        x=x_var,
        y=y_var,
        trendline="ols",
        title=f"{x_var} vs {y_var}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================
# Correlaciones
# ======================
if show_corr and len(numeric_cols) > 1:
    st.subheader("🧠 Matriz de correlación")

    corr = df_view[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        annot=view_mode == "Detallado",
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)
