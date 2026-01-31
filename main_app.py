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
    page_title="EDA Dashboard Universal",
    layout="wide"
)

st.title("📊 Dashboard EDA Universal")
st.caption("Carga cualquier dataset y explóralo sin errores")

# ======================
# Sidebar – carga de datos
# ======================
st.sidebar.header("📂 Cargar datos")

uploaded_file = st.sidebar.file_uploader(
    "CSV o Excel",
    type=["csv", "xlsx"]
)

# ======================
# Carga segura de datos
# ======================
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

        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how="all")

        return df, None

    except Exception as e:
        return None, str(e)

if uploaded_file is None:
    st.info("⬅️ Carga un archivo para comenzar")
    st.stop()

df, error = safe_load(uploaded_file)

if error:
    st.error("❌ Error al cargar el archivo")
    st.code(error)
    st.stop()

# ======================
# Preparación del dataset
# ======================
df = df.copy()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# ======================
# KPIs principales
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
with st.expander("📄 Vista previa", expanded=True):
    st.dataframe(df.head(50), use_container_width=True)

# ======================
# Navegación
# ======================
section = st.sidebar.radio(
    "🧭 Sección",
    [
        "Análisis Cualitativo",
        "Análisis Cuantitativo",
        "Análisis Cuantitativo Gráfico"
    ]
)

# ======================================================
# ANÁLISIS CUALITATIVO
# ======================================================
if section == "Análisis Cualitativo":
    st.subheader("🧩 Análisis Cualitativo")

    if not categorical_cols:
        st.warning("No hay variables categóricas disponibles")
        st.stop()

    cat_col = st.selectbox(
        "Variable categórica",
        categorical_cols
    )

    freq = (
        df[cat_col]
        .value_counts(dropna=False)
        .reset_index()
    )
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
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# ANÁLISIS CUANTITATIVO
# ======================================================
elif section == "Análisis Cuantitativo":
    st.subheader("📐 Análisis Cuantitativo")

    if not numeric_cols:
        st.warning("No hay variables numéricas disponibles")
        st.stop()

    num_col = st.selectbox(
        "Variable numérica",
        numeric_cols
    )

    series = df[num_col].dropna()

    stats = pd.DataFrame({
        "Métrica": [
            "Media", "Mediana", "Desv. estándar",
            "Mínimo", "Máximo",
            "Asimetría", "Curtosis"
        ],
        "Valor": [
            series.mean(),
            series.median(),
            series.std(),
            series.min(),
            series.max(),
            series.skew(),
            series.kurtosis()
        ]
    })

    st.dataframe(stats, use_container_width=True)

# ======================================================
# ANÁLISIS CUANTITATIVO GRÁFICO
# ======================================================
elif section == "Análisis Cuantitativo Gráfico":
    st.subheader("📊 Análisis Cuantitativo Gráfico")

    if not numeric_cols:
        st.warning("No hay variables numéricas disponibles")
        st.stop()

    selected = st.selectbox(
        "Variable",
        numeric_cols
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x=selected,
            nbins=30,
            marginal="box",
            title=f"Histograma de {selected}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            y=selected,
            title=f"Boxplot de {selected}"
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        st.divider()
        st.subheader("🔗 Relación entre variables")

        x = st.selectbox("Eje X", numeric_cols, index=0)
        y = st.selectbox("Eje Y", numeric_cols, index=1)

        fig = px.scatter(
            df,
            x=x,
            y=y,
            trendline="ols",
            title=f"{x} vs {y}"
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
