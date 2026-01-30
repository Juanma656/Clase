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
    page_title="EDA Dashboard",
    layout="wide"
)

# ======================
# Sidebar
# ======================
st.sidebar.title("📊 EDA por Secciones")

uploaded_file = st.sidebar.file_uploader(
    "📂 Cargar archivo (CSV / Excel)",
    type=["csv", "xlsx"]
)

section = st.sidebar.radio(
    "🧭 Selecciona el análisis",
    [
        "Análisis Cualitativo",
        "Análisis Cuantitativo",
        "Análisis Cuantitativo Gráfico"
    ]
)

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# ======================
# Carga de datos
# ======================
if uploaded_file is None:
    st.info("⬅️ Carga un archivo desde el panel lateral para comenzar")
    st.stop()

df = load_data(uploaded_file)

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.title("📊 Exploratory Data Analysis (EDA)")
st.caption(f"Archivo cargado: **{uploaded_file.name}**")
st.divider()

# =========================================================
# 1️⃣ ANÁLISIS CUALITATIVO
# =========================================================
if section == "Análisis Cualitativo":
    st.subheader("🧩 Análisis Cualitativo (Variables categóricas)")

    if not categorical_cols:
        st.warning("No se encontraron variables categóricas")
        st.stop()

    selected_cat = st.selectbox(
        "Selecciona una variable categórica",
        categorical_cols
    )

    freq = (
        df[selected_cat]
        .value_counts(dropna=False)
        .reset_index()
    )
    freq.columns = ["Categoría", "Frecuencia"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Tabla de frecuencias")
        st.dataframe(freq, use_container_width=True)

        st.markdown(
            f"**Moda:** `{freq.iloc[0]['Categoría']}` "
            f"({freq.iloc[0]['Frecuencia']} registros)"
        )

    with col2:
        fig = px.bar(
            freq,
            x="Categoría",
            y="Frecuencia",
            title=f"Distribución de {selected_cat}",
            text="Frecuencia"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2️⃣ ANÁLISIS CUANTITATIVO (TABULAR)
# =========================================================
elif section == "Análisis Cuantitativo":
    st.subheader("📐 Análisis Cuantitativo (Estadístico)")

    if not numeric_cols:
        st.warning("No hay variables numéricas")
        st.stop()

    selected_num = st.selectbox(
        "Selecciona una variable numérica",
        numeric_cols
    )

    series = df[selected_num].dropna()

    stats = pd.DataFrame({
        "Métrica": [
            "Media", "Mediana", "Moda",
            "Desviación estándar",
            "Varianza",
            "Mínimo", "Máximo",
            "Asimetría", "Curtosis"
        ],
        "Valor": [
            series.mean(),
            series.median(),
            series.mode().iloc[0] if not series.mode().empty else np.nan,
            series.std(),
            series.var(),
            series.min(),
            series.max(),
            series.skew(),
            series.kurtosis()
        ]
    })

    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Estadística descriptiva")
        st.dataframe(stats, use_container_width=True)

    with col2:
        st.markdown("### 🚨 Detección de outliers (IQR)")
        st.metric("Cantidad de outliers", len(outliers))
        st.metric("Porcentaje",
                  f"{len(outliers) / len(series) * 100:.2f}%")

# =========================================================
# 3️⃣ ANÁLISIS CUANTITATIVO GRÁFICO
# =========================================================
elif section == "Análisis Cuantitativo Gráfico":
    st.subheader("📈 Análisis Cuantitativo Gráfico")

    if not numeric_cols:
        st.warning("No hay variables numéricas")
        st.stop()

    st.markdown("### 🔹 Distribución")

    selected_var = st.selectbox(
        "Selecciona una variable",
        numeric_cols
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x=selected_var,
            nbins=30,
            marginal="box",
            title=f"Histograma de {selected_var}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            y=selected_var,
            title=f"Boxplot de {selected_var}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🔗 Relación entre variables")

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

    st.divider()
    st.markdown("### 🧠 Correlaciones")

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
