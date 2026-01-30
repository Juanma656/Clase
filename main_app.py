import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="EDA con Streamlit",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Carga un archivo y explora tus datos de forma interactiva")

# =========================
# Carga de datos
# =========================
st.sidebar.header("📂 Cargar datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV o Excel",
    type=["csv", "xlsx"]
)

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.success(f"Archivo cargado: **{uploaded_file.name}**")

    # =========================
    # Vista general
    # =========================
    st.subheader("🔍 Vista general del dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores nulos", df.isna().sum().sum())

    st.dataframe(df.head())

    # =========================
    # Información del dataset
    # =========================
    st.subheader("ℹ️ Información de las variables")

    info_df = pd.DataFrame({
        "Tipo de dato": df.dtypes,
        "Valores nulos": df.isna().sum(),
        "Valores únicos": df.nunique()
    })

    st.dataframe(info_df)

    # =========================
    # Estadística descriptiva
    # =========================
    st.subheader("📈 Estadística descriptiva")
    st.dataframe(df.describe(include="all").transpose())

    # =========================
    # Selección de variables numéricas
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        st.subheader("📊 Análisis de variables numéricas")

        selected_col = st.selectbox(
            "Selecciona una variable",
            numeric_cols
        )

        col1, col2 = st.columns(2)

        # Histograma
        with col1:
            st.markdown("**Histograma**")
            fig = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Boxplot
        with col2:
            st.markdown("**Diagrama de caja**")
            fig = px.box(
                df,
                y=selected_col
            )
            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Correlación
        # =========================
        if len(numeric_cols) > 1:
            st.subheader("🔗 Matriz de correlación")

            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax
            )
            st.pyplot(fig)
    else:
        st.warning("No hay variables numéricas en el dataset.")

else:
    st.info("⬅️ Sube un archivo desde el panel lateral para comenzar.")

