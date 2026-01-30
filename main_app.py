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
    page_title="Dashboard EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-box {
    background-color: #f5f7fa;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Sidebar
# ======================
st.sidebar.title("📊 EDA Dashboard")
st.sidebar.markdown("Exploración visual de datos")

uploaded_file = st.sidebar.file_uploader(
    "📂 Cargar archivo (CSV / Excel)",
    type=["csv", "xlsx"]
)

section = st.sidebar.radio(
    "🧭 Navegación",
    [
        "Vista General",
        "Calidad de Datos",
        "Análisis Univariado",
        "Análisis Bivariado"
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

# ======================
# Header
# ======================
st.title("📊 Dashboard de Análisis Exploratorio")
st.caption(f"Archivo cargado: **{uploaded_file.name}**")

# ======================
# KPIs principales
# ======================
total_rows = df.shape[0]
total_cols = df.shape[1]
total_nulls = df.isna().sum().sum()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 Filas", total_rows)
col2.metric("📊 Columnas", total_cols)
col3.metric("⚠️ Nulos", total_nulls)
col4.metric("🔢 Numéricas", len(numeric_cols))

st.divider()

# ======================
# VISTA GENERAL
# ======================
if section == "Vista General":
    st.subheader("🔍 Vista general del dataset")

    with st.expander("📄 Vista previa de los datos", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("ℹ️ Información de variables"):
        info_df = pd.DataFrame({
            "Tipo": df.dtypes.astype(str),
            "Nulos": df.isna().sum(),
            "Únicos": df.nunique()
        })
        st.dataframe(info_df, use_container_width=True)

    with st.expander("📈 Estadística descriptiva"):
        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True
        )

# ======================
# CALIDAD DE DATOS
# ======================
elif section == "Calidad de Datos":
    st.subheader("🧪 Calidad de los datos")

    null_df = df.isna().sum().reset_index()
    null_df.columns = ["Variable", "Valores nulos"]
    null_df = null_df[null_df["Valores nulos"] > 0]

    if null_df.empty:
        st.success("✅ No se encontraron valores nulos")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(null_df, use_container_width=True)

        with col2:
            fig = px.bar(
                null_df,
                x="Variable",
                y="Valores nulos",
                title="Valores nulos por variable"
            )
            st.plotly_chart(fig, use_container_width=True)

# ======================
# ANÁLISIS UNIVARIADO
# ======================
elif section == "Análisis Univariado":
    st.subheader("📊 Análisis univariado")

    if not numeric_cols:
        st.warning("No hay variables numéricas disponibles")
        st.stop()

    selected_col = st.selectbox(
        "Selecciona una variable numérica",
        numeric_cols
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x=selected_col,
            nbins=30,
            marginal="box",
            title=f"Distribución de {selected_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stats = df[selected_col].describe().to_frame("Valor")
        st.dataframe(stats, use_container_width=True)

# ======================
# ANÁLISIS BIVARIADO
# ======================
elif section == "Análisis Bivariado":
    st.subheader("🔗 Análisis bivariado y correlaciones")

    if len(numeric_cols) < 2:
        st.warning("Se requieren al menos dos variables numéricas")
        st.stop()

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("### 🔎 Relación entre dos variables")

    col_x = st.selectbox("Eje X", numeric_cols, index=0)
    col_y = st.selectbox("Eje Y", numeric_cols, index=1)

    fig = px.scatter(
        df,
        x=col_x,
        y=col_y,
        trendline="ols",
        title=f"{col_x} vs {col_y}"
    )
    st.plotly_chart(fig, use_container_width=True)
