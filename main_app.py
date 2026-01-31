import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
import json

# ======================================================
# Configuración general
# ======================================================
st.set_page_config(
    page_title="EDA Dashboard Universal + IA",
    layout="wide"
)

st.title("📊 Dashboard EDA Universal + Asistente IA")
st.caption("Explora cualquier dataset de forma interactiva y asistida por IA")

# ======================================================
# Sidebar – Carga de datos
# ======================================================
st.sidebar.header("📂 Cargar datos")

uploaded_file = st.sidebar.file_uploader(
    "CSV o Excel",
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

        # Limpieza mínima
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

# ======================================================
# Sidebar – Controles dinámicos
# ======================================================
st.sidebar.header("🎛️ Controles de análisis")

max_rows = len(df)
n_rows = st.sidebar.slider(
    "Cantidad de muestras a analizar",
    min_value=min(10, max_rows),
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
categorical_cols = df_view.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# ======================================================
# Sidebar – Opciones de vista
# ======================================================
st.sidebar.header("🧭 Vistas")

show_table = st.sidebar.checkbox("Mostrar tabla de datos", True)
show_stats = st.sidebar.checkbox("Mostrar estadística descriptiva", True)
show_graphs = st.sidebar.checkbox("Mostrar gráficos", True)
show_corr = st.sidebar.checkbox("Mostrar correlaciones", False)

view_mode = st.sidebar.selectbox(
    "Modo de visualización",
    ["Compacto", "Detallado"]
)

# ======================================================
# KPIs
# ======================================================
st.subheader("📌 Resumen del Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas analizadas", df_view.shape[0])
c2.metric("Columnas", df_view.shape[1])
c3.metric("Numéricas", len(numeric_cols))
c4.metric("Categóricas", len(categorical_cols))

st.divider()

# ======================================================
# Tabla de datos
# ======================================================
if show_table:
    with st.expander("📄 Datos", expanded=view_mode == "Detallado"):
        st.dataframe(df_view, use_container_width=True)

# ======================================================
# Estadística descriptiva
# ======================================================
if show_stats and numeric_cols:
    with st.expander("📐 Estadística descriptiva", expanded=view_mode == "Detallado"):
        st.dataframe(
            df_view[numeric_cols].describe().T,
            use_container_width=True
        )

# ======================================================
# Gráficos dinámicos
# ======================================================
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

# ======================================================
# Scatter dinámico
# ======================================================
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

# ======================================================
# Correlaciones
# ======================================================
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

# ======================================================
# ASISTENTE IA – GROQ + LLAMA 3.3
# ======================================================
st.sidebar.divider()
st.sidebar.header("🤖 Asistente de Análisis IA")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

enable_ai = st.sidebar.checkbox("Activar asistente IA", False)

def run_groq_analysis(api_key, df_context):
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        summary = {
            "shape": df_context.shape,
            "columns": list(df_context.columns),
            "numeric_summary": df_context.select_dtypes(include=np.number)
            .describe()
            .round(2)
            .to_dict(),
            "categorical_summary": {
                col: df_context[col].value_counts().head(5).to_dict()
                for col in df_context.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns
            }
        }

        prompt = f"""
Eres un analista de datos senior.
Analiza el siguiente resumen de un dataset y entrega:

1. Insights clave
2. Patrones relevantes
3. Posibles problemas de calidad
4. Recomendaciones de análisis o negocio

Resumen del dataset:
{json.dumps(summary, indent=2)}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis exploratorio de datos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=900
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error en el asistente IA:\n{str(e)}"

# ======================================================
# UI del asistente
# ======================================================
st.divider()
st.subheader("🤖 Asistente Inteligente de Análisis")

if not enable_ai:
    st.info("Activa el asistente IA desde el panel lateral")
elif not groq_api_key:
    st.warning("Ingresa tu Groq API Key para usar el asistente")
else:
    if st.button("🧠 Analizar datos con IA"):
        with st.spinner("Analizando datos con IA..."):
            result = run_groq_analysis(groq_api_key, df_view)

        st.markdown("### 📋 Resultados del análisis")
        st.markdown(result)
