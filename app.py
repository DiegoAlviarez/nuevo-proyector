import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
import string
from collections import Counter

# Estilos personalizados para Streamlit
st.set_page_config(
    page_title="WildPassPro - Seguridad de Contraseñas",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Establecer fondo personalizado con CSS
page_bg_img = """
<style>
    body {
        background-image: url("https://source.unsplash.com/1600x900/?technology,security");
        background-size: cover;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    h1 {
        color: #333;
    }
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Función para generar contraseñas aleatorias
def generar_contrasena(longitud=12, nivel="media"):
    caracteres = string.ascii_letters
    if nivel == "alta":
        caracteres += string.digits + string.punctuation
    elif nivel == "media":
        caracteres += string.digits
    return "".join(random.choice(caracteres) for _ in range(longitud))

# Extraer características de una contraseña
def extraer_caracteristicas(password):
    return [
        len(password),
        sum(c.isupper() for c in password),
        sum(c.isdigit() for c in password),
        sum(c in string.punctuation for c in password)
    ]

# Función para analizar seguridad de la contraseña
def analizar_contrasena(password):
    features = extraer_caracteristicas(password)
    score = (
        (features[0] / 16) * 0.4 +  # Longitud (40%)
        (features[1] / 4) * 0.2 +   # Mayúsculas (20%)
        (features[2] / 4) * 0.2 +   # Números (20%)
        (features[3] / 4) * 0.2     # Símbolos (20%)
    ) * 100
    return min(score, 100)  # Límite máximo de 100%

# Graficar análisis de la contraseña
def graficar_contrasena(password):
    features = extraer_caracteristicas(password)
    labels = ["Longitud", "Mayúsculas", "Números", "Símbolos"]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, features, color=['#FF5733', '#33FF57', '#3357FF', '#FF33A1'])
    ax.set_ylim(0, max(features) + 2)
    ax.set_title("Características de la Contraseña")
    st.pyplot(fig)

# Menú lateral interactivo
st.sidebar.title("🔐 WildPassPro")
opcion = st.sidebar.radio("Navegación", ["Inicio", "Generador", "Analizador"])

# Sección de bienvenida
if opcion == "Inicio":
    st.title("Bienvenido a WildPassPro 🔥")
    st.write("""
    **WildPassPro** es una herramienta avanzada para generar y analizar contraseñas con inteligencia artificial.
    """)
    st.image("https://source.unsplash.com/800x400/?password,security", use_column_width=True)

# Sección de generación de contraseñas
elif opcion == "Generador":
    st.title("🔑 Generador de Contraseñas")
    nivel = st.selectbox("Selecciona el nivel de seguridad", ["Baja", "Media", "Alta"])
    longitud = st.slider("Longitud de la contraseña", 6, 20, 12)
    
    if st.button("Generar Contraseña"):
        nueva_contrasena = generar_contrasena(longitud, nivel.lower())
        st.success(f"✅ Tu contraseña generada: **{nueva_contrasena}**")
        graficar_contrasena(nueva_contrasena)

# Sección de análisis de contraseñas
elif opcion == "Analizador":
    st.title("🛡️ Analizador de Contraseñas")
    password = st.text_input("Ingresa tu contraseña para analizarla", type="password")
    
    if password:
        seguridad = analizar_contrasena(password)
        st.subheader(f"🔍 Nivel de Seguridad: {seguridad:.2f}%")
        
        if seguridad > 80:
            st.success("✅ ¡Esta contraseña es muy segura!")
        elif seguridad > 50:
            st.warning("⚠️ Esta contraseña es medianamente segura.")
        else:
            st.error("❌ Esta contraseña es débil. Te recomendamos mejorarla.")

        graficar_contrasena(password)

