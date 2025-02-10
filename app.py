import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuración del diseño de la página
st.set_page_config(page_title="Generador y Analizador de Contraseñas", layout="wide")

# Estilos personalizados
st.markdown(
    '''
    <style>
        body {
            background-color: #f4f4f4;
            color: #333;
        }
        .stApp {
            background: url("https://source.unsplash.com/random/1600x900/?technology") no-repeat center center fixed;
            background-size: cover;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

# Función para generar contraseñas
def generate_password(strength="strong"):
    if strength == "weak":
        return "".join(np.random.choice(list("abcdef123"), size=6))
    return "".join(np.random.choice(list("ABCDEFGHabcdef123456!@#"), size=12))

# Función para analizar seguridad
def analyze_password(password):
    length = len(password)
    uppercase = sum(1 for c in password if c.isupper())
    digits = sum(1 for c in password if c.isdigit())
    special_chars = sum(1 for c in password if c in "!@#$%^&*")

    score = length * 2 + uppercase * 3 + digits * 3 + special_chars * 4
    return score

# Función para mostrar gráficos
def plot_password_analysis(password):
    features = {
        "Longitud": len(password),
        "Mayúsculas": sum(c.isupper() for c in password),
        "Dígitos": sum(c.isdigit() for c in password),
        "Especiales": sum(c in "!@#$%^&*" for c in password),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(features.keys()), y=list(features.values()), palette="husl", ax=ax)
    ax.set_title("Análisis de Características de la Contraseña")
    st.pyplot(fig)

# Menú de navegación
st.sidebar.title("🔒 Menú")
option = st.sidebar.radio("Selecciona una opción", ["Generar Contraseña", "Analizar Contraseña", "Entrenar Modelo"])

# Sección de generación de contraseña
if option == "Generar Contraseña":
    st.title("🔑 Generador de Contraseñas")
    strength = st.selectbox("Nivel de Seguridad", ["Débil", "Fuerte"])
    password = generate_password("weak" if strength == "Débil" else "strong")
    st.success(f"Contraseña Generada: `{password}`")
    plot_password_analysis(password)

# Sección de análisis de contraseña
elif option == "Analizar Contraseña":
    st.title("📊 Análisis de Seguridad de Contraseña")
    password = st.text_input("Introduce tu contraseña:")
    if password:
        score = analyze_password(password)
        st.write(f"Puntuación de seguridad: **{score}**")
        plot_password_analysis(password)

# Sección de entrenamiento del modelo (futuro desarrollo)
elif option == "Entrenar Modelo":
    st.title("🚀 Entrenar Modelo de IA")
    st.write("Esta función se implementará pronto.")

# Mensaje final
st.sidebar.info("Desarrollado por WildPassPro 2.0")
