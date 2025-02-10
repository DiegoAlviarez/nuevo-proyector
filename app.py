import streamlit as st
import random
import string
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Password AI", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: cyan;
            animation: fadeIn 2s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .rotating-text {
            font-size: 24px;
            color: yellow;
            animation: rotateWords 6s infinite;
        }
        @keyframes rotateWords {
            0% { opacity: 0; }
            25% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 0; }
        }
    </style>
""", unsafe_allow_html=True)

# Función para generar contraseñas
def generate_password(length=12, strong=True):
    if strong:
        chars = string.ascii_letters + string.digits + string.punctuation
    else:
        chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

# Sección de bienvenida con animación
st.markdown('<h1 class="title">🔐 Bienvenido a Password AI 🔐</h1>', unsafe_allow_html=True)

# Tablero de palabras rotativas
rotating_words = ["🔒 Seguridad", "🔑 Contraseñas", "🤖 Inteligencia Artificial", "💡 Protección", "📊 Análisis"]
st.markdown(f'<p class="rotating-text">{random.choice(rotating_words)}</p>', unsafe_allow_html=True)

# Sección de generación de contraseñas
st.sidebar.title("📌 Menú")
menu_option = st.sidebar.radio("Selecciona una opción:", ["Inicio", "Generar Contraseña", "Analizar Seguridad"])

if menu_option == "Generar Contraseña":
    st.subheader("🛠️ Generador de Contraseñas")
    length = st.slider("Selecciona la longitud:", 6, 20, 12)
    strong = st.checkbox("¿Contraseña segura?", True)
    
    if st.button("Generar"):
        password = generate_password(length, strong)
        st.success(f"Tu contraseña generada es: `{password}`")
        
        # Análisis básico visual
        fig, ax = plt.subplots()
        sns.barplot(x=["Longitud", "Mayúsculas", "Dígitos", "Símbolos"],
                    y=[len(password), sum(c.isupper() for c in password), sum(c.isdigit() for c in password),
                       sum(c in string.punctuation for c in password)], palette="viridis", ax=ax)
        ax.set_title("Características de la Contraseña")
        st.pyplot(fig)

elif menu_option == "Analizar Seguridad":
    st.subheader("🔍 Analizador de Seguridad")
    password_input = st.text_input("Introduce tu contraseña:")
    
    if st.button("Analizar"):
        if password_input:
            score = sum([len(password_input) >= 8, any(c.isupper() for c in password_input),
                         any(c.isdigit() for c in password_input), any(c in string.punctuation for c in password_input)])
            st.info(f"🔎 Seguridad de la contraseña: {['Muy Débil', 'Débil', 'Media', 'Fuerte', 'Muy Fuerte'][score]}")
            
            # Gráfica de análisis
            fig, ax = plt.subplots()
            sns.barplot(x=["Longitud", "Mayúsculas", "Dígitos", "Símbolos"],
                        y=[len(password_input), sum(c.isupper() for c in password_input), sum(c.isdigit() for c in password_input),
                           sum(c in string.punctuation for c in password_input)], palette="magma", ax=ax)
            ax.set_title("Análisis de Seguridad")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Ingresa una contraseña para analizar.")

else:
    st.write("💡 Usa el menú para generar o analizar contraseñas.")

# Pie de página
st.markdown("<br><br><center>🔥 Desarrollado con ❤️ por Password AI</center>", unsafe_allow_html=True)
