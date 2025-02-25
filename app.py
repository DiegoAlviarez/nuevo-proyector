import streamlit as st
import hashlib
import pandas as pd
import numpy as np
import requests
import openai
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from cryptography.fernet import Fernet
import os

# ================= CONFIGURACIÓN =================
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"
MODEL_URL = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/password_strength_model.h5"

# ================= FUNCIONES =================
def generar_clave_cifrado():
    if not os.path.exists("clave.key"):
        clave = Fernet.generate_key()
        with open("clave.key", "wb") as archivo_clave:
            archivo_clave.write(clave)
    return open("clave.key", "rb").read()

CLAVE_CIFRADO = generar_clave_cifrado()
fernet = Fernet(CLAVE_CIFRADO)

def cargar_modelo_desde_github(url=MODEL_URL):
    """Descarga y carga el modelo desde GitHub."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("password_strength_model.h5", "wb") as f:
            f.write(response.content)
        return load_model("password_strength_model.h5")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def extraer_caracteristicas(password):
    """Extrae características relevantes de la contraseña."""
    return np.array([
        len(password),
        int(any(c.isupper() for c in password)),
        int(any(c.isdigit() for c in password)),
        int(any(c in "!@#$%^&*()" for c in password)),
        int(password.lower() in ["diego", "juan", "maria", "pedro", "media"]),
        int(any(seq in password.lower() for seq in ["123", "abc", "809"]))
    ]).reshape(1, -1)

def predecir_fortaleza(model, password):
    """Predice la fortaleza de la contraseña con la red neuronal."""
    features = extraer_caracteristicas(password)
    prediction = model.predict(features, verbose=0)
    return np.argmax(prediction)  # 0: débil, 1: media, 2: fuerte

def explicar_fortaleza(password):
    """Explica las razones detrás de la predicción de fortaleza."""
    explicaciones = []
    explicaciones.append("✅ Longitud adecuada" if len(password) >= 12 else "❌ Longitud insuficiente")
    explicaciones.append("✅ Contiene mayúsculas" if any(c.isupper() for c in password) else "❌ Falta de mayúsculas")
    explicaciones.append("✅ Contiene números" if any(c.isdigit() for c in password) else "❌ Falta de números")
    explicaciones.append("✅ Contiene símbolos" if any(c in "!@#$%^&*()" for c in password) else "❌ Falta de símbolos")
    explicaciones.append("❌ Contiene nombre común" if password.lower() in ["diego", "juan", "maria", "pedro", "media"] else "✅ No contiene nombres comunes")
    explicaciones.append("❌ Contiene secuencia simple" if any(seq in password.lower() for seq in ["123", "abc", "809"]) else "✅ No contiene secuencias simples")
    return explicaciones

# ================= INTERFAZ STREAMLIT =================
def main():
    st.set_page_config(page_title="🔐 WildPassPro", layout="wide")
    st.title("🔐 WildPassPro - Evaluador de Contraseñas")

    with st.spinner("Cargando modelo..."):
        model = cargar_modelo_desde_github()
        if model:
            st.success("Modelo cargado con éxito.")
        else:
            st.stop()

    password = st.text_input("Ingrese la contraseña para analizar:", type="password")

    if password:
        prediction = predecir_fortaleza(model, password)
        labels = ["DÉBIL 🔴", "MEDIA 🟡", "FUERTE 🟢"]
        st.subheader(f"Fortaleza: {labels[prediction]}")

        st.markdown("### Explicación de la fortaleza:")
        for exp in explicar_fortaleza(password):
            st.write(exp)

if __name__ == "__main__":
    main()
