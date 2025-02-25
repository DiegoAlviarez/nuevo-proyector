import streamlit as st
import hashlib
import pandas as pd
import numpy as np
import re
import requests
import openai
import joblib
import tensorflow as tf
import secrets
import string
import os
import io
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cryptography.fernet import Fernet
from tensorflow.keras.models import load_model

# Configuración de Groq
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# ========== NUEVAS CONSTANTES ==========
MASTER_PASSWORD = "WildPassPro2024!"  # Contraseña maestra (cambiar en producción)

# ========== FUNCIONES DE SEGURIDAD ==========
def generar_clave_cifrado():
    if not os.path.exists("clave.key"):
        clave = Fernet.generate_key()
        with open("clave.key", "wb") as archivo_clave:
            archivo_clave.write(clave)
    return open("clave.key", "rb").read()

CLAVE_CIFRADO = generar_clave_cifrado()
fernet = Fernet(CLAVE_CIFRADO)

def cifrar_archivo(ruta_archivo):
    with open(ruta_archivo, "rb") as archivo:
        datos = archivo.read()
    datos_cifrados = fernet.encrypt(datos)
    with open(ruta_archivo + ".encrypted", "wb") as archivo_cifrado:
        archivo_cifrado.write(datos_cifrados)
    os.remove(ruta_archivo)
    return f"{ruta_archivo}.encrypted"

def descifrar_archivo(ruta_archivo):
    with open(ruta_archivo, "rb") as archivo:
        datos_cifrados = archivo.read()
    datos_descifrados = fernet.decrypt(datos_cifrados)
    ruta_original = ruta_archivo.replace(".encrypted", "")
    with open(ruta_original, "wb") as archivo_descifrado:
        archivo_descifrado.write(datos_descifrados)
    return ruta_original

# ========== EFECTO MAQUINA DE ESCRIBIR ==========
def typewriter_effect(text):
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(f'<div class="chat-message">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(0.02)
    return displayed_text

# ========== FUNCIONES PRINCIPALES ==========
def generate_secure_password(length=16):
    characters = string.ascii_letters + string.digits + "!@#$%^&*()"
    return ''.join(secrets.choice(characters) for _ in range(length))

def generate_access_key():
    return secrets.token_urlsafe(32)

def load_weak_passwords(url):
    response = requests.get(url)
    return set(line.strip().lower() for line in response.text.splitlines() if line.strip())

WEAK_PASSWORDS = load_weak_passwords("https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/rockyou.txt")

def detect_weakness(password):
    weaknesses = []
    password_lower = password.lower()
    
    if password_lower in WEAK_PASSWORDS:
        weaknesses.append("❌ Está en la lista rockyou.txt")
    if password.islower():
        weaknesses.append("❌ Solo minúsculas")
    if password.isupper():
        weaknesses.append("❌ Solo mayúsculas")
    if not any(c.isdigit() for c in password):
        weaknesses.append("❌ Sin números")
    if not any(c in "!@#$%^&*()" for c in password):
        weaknesses.append("❌ Sin símbolos")
    if len(password) < 12:
        weaknesses.append(f"❌ Longitud insuficiente ({len(password)}/12)")
    if password_lower in ["diego", "juan", "maria", "pedro", "media"]:
        weaknesses.append("❌ Contiene un nombre común")
    if "123" in password or "abc" in password_lower or "809" in password:
        weaknesses.append("❌ Contiene una secuencia simple")
        
    return weaknesses

def groq_analysis(password):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""Analiza esta contraseña: '{password}'
                1. Vulnerabilidades críticas (longitud, complejidad, nombres comunes, secuencias simples)
                2. Comparación con patrones comunes (nombres propios, secuencias numéricas)
                3. Recomendaciones personalizadas (longitud mínima, uso de símbolos, evitar nombres comunes)
                Formato: Lista markdown con emojis"""
            }],
            temperature=0.4,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Error:** {str(e)}"

# ========== FUNCIONES DE LA RED NEURONAL ==========
def cargar_modelo_desde_github():
    """
    Descarga el modelo entrenado desde GitHub y lo carga.
    """
    url = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/password_strength_model.keras"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error si la descarga falla
        with open("password_strength_model.keras", "wb") as f:
            f.write(response.content)
        model = load_model("password_strength_model.keras")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def predecir_fortaleza(model, password):
    """
    Predice la fortaleza de una contraseña utilizando el modelo.
    """
    features = np.array([
        len(password),  # Longitud de la contraseña
        int(any(c.isupper() for c in password)),  # Contiene mayúsculas
        int(any(c.isdigit() for c in password)),  # Contiene números
        int(any(c in "!@#$%^&*()" for c in password)),  # Contiene símbolos
        int(password.lower() in ["diego", "juan", "maria", "pedro", "media"]),  # Nombres comunes
        int("123" in password or "abc" in password.lower() or "809" in password)  # Secuencias comunes
    ]).reshape(1, 6)  # Asegurarse de que tenga la forma correcta (1, 6)
    
    prediction = model.predict(features, verbose=0)
    return np.argmax(prediction)  # 0: débil, 1: media, 2: fuerte

def explicar_fortaleza(password):
    """
    Explica las razones de la fortaleza de la contraseña.
    """
    explicaciones = []
    if len(password) >= 12:
        explicaciones.append("✅ Longitud adecuada (más de 12 caracteres)")
    else:
        explicaciones.append("❌ Longitud insuficiente (menos de 12 caracteres)")
    if any(c.isupper() for c in password):
        explicaciones.append("✅ Contiene mayúsculas")
    if any(c.isdigit() for c in password):
        explicaciones.append("✅ Contiene números")
    if any(c in "!@#$%^&*()" for c in password):
        explicaciones.append("✅ Contiene símbolos especiales")
    if password.lower() in ["diego", "juan", "maria", "pedro", "media"]:
        explicaciones.append("❌ Contiene un nombre común")
    if "123" in password or "abc" in password.lower() or "809" in password:
        explicaciones.append("❌ Contiene una secuencia simple")
    return explicaciones

# ========== INTERFAZ PRINCIPAL ==========
def main():
    # Configuración de estilos CSS (sin cambios)
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                        url('https://raw.githubusercontent.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/main/secuencia-vector-diseno-codigo-binario_53876-164420.png');
            background-size: cover;
            background-attachment: fixed;
            animation: fadeIn 1.5s ease-in;
        }}
        
        @keyframes fadeIn {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        
        .stExpander > div {{
            background: rgba(18, 25, 38, 0.95) !important;
            backdrop-filter: blur(12px);
            border-radius: 15px;
            border: 1px solid rgba(0, 168, 255, 0.3);
            transition: all 0.3s ease;
        }}
        
        .stExpander > div:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,150,255,0.2);
        }}
        
        .stButton > button {{
            transition: all 0.3s !important;
            border: 1px solid #00a8ff !important;
        }}
        
        .stButton > button:hover {{
            transform: scale(1.03);
            background: rgba(0,168,255,0.15) !important;
        }}
        
        .chat-message {{
            animation: slideIn 0.4s ease-out;
        }}
        
        @keyframes slideIn {{
            0% {{ transform: translateX(15px); opacity: 0; }}
            100% {{ transform: translateX(0); opacity: 1; }}
        }}
        
        h1, h2, h3 {{
            text-shadow: 0 0 12px rgba(0,168,255,0.5);
        }}
        
        .stProgress > div > div {{
            background: linear-gradient(90deg, #00a8ff, #00ff88);
            border-radius: 3px;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.title("🔐 WildPassPro - Suite de Seguridad")
    
    # Cargar el modelo desde GitHub
    with st.spinner("Cargando el modelo de red neuronal..."):
        model = cargar_modelo_desde_github()
        if model is not None:
            st.success("Modelo cargado exitosamente!")
        else:
            st.error("No se pudo cargar el modelo. Por favor, verifica el archivo.")
            return

    # Interfaz con pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🛠️ Generadores", "🔒 Bóveda", "🔍 Analizador", "💬 Chatbot", "🌐 Escáner Web", "🔐 Verificador de Fugas"])

    # ========== PESTAÑA 3: ANALIZADOR ==========
    with tab3:
        st.subheader("🔍 Analizar Contraseña")
        password = st.text_input("Ingresa tu contraseña:", type="password", key="pwd_input")
        
        if password:
            # Predicción de la red neuronal
            strength_prediction = predecir_fortaleza(model, password)
            strength_labels = ["DÉBIL 🔴", "MEDIA 🟡", "FUERTE 🟢"]
            neural_strength = strength_labels[strength_prediction]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("🧠 Predicción de Red Neuronal")
                st.markdown(f"## {neural_strength}")
                
                if strength_prediction == 2:  # Si es fuerte
                    st.success("### Explicación de la fortaleza:")
                    explicaciones = explicar_fortaleza(password)
                    for explicacion in explicaciones:
                        st.write(explicacion)
                    
            with col2:
                st.subheader("🧠 Análisis de Groq")
                analysis = groq_analysis(password)
                st.markdown(analysis)

if __name__ == "__main__":
    main()
