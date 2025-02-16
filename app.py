import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import openai
import joblib
import tensorflow as tf
import secrets
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuración de Groq
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# ========== FUNCIONES NUEVAS (GENERADORES) ==========
def generate_secure_password(length=16):
    characters = string.ascii_letters + string.digits + "!@#$%^&*()"
    return ''.join(secrets.choice(characters) for _ in range(length))

def generate_access_key():
    return secrets.token_urlsafe(32)

# ========== FUNCIONES EXISTENTES ==========
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
        
    return weaknesses

def groq_analysis(password):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""Analiza esta contraseña: '{password}'
                1. Vulnerabilidades críticas
                2. Comparación con patrones comunes
                3. Recomendaciones personalizadas
                Formato: Lista markdown con emojis"""
            }],
            temperature=0.4,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Error:** {str(e)}"

# ========== INTERFAZ PRINCIPAL ==========
def main():
    st.title("🔐 WildPassPro - Suite de Seguridad")
    
    # Sección de generadores
    with st.expander("🛠️ Generadores Seguros", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔑 Generar Contraseña")
            pwd_length = st.slider("Longitud", 12, 32, 16)
            if st.button("Generar Contraseña"):
                secure_pwd = generate_secure_password(pwd_length)
                st.code(secure_pwd, language="text")
                
        with col2:
            st.subheader("🔑 Generar Llave de Acceso")
            if st.button("Generar Llave"):
                access_key = generate_access_key()
                st.code(access_key, language="text")

    # Sección de análisis de contraseñas
    with st.expander("🔍 Analizar Contraseña", expanded=True):
        password = st.text_input("Ingresa tu contraseña:", type="password", key="pwd_input")
        
        if password:
            weaknesses = detect_weakness(password)
            final_strength = "DÉBIL 🔴" if weaknesses else "FUERTE 🟢"
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📋 Clasificación Final")
                st.markdown(f"## {final_strength}")
                if weaknesses:
                    st.error("### Razones de debilidad:")
                    for weakness in weaknesses:
                        st.write(weakness)
                else:
                    st.success("### Cumple con todos los criterios")
                    
            with col2:
                st.subheader("🧠 Análisis de Groq")
                analysis = groq_analysis(password)
                st.markdown(analysis)

    # Sección de Chatbot (existente)
    st.divider()
    st.subheader("💬 Asistente de Seguridad")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "¡Hola! Soy tu experto en seguridad. Pregúntame sobre:"}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analizando..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "system",
                        "content": "Eres un experto en seguridad especializado en gestión de credenciales. Responde solo sobre: contraseñas, llaves de acceso, 2FA, y mejores prácticas."
                    }] + st.session_state.chat_history[-3:],
                    temperature=0.3,
                    max_tokens=300
                ).choices[0].message.content
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error en el chatbot: {str(e)}")

if __name__ == "__main__":
    main()
