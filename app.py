import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import openai
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuración de Groq
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Cargar contraseñas débiles
def load_weak_passwords(url):
    response = requests.get(url)
    return set(line.strip().lower() for line in response.text.splitlines() if line.strip())

WEAK_PASSWORDS = load_weak_passwords("https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/rockyou.txt")

# Funciones clave mejoradas
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
                "content": f"""Analiza esta contraseña como experto en seguridad: '{password}'
                - Detalla todas las vulnerabilidades encontradas
                - Comparación con bases de datos de leaks
                - Recomendaciones específicas
                Formato: Lista con emojis en markdown"""
            }],
            temperature=0.4,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Error:** {str(e)}"

# Interfaz principal actualizada
def main():
    st.title("🔐 WildPassPro - Analizador Profesional")
    
    with st.expander("🔑 Analizar Contraseña", expanded=True):
        password = st.text_input("Ingresa tu contraseña:", type="password", key="pwd_input")
        
        if password:
            # Detección de debilidades
            weaknesses = detect_weakness(password)
            
            # Clasificación definitiva
            final_strength = "DÉBIL 🔴" if weaknesses else "FUERTE 🟢"
            
            # Mostrar resultados
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📋 Clasificación Final")
                st.markdown(f"## {final_strength}")
                if weaknesses:
                    st.error("### Razones de debilidad:")
                    for weakness in weaknesses:
                        st.write(weakness)
                else:
                    st.success("### Cumple con todos los criterios de seguridad")
                    
            with col2:
                st.subheader("🧠 Análisis Detallado de Groq")
                analysis = groq_analysis(password)
                st.markdown(analysis)

if __name__ == "__main__":
    main()
