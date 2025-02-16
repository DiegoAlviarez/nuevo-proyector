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

# Cargar rockyou.txt como conjunto de contraseñas débiles
def load_weak_passwords(url):
    response = requests.get(url)
    weak_passwords = set(line.strip().lower() for line in response.text.splitlines() if line.strip())
    return weak_passwords

WEAK_PASSWORDS = load_weak_passwords("https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/rockyou.txt")

# Funciones clave modificadas
def is_weak_password(password):
    # Reglas básicas de debilidad
    rules = {
        "all_lower": password.islower(),
        "all_upper": password.isupper(),
        "no_numbers": not any(c.isdigit() for c in password),
        "no_symbols": not any(c in "!@#$%^&*()" for c in password),
        "in_rockyou": password.lower() in WEAK_PASSWORDS
    }
    return any(rules.values())

def full_analysis(password):
    # ========== DETECCIÓN DE DEBILIDAD ==========
    weak_reasons = []
    if password.lower() in WEAK_PASSWORDS:
        weak_reasons.append("Está en la lista rockyou.txt")
    if password.islower():
        weak_reasons.append("Solo minúsculas")
    if not any(c.isdigit() for c in password):
        weak_reasons.append("Sin números")
    if not any(c in "!@#$%^&*()" for c in password):
        weak_reasons.append("Sin símbolos")
    
    # ========== ANÁLISIS VISUAL ==========
    st.subheader("📊 Detección de Debilidad")
    if weak_reasons:
        st.error("**Contraseña Débil** - Razones:")
        for reason in weak_reasons:
            st.write(f"❌ {reason}")
        return "DÉBIL 🔴"
    else:
        st.success("**Contraseña Fuerte** - Cumple con los criterios básicos")
        return "FUERTE 🟢"

    # ========== ANÁLISIS GROQ ==========
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""Analiza esta contraseña como experto: '{password}'
                - Longitud: {len(password)}/12 caracteres
                - Complejidad de caracteres
                - Patrones detectados
                - Comparación con bases de datos de leaks
                Devuelve el análisis en markdown con emojis."""
            }],
            temperature=0.4,
            max_tokens=400
        )
        st.subheader("🧠 Evaluación de Groq")
        st.markdown(response.choices[0].message.content)
        
    except Exception as e:
        st.error(f"Error en análisis Groq: {str(e)}")

# Interfaz principal
def main():
    st.title("🔐 WildPassPro - Analizador Profesional")
    
    # Sección de análisis
    with st.expander("🔑 Analizar Contraseña", expanded=True):
        rockyou_url = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/rockyou.txt"
        
        if st.button("🔄 Cargar Modelo de Seguridad", type="primary"):
            with st.spinner("Procesando 14M de contraseñas..."):
                try:
                    df = download_and_clean_rockyou(rockyou_url)
                    X, y, tokenizer, le = preprocess_data(df)
                    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
                    
                    model = build_model(len(tokenizer.word_index) + 1)
                    model.fit(X_train, y_train, epochs=2, batch_size=128, verbose=0)
                    model.save("password_model.h5")
                    joblib.dump(tokenizer, "tokenizer.pkl")
                    st.success("¡Modelo cargado correctamente!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        password = st.text_input("Ingresa tu contraseña:", type="password", key="pwd_input")
        
        if password:
            try:
                model = tf.keras.models.load_model("password_model.h5")
                tokenizer = joblib.load("tokenizer.pkl")
                
                sequence = tokenizer.texts_to_sequences([password])
                padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=20)
                prediction = model.predict(padded, verbose=0)[0][0]
                
                # Resultado corregido (umbral ajustado)
                strength = "DÉBIL 🔴" if prediction > 0.65 else "MEDIA 🟡" if prediction > 0.35 else "FUERTE 🟢"
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("🤖 Modelo RockYou")
                    st.metric("Resultado", strength)
                    st.progress(prediction if strength == "DÉBIL 🔴" else 1 - prediction)
                    
                with col2:
                    full_analysis(password)
                    
            except Exception as e:
                st.error("Primero carga el modelo con el botón superior")

    # ========== CHATBOT RESTAURADO ==========
    st.divider()
    st.subheader("💬 Asistente de Seguridad")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "¡Hola! Soy tu experto en seguridad. Pregúntame sobre contraseñas seguras."}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta sobre contraseñas..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analizando..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "system",
                        "content": "Eres un experto en seguridad especializado en contraseñas. Responde solo sobre: creación segura, almacenamiento, recuperación y mejores prácticas. Si la pregunta es off-topic, responde: 'Soy especialista en contraseñas, ¿en qué más puedo ayudarte?'"
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
