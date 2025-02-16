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

# Funciones clave
def download_and_clean_rockyou(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error descargando archivo: {response.status_code}")
    
    lines = response.text.splitlines()
    cleaned_data = [re.sub(r"[^a-zA-Z0-9]", "", line.strip()).lower() for line in lines if line.strip()]
    return pd.DataFrame({"password": cleaned_data, "label": "weak"})

def preprocess_data(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df["password"])
    sequences = tokenizer.texts_to_sequences(df["password"])
    
    X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)
    y = df["label"].values
    
    return X, y, tokenizer, le

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim, 64, input_length=20),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def full_analysis(password):
    # ========== ANÁLISIS VISUAL ==========
    st.subheader("📊 Criterios de Seguridad")
    
    criteria = {
        "length": len(password) >= 12,
        "upper": any(c.isupper() for c in password),
        "lower": any(c.islower() for c in password),
        "digit": any(c.isdigit() for c in password),
        "special": any(c in "!@#$%^&*()" for c in password)
    }
    
    chart_data = pd.DataFrame({
        "Criterio": ["Longitud (12+)", "Mayúsculas", "Minúsculas", "Números", "Símbolos"],
        "Cumple": list(criteria.values())
    })
    
    st.bar_chart(chart_data, x="Criterio", color="#FF4B4B", height=300)

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
