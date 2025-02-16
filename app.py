import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import openai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib

# 1. Configuración de Groq
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# 2. Funciones de seguridad
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

# 3. Chatbot especializado en contraseñas
def password_security_chatbot(messages):
    system_prompt = """Eres un experto en seguridad de contraseñas. Solo responde preguntas sobre:
    - Creación de contraseñas seguras
    - Almacenamiento seguro de contraseñas
    - Detección de contraseñas débiles
    - Mejores prácticas de seguridad
    - Recuperación de contraseñas
    - Gestión de credenciales

    No respondas preguntas fuera de este tema. Si te preguntan algo no relacionado, di:
    'Soy un especialista en seguridad de contraseñas. ¿En qué puedo ayudarte sobre este tema?'"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# 4. Interfaz de Streamlit
def main():
    st.title("🔒 WildPassPro - Seguridad de Contraseñas")
    
    # Historial del chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "¡Hola! Soy tu experto en seguridad de contraseñas. ¿En qué puedo ayudarte?"}]

    # Sección de análisis de contraseñas
    with st.expander("🔑 Analizar Fortaleza de Contraseña"):
        rockyou_url = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/rockyou.txt"
        
        if st.button("🔄 Cargar Dataset"):
            with st.spinner("Procesando 14 millones de contraseñas..."):
                df = download_and_clean_rockyou(rockyou_url)
                X, y, tokenizer, le = preprocess_data(df)
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
                
                model = build_model(len(tokenizer.word_index) + 1)
                model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
                model.save("password_model.h5")
                joblib.dump(tokenizer, "tokenizer.pkl")
                st.success("Modelo listo para analizar!")

        user_password = st.text_input("Ingresa una contraseña para analizar:")
        if user_password:
            try:
                model = tf.keras.models.load_model("password_model.h5")
                tokenizer = joblib.load("tokenizer.pkl")
                
                sequence = tokenizer.texts_to_sequences([user_password])
                padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=20)
                prediction = model.predict(padded, verbose=0)[0][0]
                
                strength = "DÉBIL 🔴" if prediction > 0.5 else "FUERTE 🟢"
                st.subheader(f"Resultado: {strength}")
                st.progress(prediction if strength == "DÉBIL 🔴" else 1 - prediction)
                
            except Exception as e:
                st.error("Primero carga el modelo con el botón 'Cargar Dataset'")

    # Sección del chatbot
    st.divider()
    st.subheader("💬 Chat de Seguridad")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta sobre contraseñas..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analizando..."):
            response = password_security_chatbot(st.session_state.chat_history[-5:])  # Usar último contexto
            
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
