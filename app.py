import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
import io

# 1. Función para descargar y limpiar el dataset rockyou.txt
def download_and_clean_rockyou(url):
    # Descargar el archivo desde GitHub
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"No se pudo descargar el archivo. Código de estado: {response.status_code}")
    
    # Leer el contenido del archivo
    lines = response.text.splitlines()
    
    # Limpiar y normalizar
    cleaned_data = []
    for line in lines:
        line = line.strip()  # Eliminar espacios en blanco
        if line:  # Ignorar líneas vacías
            # Normalizar: eliminar caracteres no alfanuméricos y convertir a minúsculas
            normalized_line = re.sub(r"[^a-zA-Z0-9]", "", line).lower()
            if normalized_line:  # Asegurarse de que no esté vacía después de la normalización
                cleaned_data.append(normalized_line)
    
    # Convertir a DataFrame
    df = pd.DataFrame(cleaned_data, columns=["password"])
    df["label"] = "weak"  # Etiquetar todas las contraseñas como "débiles" (puedes ajustar esto)
    
    return df

# 2. Preprocesamiento de datos
def preprocess_data(df):
    # Codificación de etiquetas
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    
    # Tokenización
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df["password"])
    sequences = tokenizer.texts_to_sequences(df["password"])
    
    # Padding
    X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)
    y = df["label"].values
    
    return X, y, tokenizer, le

# 3. Modelo de Deep Learning
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=20),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# 4. Entrenamiento y guardado del modelo
def train_and_save_model(X_train, y_train):
    model = build_model(input_dim=len(tokenizer.word_index) + 1)
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)  # Reducir epochs para mayor velocidad
    return history

# 5. Interfaz de Streamlit
def main():
    st.title("🔒 WildPassPro - Análisis de Fortaleza de Contraseñas")
    
    # URL del archivo rockyou.txt en GitHub
    rockyou_url = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/rockyou.txt"
    
    # Cargar y limpiar el dataset
    if st.button("🔄 Cargar y Limpiar Dataset"):
        with st.spinner("Cargando y limpiando rockyou.txt..."):
            df = download_and_clean_rockyou(rockyou_url)
            st.success(f"Dataset cargado y limpiado. Total de contraseñas: {len(df)}")
            st.write(df.head())
        
        # Preprocesar datos
        X, y, tokenizer, le = preprocess_data(df)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        with st.spinner("Entrenando modelo..."):
            history = train_and_save_model(X_train, y_train)
            st.success("Modelo entrenado.")
            st.line_chart(pd.DataFrame(history.history))  # Gráfico de entrenamiento
    
    # Interfaz de predicción
    user_input = st.text_input("Ingrese una contraseña para evaluar su fortaleza:")
    
    if user_input:
        # Cargar modelo y componentes
        model = tf.keras.models.load_model("password_strength_model.h5")
        tokenizer = joblib.load("tokenizer.pkl")
        le = joblib.load("label_encoder.pkl")
        
        # Preprocesar entrada
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=20)
        
        # Predecir
        prediction = model.predict(padded)[0][0]
        label = "débil" if prediction > 0.5 else "fuerte"
        confidence = prediction if label == "débil" else 1 - prediction
        
        # Mostrar resultados
        st.subheader(f"Clasificación: {label.upper()} (Confianza: {confidence:.2%})")
        st.write("Recomendaciones:")
        if label == "débil":
            st.markdown("1. Usa una combinación de letras, números y símbolos.")
            st.markdown("2. Evita contraseñas comunes o fáciles de adivinar.")
            st.markdown("3. Considera usar una frase larga como contraseña.")
        else:
            st.markdown("¡Buena contraseña! Sigue así.")

if __name__ == "__main__":
    main()
