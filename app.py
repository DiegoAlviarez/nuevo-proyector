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
import os
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cryptography.fernet import Fernet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# ========== FUNCIONES DE LA RED NEURONAL ==========
def generar_dataset_groq(num_samples=1000):
    dataset = []
    for _ in range(num_samples):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": """Genera una contraseña y clasifícala como 'débil', 'media' o 'fuerte'.
                Devuelve el resultado en formato JSON: {"password": "contraseña", "label": "clasificación"}"""
            }],
            temperature=0.5,
            max_tokens=100
        )
        try:
            result = response.choices[0].message.content
            result_json = eval(result)  # Convertir a diccionario
            dataset.append([result_json["password"], result_json["label"]])
        except Exception as e:
            print(f"Error al procesar la respuesta de Groq: {e}")
    
    df = pd.DataFrame(dataset, columns=["password", "label"])
    df.to_csv("password_dataset.csv", index=False)
    return df

def preprocesar_dataset(df):
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])  # 0: débil, 1: media, 2: fuerte
    
    df["length"] = df["password"].apply(len)
    df["has_upper"] = df["password"].apply(lambda x: int(any(c.isupper() for c in x)))
    df["has_digit"] = df["password"].apply(lambda x: int(any(c.isdigit() for c in x)))
    df["has_symbol"] = df["password"].apply(lambda x: int(any(c in "!@#$%^&*()" for c in x)))
    
    X = df[["length", "has_upper", "has_digit", "has_symbol"]].values
    y = df["label"].values
    
    return X, y, label_encoder

def crear_modelo():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def entrenar_modelo(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    model.save("password_strength_model.h5")
    return model

def predecir_fortaleza(model, password):
    features = np.array([
        len(password),
        int(any(c.isupper() for c in password)),
        int(any(c.isdigit() for c in password)),
        int(any(c in "!@#$%^&*()" for c in password))
    ]).reshape(1, -1)
    prediction = model.predict(features)
    return np.argmax(prediction)  # 0: débil, 1: media, 2: fuerte

# ========== INTERFAZ PRINCIPAL ==========
def main():
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
    
    # Verificar si el dataset ya existe
    if not os.path.exists("password_dataset.csv"):
        st.info("Generando dataset con Groq...")
        df = generar_dataset_groq(num_samples=1000)
    else:
        df = pd.read_csv("password_dataset.csv")

    # Preprocesar el dataset
    X, y, label_encoder = preprocesar_dataset(df)

    # Verificar si el modelo ya está entrenado
    if not os.path.exists("password_strength_model.h5"):
        st.info("Entrenando la red neuronal...")
        model = crear_modelo()
        model = entrenar_modelo(model, X, y)
    else:
        model = tf.keras.models.load_model("password_strength_model.h5")

    # Interfaz con pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Generadores", "🔒 Bóveda", "🔍 Analizador", "💬 Chatbot"])
    
    with tab1:
        st.subheader("🔑 Generar Contraseña")
        pwd_length = st.slider("Longitud", 12, 32, 16, key="pwd_length")
        if st.button("Generar Contraseña", key="gen_pwd"):
            secure_pwd = generate_secure_password(pwd_length)
            st.code(secure_pwd, language="text")
            st.download_button(
                label="📥 Descargar Contraseña",
                data=secure_pwd,
                file_name="contraseña_segura.txt",
                mime="text/plain"
            )
        
        st.subheader("🔑 Generar Llave de Acceso")
        if st.button("Generar Llave", key="gen_key"):
            access_key = generate_access_key()
            st.code(access_key, language="text")
            st.download_button(
                label="📥 Descargar Llave",
                data=access_key,
                file_name="llave_acceso.txt",
                mime="text/plain"
            )

    with tab2:
        st.subheader("🔒 Bóveda Segura - Acceso Protegido")
        password = st.text_input("Ingresa la contraseña maestra:", type="password", key="vault_pwd")
        
        if password == MASTER_PASSWORD:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Subir Archivo Cifrado")
                archivo_subido = st.file_uploader("Selecciona un archivo:", type=None, key="uploader")
                if archivo_subido:
                    if not os.path.exists("secure_vault"):
                        os.makedirs("secure_vault")
                    ruta_temporal = os.path.join("secure_vault", archivo_subido.name)
                    with open(ruta_temporal, "wb") as f:
                        f.write(archivo_subido.getbuffer())
                    with st.spinner("🔒 Cifrando y guardando archivo..."):
                        time.sleep(1)  # Simula proceso de cifrado
                        ruta_cifrado = cifrar_archivo(ruta_temporal)
                        st.success(f"✅ Archivo protegido: {os.path.basename(ruta_cifrado)}")
            
            with col2:
                st.subheader("📥 Archivos Cifrados")
                if os.path.exists("secure_vault"):
                    archivos = [f for f in os.listdir("secure_vault") if f.endswith(".encrypted")]
                    if archivos:
                        archivo_seleccionado = st.selectbox("Selecciona un archivo:", archivos)
                        if st.button("Descifrar y Descargar"):
                            ruta_completa = os.path.join("secure_vault", archivo_seleccionado)
                            ruta_descifrado = descifrar_archivo(ruta_completa)
                            with open(ruta_descifrado, "rb") as f:
                                datos = f.read()
                            st.download_button(
                                label="Descargar Archivo Descifrado",
                                data=datos,
                                file_name=os.path.basename(ruta_descifrado),
                                mime="application/octet-stream"
                            )
                            os.remove(ruta_descifrado)
                    else:
                        st.info("No hay archivos cifrados en la bóveda")
        else:
            if password:
                st.error("Contraseña incorrecta ⚠️")

    with tab3:
        st.subheader("🔍 Analizar Contraseña")
        password = st.text_input("Ingresa tu contraseña:", type="password", key="pwd_input")
        
        if password:
            weaknesses = detect_weakness(password)
            final_strength = "DÉBIL 🔴" if weaknesses else "FUERTE 🟢"
            
            # Predicción de la red neuronal
            strength_prediction = predecir_fortaleza(model, password)
            strength_labels = ["DÉBIL 🔴", "MEDIA 🟡", "FUERTE 🟢"]
            neural_strength = strength_labels[strength_prediction]
            
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
                
                st.subheader("🧠 Predicción de Red Neuronal")
                st.markdown(f"## {neural_strength}")
                    
            with col2:
                st.subheader("🧠 Análisis de Groq")
                analysis = groq_analysis(password)
                st.markdown(analysis)

    with tab4:
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
                    
                    # Efecto máquina de escribir
                    with st.chat_message("assistant"):
                        typewriter_effect(response)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error en el chatbot: {str(e)}")

if __name__ == "__main__":
    main()
