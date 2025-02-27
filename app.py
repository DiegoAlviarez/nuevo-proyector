import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import openai
import secrets
import string
import os
import time
import json
from sklearn.model_selection import train_test_split
from cryptography.fernet import Fernet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
# Añadir estos imports al inicio
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Configuración de Groq
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# ========== CONSTANTES ==========
MASTER_PASSWORD = "WildPassPro2024!"

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

# ========== FUNCIONES DEL DATASET ==========
def generar_dataset_groq(num_samples=100):
    """Genera dataset de contraseñas usando Groq"""
    if not os.path.exists("password_dataset.csv"):
        dataset = []
        progress_bar = st.progress(0)
        
        for i in range(num_samples):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": "Genera una contraseña segura de 16 caracteres con letras, números y símbolos. Solo responde con la contraseña."
                    }],
                    temperature=0.5,
                    max_tokens=20
                )
                
                password = response.choices[0].message.content.strip()
                password = re.sub(r'[^a-zA-Z0-9!@#$%^&*()]', '', password)
                dataset.append([password, 2])  # 2 = fuerte
                progress_bar.progress((i+1)/num_samples)
                
            except Exception as e:
                st.error(f"Error en muestra {i}: {str(e)}")
                continue
        
        df = pd.DataFrame(dataset, columns=["password", "label"])
        df.to_csv("password_dataset.csv", index=False)
    else:
        df = pd.read_csv("password_dataset.csv")
    return df

def preprocesar_dataset(df):
    """Prepara los datos para el modelo"""
    df["length"] = df["password"].apply(len)
    df["has_upper"] = df["password"].apply(lambda x: int(any(c.isupper() for c in x)))
    df["has_digit"] = df["password"].apply(lambda x: int(any(c.isdigit() for c in x)))
    df["has_symbol"] = df["password"].apply(lambda x: int(any(c in "!@#$%^&*()" for c in x)))
    
    X = df[["length", "has_upper", "has_digit", "has_symbol"]].values
    y = df["label"].values
    
    return X, y, None

# ========== FUNCIONES DE LA RED NEURONAL (MODIFICADAS) ==========
def crear_modelo():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(6,)),
        BatchNormalization(),  # Mejora 1: Batch Normalization
        Dense(16, activation='relu'),
        BatchNormalization(),  # Mejora 1: Batch Normalization
        Dense(3, activation='softmax')
    ])
    # Mejora 4: Métricas adicionales
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def entrenar_modelo(model, X, y):
    # Mejora 3: Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    avg_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    
    for train_idx, val_idx in kf.split(X, y):
        # Mejora 2: Early Stopping
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(f'best_model_fold_{fold_no}.h5', save_best_only=True)
        ]
        
        history = model.fit(
            X[train_idx], y[train_idx],
            epochs=50,
            batch_size=32,
            validation_data=(X[val_idx], y[val_idx]),
            callbacks=callbacks,
            verbose=0
        )
        
        # Acumular métricas
        for metric in avg_metrics:
            avg_metrics[metric] += history.history[metric][-1]
        
        fold_no += 1
    
    # Calcular promedios
    for metric in avg_metrics:
        avg_metrics[metric] /= kf.n_splits
    
    # Guardar modelo final
    model.save("password_strength_model.h5")
    
    return model, avg_metrics

#3. Mejora en la Predicción (🎯 predecir_fortaleza())
def predecir_fortaleza(model, password):
    try:
        features = np.array([
            len(password),
            int(any(c.isupper() for c in password)),
            int(any(c.isdigit() for c in password)),
            int(any(c in "!@#$%^&*()" for c in password))
        ]).reshape(1, -1).astype(np.float32)
        
        prediction = model.predict(features, verbose=0)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return 0  # Valor seguro por defecto

def explicar_fortaleza(password):
    explicaciones = []
    if len(password) >= 12:
        explicaciones.append("✅ Longitud adecuada (12+ caracteres)")
    if any(c.isupper() for c in password):
        explicaciones.append("✅ Contiene mayúsculas")
    if any(c.isdigit() for c in password):
        explicaciones.append("✅ Contiene números")
    if any(c in "!@#$%^&*()" for c in password):
        explicaciones.append("✅ Contiene símbolos")
    return explicaciones

#Validación Cruzada: Agrega esta función para evaluar robustez:
def validacion_cruzada(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits)
    accuracies = []
    
    for train_idx, val_idx in kf.split(X):
        model = crear_modelo()
        model.fit(X[train_idx], y[train_idx], 
                 epochs=30, verbose=0)
        _, acc = model.evaluate(X[val_idx], y[val_idx], verbose=0)
        accuracies.append(acc)
    
    st.write(f"Precisión promedio en CV: {np.mean(accuracies):.2%}")

# ========== ACTUALIZAR INTERFAZ PRINCIPAL ==========
def main():
    # ... (código existente)
# ========== INTERFAZ PRINCIPAL ==========
#Monitorización en Tiempo Real:    
# En la interfaz principal
    with st.expander("📈 Rendimiento del Modelo"):
        if os.path.exists('training_metrics.json'):
            metrics = json.load(open('training_metrics.json'))
            st.line_chart(pd.DataFrame(metrics))

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
        
        .stButton > button {{
            transition: all 0.3s !important;
            border: 1px solid #00a8ff !important;
        }}
        
        .chat-message {{
            animation: slideIn 0.4s ease-out;
        }}
        h1, h2, h3 {{ text-shadow: 0 0 12px rgba(0,168,255,0.5); }}
    </style>
    """, unsafe_allow_html=True)

    st.title("🔐 WildPassPro - Suite de Seguridad")
    
    # Dataset y Modelo
    if not os.path.exists("password_dataset.csv"):
        with st.spinner("Generando dataset inicial (2-3 mins)..."):
            df = generar_dataset_groq(50)
    else:
        df = pd.read_csv("password_dataset.csv")
    
    X, y, _ = preprocesar_dataset(df)
    
 # Dentro de la sección de entrenamiento del modelo
    if not os.path.exists("password_strength_model.h5"):
        with st.spinner("Entrenando la red neuronal con validación cruzada..."):
            model = crear_modelo()
            model, cv_metrics = entrenar_modelo(model, X, y)
            
            # Mostrar métricas de CV
            with st.expander("📊 Métricas de Validación Cruzada"):
                st.metric("Precisión Promedio", f"{cv_metrics['accuracy']*100:.2f}%")
                st.metric("Precisión", f"{cv_metrics['precision']*100:.2f}%")
                st.metric("Recall", f"{cv_metrics['recall']*100:.2f}%")
                
                # Gráfico de métricas
                fig, ax = plt.subplots()
                ax.bar(['Precisión', 'Recall', 'Exactitud'], 
                      [cv_metrics['precision'], cv_metrics['recall'], cv_metrics['accuracy']])
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
            st.success("Modelo entrenado exitosamente!")

    # Interfaz con pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Generadores", "🔒 Bóveda", "🔍 Analizador", "💬 Chatbot"])
    
    with tab1:
        st.subheader("🔑 Generar Contraseña")
        pwd_length = st.slider("Longitud", 12, 32, 16)
        if st.button("Generar Contraseña"):
            secure_pwd = generate_secure_password(pwd_length)
            st.code(secure_pwd)
            st.download_button("📥 Descargar", secure_pwd, "contraseña_segura.txt")
        
        st.subheader("🔑 Generar Llave de Acceso")
        if st.button("Generar Llave"):
            access_key = generate_access_key()
            st.code(access_key)
            st.download_button("📥 Descargar", access_key, "llave_acceso.txt")

    with tab2:
        st.subheader("🔒 Bóveda Segura")
        password = st.text_input("Contraseña maestra:", type="password")
        
        if password == MASTER_PASSWORD:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📤 Subir Archivo")
                archivo = st.file_uploader("Seleccionar archivo:")
                if archivo:
                    ruta = os.path.join("secure_vault", archivo.name)
                    with open(ruta, "wb") as f:
                        f.write(archivo.getbuffer())
                    with st.spinner("Cifrando..."):
                        cifrar_archivo(ruta)
                        st.success("Archivo protegido")
            with col2:
                st.subheader("📥 Descargar Archivo")
                if os.path.exists("secure_vault"):
                    archivos = [f for f in os.listdir("secure_vault") if f.endswith(".encrypted")]
                    if archivos:
                        seleccion = st.selectbox("Archivos cifrados:", archivos)
                        if st.button("Descifrar"):
                            ruta = os.path.join("secure_vault", seleccion)
                            descifrar_archivo(ruta)
                            st.success("Archivo listo para descargar")
        elif password:
            st.error("Contraseña incorrecta")

    with tab3:
        st.subheader("🔍 Analizar Contraseña")
        password = st.text_input("Ingresa tu contraseña:", type="password")
        
        if password:
            debilidades = detect_weakness(password)
            prediccion = predecir_fortaleza(model, password)
            fuerza = ["DÉBIL 🔴", "MEDIA 🟡", "FUERTE 🟢"][prediccion]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"## {fuerza}")
                if debilidades:
                    st.error("Debilidades encontradas:")
                    for d in debilidades:
                        st.write(d)
                else:
                    st.success("Cumple todos los criterios")
                
                if prediccion == 2:
                    st.success("Razones de fortaleza:")
                    for razon in explicar_fortaleza(password):
                        st.write(razon)
            
            with col2:
                st.markdown(groq_analysis(password))

    with tab4:
        st.subheader("💬 Asistente de Seguridad")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Escribe tu pregunta:"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.spinner("Analizando..."):
                try:
                    respuesta = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{
                            "role": "system",
                            "content": "Eres un experto en seguridad especializado en gestión de credenciales."
                        }] + st.session_state.chat_history[-3:],
                        temperature=0.3,
                        max_tokens=300
                    ).choices[0].message.content
                    
                    with st.chat_message("assistant"):
                        typewriter_effect(respuesta)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
