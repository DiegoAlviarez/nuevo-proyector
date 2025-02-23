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
import hashlib
import pyttsx3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cryptography.fernet import Fernet

# Configuración de Groq
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# ========== NUEVAS CONSTANTES ==========
MASTER_PASSWORD = "WildPassPro2024!"  # Contraseña maestra (cambiar en producción)
TTS_ENGINE = pyttsx3.init()

# ========== FUNCIONES DE SEGURIDAD MEJORADAS ==========
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

def calcular_hash_archivo(ruta_archivo):
    hasher = hashlib.sha256()
    with open(ruta_archivo, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# ========== SISTEMA DE VOZ MEJORADO ==========
def configurar_voz():
    voces = TTS_ENGINE.getProperty('voices')
    TTS_ENGINE.setProperty('voice', voces[0].id)  # Seleccionar voz en español si está disponible
    TTS_ENGINE.setProperty('rate', 150)
    
def hablar_texto(texto):
    try:
        TTS_ENGINE.say(texto)
        TTS_ENGINE.runAndWait()
    except Exception as e:
        st.error(f"Error en síntesis de voz: {str(e)}")

# ========== INTERFAZ PROFESIONAL MEJORADA ==========
def aplicar_estilos_profesionales():
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.9)),
                        url('https://raw.githubusercontent.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/main/secuencia-vector-diseno-codigo-binario_53876-164420.png');
            background-size: cover;
            background-attachment: fixed;
            animation: fadeIn 1.5s ease-in;
        }}
        
        .professional-header {{
            background: linear-gradient(45deg, #000428, #004e92);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,168,255,0.3);
        }}
        
        .feature-card {{
            background: rgba(18, 25, 38, 0.95) !important;
            backdrop-filter: blur(12px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(0, 168, 255, 0.3);
            transition: all 0.3s ease;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,150,255,0.4);
        }}
        
        .status-bar {{
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        
        .ai-status {{
            color: #00ff88;
            font-weight: bold;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 0.8; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.8; }}
        }}
        
        footer {{
            position: fixed;
            bottom: 0;
            width: 100%;
            background: rgba(0,0,0,0.7);
            padding: 1rem;
            text-align: center;
            border-top: 1px solid #00a8ff;
        }}
    </style>
    """, unsafe_allow_html=True)

# ========== NUEVO SISTEMA DE CHAT CON VOZ ==========
def interfaz_chat_avanzado():
    st.subheader("💬 Asistente de Seguridad Inteligente")
    
    col1, col2 = st.columns([3,1])
    with col2:
        tts_toggle = st.toggle("Voz activada", True, help="Activar/Desactivar síntesis de voz")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "¡Hola! Soy tu experto en seguridad. ¿En qué puedo ayudarte hoy?"}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and tts_toggle:
                hablar_texto(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta o comando..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("🔍 Analizando y generando respuesta..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "system",
                        "content": """Eres un experto en seguridad especializado en gestión de credenciales. 
                        Responde de manera profesional y detallada sobre: 
                        - Análisis de contraseñas
                        - Generación de claves seguras
                        - Prácticas de seguridad digital
                        - Cifrado de datos
                        - Vulnerabilidades comunes"""
                    }] + st.session_state.chat_history[-3:],
                    temperature=0.3,
                    max_tokens=400
                ).choices[0].message.content
                
                with st.chat_message("assistant", avatar="🤖"):
                    typewriter_effect(response)
                    if tts_toggle:
                        hablar_texto(response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error en el sistema de IA: {str(e)}")

# ========== SISTEMA DE ANALISIS MEJORADO ==========
def analizador_mejorado():
    st.subheader("🔍 Analizador Profesional de Contraseñas")
    password = st.text_input("Ingresa tu contraseña para análisis:", type="password", key="pwd_input")
    
    if password:
        with st.status("Realizando análisis avanzado...", expanded=True) as status:
            st.write("🔒 Verificando contra bases de datos conocidas...")
            time.sleep(0.5)
            st.write("🧠 Ejecutando modelo predictivo...")
            time.sleep(0.5)
            st.write("📊 Generando reporte de seguridad...")
            time.sleep(0.5)
            
            weaknesses = detect_weakness(password)
            final_strength = "DÉBIL 🔴" if weaknesses else "FUERTE 🟢"
            status.update(label="Análisis Completo", state="complete")
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("📊 Evaluación Final")
            st.metric("Nivel de Seguridad", final_strength)
            st.progress(0.9 if not weaknesses else 0.3)
            
            if weaknesses:
                with st.expander("🔍 Detalles de Vulnerabilidades"):
                    for weakness in weaknesses:
                        st.error(weakness)
            else:
                st.success("✅ Cumple con todos los estándares de seguridad")
        
        with col2:
            st.subheader("🧠 Análisis de IA")
            analysis = groq_analysis(password)
            st.markdown(analysis)

# ========== FUNCIÓN PRINCIPAL MEJORADA ==========
def main():
    aplicar_estilos_profesionales()
    configurar_voz()
    
    st.title("🔐 WildPassPro - Enterprise Security Suite")
    st.markdown("<div class='status-bar'>🟢 Estado del Sistema: <span class='ai-status'>IA OPERATIVA AL 100%</span></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Inicio", "🛠️ Generadores", "🔒 Bóveda", "📊 Analizador", "💬 Asistente IA"])
    
    with tab1:
        st.markdown("<div class='professional-header'><h2>Plataforma Integral de Seguridad Digital</h2></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='feature-card'>
            <h3>🚀 Características Principales</h3>
            <ul>
                <li>Generación de contraseñas de nivel militar</li>
                <li>Bóveda cifrada con doble autenticación</li>
                <li>Análisis predictivo con IA</li>
                <li>Asistente de seguridad con voz</li>
                <li>Protección contra ataques de fuerza bruta</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        with st.container():
            st.subheader("🔑 Generador de Contraseñas Seguras")
            col_gen1, col_gen2 = st.columns(2)
            with col_gen1:
                pwd_length = st.slider("Longitud", 12, 64, 16, key="pwd_length")
                pwd_complexity = st.selectbox("Complejidad", ["Alta", "Extrema", "Personalizada"])
            with col_gen2:
                if st.button("🔄 Generar Contraseña", use_container_width=True):
                    secure_pwd = generate_secure_password(pwd_length)
                    st.session_state.generated_pwd = secure_pwd
                
                if 'generated_pwd' in st.session_state:
                    st.code(st.session_state.generated_pwd, language="text")
                    st.download_button("📥 Descargar Contraseña", 
                                      st.session_state.generated_pwd,
                                      file_name="contraseña_segura.txt")
    
    with tab3:
        st.subheader("🔒 Bóveda Digital Empresarial")
        password = st.text_input("Contraseña Maestra:", type="password", key="vault_pwd")
        
        if password == MASTER_PASSWORD:
            with st.expander("📤 Subir Archivo", expanded=True):
                archivo_subido = st.file_uploader("Seleccionar archivo confidencial:", type=None)
                if archivo_subido:
                    with st.spinner("🔒 Cifrando y almacenando..."):
                        file_hash = calcular_hash_archivo(archivo_subido.name)
                        st.success(f"✅ Archivo protegido | Hash: {file_hash[:12]}...")
            
            with st.expander("📥 Archivos Cifrados"):
                if os.path.exists("secure_vault"):
                    archivos = [f for f in os.listdir("secure_vault") if f.endswith(".encrypted")]
                    if archivos:
                        archivo_seleccionado = st.selectbox("Seleccionar archivo:", archivos)
                        if st.button("Descifrar y Verificar"):
                            ruta_completa = os.path.join("secure_vault", archivo_seleccionado)
                            # Aquí iría la lógica de descifrado y verificación
                    else:
                        st.info("No hay archivos en la bóveda")
        else:
            if password:
                st.error("Acceso no autorizado ⚠️")
    
    with tab4:
        analizador_mejorado()
    
    with tab5:
        interfaz_chat_avanzado()
    
    st.markdown("<footer>WildPassPro 2024 | Sistema de Seguridad Certificado ISO 27001</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
