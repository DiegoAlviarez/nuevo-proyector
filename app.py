# app.py
import streamlit as st
import joblib
import numpy as np
import string
import random
import time
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from groq import Groq
from typing import Generator

# Configuración inicial
SECURE_FOLDER = "secure_vault"
os.makedirs(SECURE_FOLDER, exist_ok=True)

# Configuración de Groq
groq_client = Groq(api_key=st.secrets["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"])
MODELOS_IA = ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768']

# Animaciones y estilos
st.set_page_config(
    page_title="WildPass Pro",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
@keyframes gradient {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

@keyframes float {{
    0% {{transform: translateY(0px);}}
    50% {{transform: translateY(-20px);}}
    100% {{transform: translateY(0px);}}
}}

.stApp {{
    background: linear-gradient(-45deg, #1a1a1a, #2a2a2a, #3a3a3a, #4a4a4a);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}}

.main {{
    background-color: rgba(0, 0, 0, 0.8) !important;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 0 30px rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    animation: slideIn 1s ease-out;
}}

@keyframes slideIn {{
    from {{transform: translateY(50px); opacity: 0;}}
    to {{transform: translateY(0); opacity: 1;}}
}}

h1, h2, h3 {{
    color: #00ff9d !important;
    text-shadow: 0 0 10px rgba(0,255,157,0.5);
}}

.stButton>button {{
    background: linear-gradient(45deg, #00ff9d, #00b8ff);
    border: none;
    color: black !important;
    border-radius: 25px;
    transition: all 0.3s ease;
    animation: float 3s ease-in-out infinite;
}}

.stButton>button:hover {{
    transform: scale(1.1);
    box-shadow: 0 0 20px rgba(0,255,157,0.5);
}}

.chat-container {{
    max-height: 500px;
    overflow-y: auto;
    padding: 20px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #00ff9d;
}}

.user-message {{
    background: rgba(0, 255, 157, 0.1);
    padding: 10px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 70%;
    float: right;
    clear: both;
    border: 1px solid #00ff9d;
    animation: messageIn 0.5s ease-out;
}}

.bot-message {{
    background: rgba(0, 184, 255, 0.1);
    padding: 10px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 70%;
    float: left;
    clear: both;
    border: 1px solid #00b8ff;
    animation: messageIn 0.5s ease-out;
}}

@keyframes messageIn {{
    from {{transform: translateX(100px); opacity: 0;}}
    to {{transform: translateX(0); opacity: 1;}}
}}

.sidebar .sidebar-content {{
    background-color: rgba(0, 0, 0, 0.8) !important;
    backdrop-filter: blur(5px);
    border-right: 1px solid #00ff9d;
}}

#particles {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}}
</style>

<div id="particles"></div>

<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS('particles', {{
  "particles": {{
    "number": {{ "value": 80 }},
    "color": {{ "value": "#00ff9d" }},
    "shape": {{ "type": "circle" }},
    "opacity": {{ "value": 0.5 }},
    "size": {{ "value": 3 }},
    "move": {{
      "enable": true,
      "speed": 2,
      "direction": "none",
      "random": false,
      "straight": false,
      "out_mode": "out",
      "bounce": false
    }}
  }},
  "interactivity": {{
    "events": {{
      "onhover": {{ "enable": true, "mode": "repulse" }}
    }}
  }}
}});
</script>
""", unsafe_allow_html=True)

# Clases principales
class SecureVault:
    @staticmethod
    def generate_key(password: str, salt: bytes = None) -> bytes:
        salt = salt or os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode())), salt

    @staticmethod
    def encrypt_file(file_path: str, password: str):
        salt = os.urandom(16)
        key, _ = SecureVault.generate_key(password, salt)
        fernet = Fernet(key)
        
        with open(file_path, "rb") as file:
            original = file.read()
        
        encrypted = fernet.encrypt(original)
        
        with open(file_path, "wb") as file:
            file.write(salt + encrypted)

    @staticmethod
    def decrypt_file(file_path: str, password: str):
        with open(file_path, "rb") as file:
            data = file.read()
        
        salt = data[:16]
        encrypted = data[16:]
        
        try:
            key, _ = SecureVault.generate_key(password, salt)
            fernet = Fernet(key)
            return fernet.decrypt(encrypted)
        except:
            return None

class PasswordGenerator:
    @staticmethod
    def generate_weak_password():
        patterns = [
            lambda: ''.join(random.choice(string.ascii_lowercase) for _ in range(8)),
            lambda: ''.join(random.choice(["123456", "password", "qwerty", "admin"])),
            lambda: ''.join(random.choice(string.digits) for _ in range(6))
        ]
        return random.choice(patterns)()
    
    @staticmethod
    def generate_strong_password(length=16, use_symbols=True):
        chars = string.ascii_letters + string.digits
        if use_symbols:
            chars += string.punctuation
        return ''.join(random.SystemRandom().choice(chars) for _ in range(length))
    
    @staticmethod
    def generate_pin(length=6):
        return ''.join(random.SystemRandom().choice(string.digits) for _ in range(length))
    
    @staticmethod
    def generate_access_key():
        chars = string.ascii_letters + string.digits + "-_"
        return ''.join(random.SystemRandom().choice(chars) for _ in range(24))

class SecurityAnalyzer:
    @staticmethod
    def extract_features(password):
        return [
            len(password),
            sum(c.isupper() for c in password),
            sum(c.isdigit() for c in password),
            sum(c in string.punctuation for c in password),
            len(set(password))/max(len(password), 1)
        ]

class AISecurityAssistant:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load("local_pass_model.pkl")
            st.success("✅ Modelo de seguridad cargado!")
        except:
            st.warning("⚠ Modelo no encontrado, entrénalo primero")
            self.model = None
    
    def generate_training_data(self, samples=1000):
        X, y = [], []
        generator = PasswordGenerator()
        for _ in range(samples//2):
            X.append(SecurityAnalyzer.extract_features(generator.generate_weak_password()))
            y.append(0)
            X.append(SecurityAnalyzer.extract_features(generator.generate_strong_password()))
            y.append(1)
        return np.array(X), np.array(y)
    
    def train_model(self):
        try:
            X, y = self.generate_training_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            self.model = RandomForestClassifier(n_estimators=100)
            training_history = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(1, 101):
                self.model.fit(X_train, y_train)
                acc = self.model.score(X_test, y_test)
                training_history.append(acc)
                
                progress_bar.progress(epoch/100)
                status_text.text(f"Época: {epoch} - Precisión: {acc:.2%}")
                time.sleep(0.05)
            
            joblib.dump(self.model, "local_pass_model.pkl")
            st.success(f"🎉 Modelo entrenado! Precisión: {acc:.2%}")
            return True
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False

# Funciones principales
def secure_folder_section():
    st.subheader("🔐 Carpeta Segura")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Subir Archivos")
        uploaded_file = st.file_uploader("Selecciona archivo a proteger:", 
                                       type=["txt", "pdf", "png", "jpg", "docx", "xlsx"])
        access_key = st.text_input("Crea una llave de acceso:", type="password")
        
        if uploaded_file and access_key:
            if len(access_key) < 12:
                st.error("La llave debe tener al menos 12 caracteres")
            else:
                file_path = os.path.join(SECURE_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                SecureVault.encrypt_file(file_path, access_key)
                st.success("Archivo protegido con éxito!")

    with col2:
        st.markdown("### Acceder a Archivos")
        access_key_download = st.text_input("Ingresa tu llave de acceso:", type="password")
        
        if access_key_download:
            files = [f for f in os.listdir(SECURE_FOLDER) 
                    if os.path.isfile(os.path.join(SECURE_FOLDER, f))]
            
            if files:
                selected_file = st.selectbox("Archivos disponibles:", files)
                
                if selected_file:
                    file_path = os.path.join(SECURE_FOLDER, selected_file)
                    decrypted = SecureVault.decrypt_file(file_path, access_key_download)
                    
                    if decrypted:
                        with open(file_path, "wb") as f:
                            f.write(decrypted)
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="Descargar archivo",
                                data=f,
                                file_name=selected_file,
                                mime="application/octet-stream"
                            )
                        SecureVault.encrypt_file(file_path, access_key_download)
                    else:
                        st.error("Llave de acceso incorrecta o archivo corrupto")
            else:
                st.info("No hay archivos protegidos disponibles")

def chat_interface():
    st.subheader("💬 Asistente de Seguridad IA")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            msg_type = "user-message" if msg['role'] == "user" else "bot-message"
            st.markdown(f'''
            <div class="{msg_type}">
                {msg["content"]}
                <div class="timestamp">{msg["time"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Escribe tu mensaje:", key="chat_input")
        with col2:
            model_choice = st.selectbox("Modelo", MODELOS_IA, index=0, key="model_choice")
        
        if user_input and st.button("Enviar", key="send_button"):
            try:
                # Actualizar historial
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'time': datetime.now().strftime("%H:%M")
                })
                
                # Consultar a Groq
                chat_completion = groq_client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ],
                    stream=True
                )
                
                # Mostrar respuesta
                response_container = st.empty()
                full_response = ""
                
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        part = chunk.choices[0].delta.content
                        full_response += part
                        response_container.markdown(f'''
                        <div class="bot-message">
                            {full_response}
                            <div class="timestamp">{datetime.now().strftime("%H:%M")}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': full_response,
                    'time': datetime.now().strftime("%H:%M")
                })
                
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")

def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("🔐 WildPass Pro - Gestor de Seguridad")
    st.markdown("---")
    
    ai_assistant = AISecurityAssistant()
    
    menu = st.sidebar.radio(
        "Menú Principal",
        ["🏠 Inicio", "📊 Analizar", "🔧 Entrenar IA", "💬 Asistente IA", "🗃️ Carpeta Segura"]
    )
    
    if menu == "🏠 Inicio":
        st.subheader("Generador de Claves")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔢 Generar PIN"):
                pin = PasswordGenerator.generate_pin()
                st.code(pin, language="text")
                st.session_state.generated_passwords.append(pin)
        
        with col2:
            if st.button("🔑 Generar Llave"):
                access_key = PasswordGenerator.generate_access_key()
                st.code(access_key, language="text")
                st.session_state.generated_passwords.append(access_key)
        
        with col3:
            password_length = st.slider("Longitud", 8, 32, 16)
            if st.button("🔒 Generar Contraseña"):
                password = PasswordGenerator.generate_strong_password(password_length)
                st.code(password, language="text")
                st.session_state.generated_passwords.append(password)
        
        if 'generated_passwords' in st.session_state and st.session_state.generated_passwords:
            pass_str = "\n".join(st.session_state.generated_passwords)
            st.download_button(
                label="⬇️ Descargar Contraseñas",
                data=pass_str,
                file_name="contraseñas_seguras.txt",
                mime="text/plain"
            )
        
        if ai_assistant.model:
            try:
                sample_passwords = [PasswordGenerator.generate_strong_password() for _ in range(50)]
                strengths = [ai_assistant.model.predict_proba([SecurityAnalyzer.extract_features(pwd)])[0][1] for pwd in sample_passwords]
                fig = px.histogram(x=strengths, nbins=20, title='Distribución de Fortaleza')
                st.plotly_chart(fig)
            except:
                pass
    
    elif menu == "📊 Analizar":
        st.subheader("Analizador de Seguridad")
        password = st.text_input("Introduce una contraseña:", type="password")
        
        if password and ai_assistant.model:
            try:
                features = SecurityAnalyzer.extract_features(password)
                score = ai_assistant.model.predict_proba([features])[0][1] * 100
                
                st.metric("Puntuación de Seguridad", f"{score:.1f}%")
                st.progress(score/100)
                
                df = pd.DataFrame({
                    'Característica': ['Longitud', 'Mayúsculas', 'Dígitos', 'Símbolos', 'Unicidad'],
                    'Valor': features
                })
                fig = px.bar(df, x='Característica', y='Valor', title='Análisis Detallado')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif menu == "🔧 Entrenar IA":
        st.subheader("Entrenamiento del Modelo")
        if st.button("🚀 Iniciar Entrenamiento"):
            with st.spinner("Entrenando modelo de seguridad..."):
                if ai_assistant.train_model():
                    st.balloons()
    
    elif menu == "💬 Asistente IA":
        chat_interface()
    
    elif menu == "🗃️ Carpeta Segura":
        secure_folder_section()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if 'generated_passwords' not in st.session_state:
        st.session_state.generated_passwords = []
    main()
