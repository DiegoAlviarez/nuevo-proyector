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

# Configuración inicial
SECURE_FOLDER = "secure_vault"
os.makedirs(SECURE_FOLDER, exist_ok=True)

# Inicialización del estado de la sesión
if 'generated_passwords' not in st.session_state:
    st.session_state.generated_passwords = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vault_key' not in st.session_state:
    st.session_state.vault_key = None

# Configuración de la página
st.set_page_config(
    page_title="WildPass Pro",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Animaciones y efectos dinámicos
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

class SecureVault:
    @staticmethod
    def generate_key(password: str) -> bytes:
        salt = b'secure_salt_123'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    @staticmethod
    def encrypt_file(file_path: str, password: str):
        key = SecureVault.generate_key(password)
        fernet = Fernet(key)
        
        with open(file_path, "rb") as file:
            original = file.read()
        
        encrypted = fernet.encrypt(original)
        
        with open(file_path, "wb") as file:
            file.write(encrypted)

    @staticmethod
    def decrypt_file(file_path: str, password: str):
        key = SecureVault.generate_key(password)
        fernet = Fernet(key)
        
        with open(file_path, "rb") as file:
            encrypted = file.read()
        
        try:
            decrypted = fernet.decrypt(encrypted)
            with open(file_path, "wb") as file:
                file.write(decrypted)
            return True
        except:
            return False

class PasswordModel:
    def __init__(self):
        self.model = None
        self.load_model()
        self.training_history = []

    def load_model(self):
        try:
            self.model = joblib.load("local_pass_model.pkl")
            st.success("✅ Modelo de seguridad cargado!")
        except Exception:
            st.warning("⚠ Modelo no encontrado, entrénalo primero")
            self.model = None

    def generate_weak_password(self):
        patterns = [
            lambda: ''.join(random.choice(string.ascii_lowercase) for _ in range(8)),
            lambda: ''.join(random.choice(["123456", "password", "qwerty", "admin"])),
            lambda: ''.join(random.choice(string.digits) for _ in range(6))
        ]
        return random.choice(patterns)()

    def generate_strong_password(self):
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.SystemRandom().choice(chars) for _ in range(16))

    def generate_pin(self, length=6):
        return ''.join(random.SystemRandom().choice(string.digits) for _ in range(length))
    
    def generate_access_key(self):
        chars = string.ascii_letters + string.digits + "-_"
        return ''.join(random.SystemRandom().choice(chars) for _ in range(24))

    def generate_training_data(self, samples=1000):
        X = []
        y = []
        for _ in range(samples//2):
            X.append(self.extract_features(self.generate_weak_password()))
            y.append(0)
            X.append(self.extract_features(self.generate_strong_password()))
            y.append(1)
        return np.array(X), np.array(y)

    def extract_features(self, password):
        return [
            len(password),
            sum(c.isupper() for c in password),
            sum(c.isdigit() for c in password),
            sum(c in string.punctuation for c in password),
            len(set(password))/max(len(password), 1)
        ]

    def train_model(self):
        try:
            X, y = self.generate_training_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            self.model = RandomForestClassifier(n_estimators=100)
            self.training_history = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            training_panel = st.empty()
            chart_placeholder = st.empty()

            def create_training_panel(epoch, accuracy, feature_importances):
                feature_bars = "\n".join([
                    f"Longitud   {'▮' * int(fi[0]*40)} {fi[0]*100:.1f}%",
                    f"Mayúsculas {'▮' * int(fi[1]*40)} {fi[1]*100:.1f}%",
                    f"Dígitos    {'▮' * int(fi[2]*40)} {fi[2]*100:.1f}%",
                    f"Símbolos   {'▮' * int(fi[3]*40)} {fi[3]*100:.1f}%",
                    f"Unicidad   {'▮' * int(fi[4]*40)} {fi[4]*100:.1f}%"
                ])
                
                panel = f"""
                ╭────────────────── WildPassPro - Entrenamiento de IA ──────────────────╮
                │                                                                        │
                │ Progreso del Entrenamiento:                                            │
                │ Árboles creados: {epoch}/100                                           │
                │ Precisión actual: {accuracy:.1%}                                      │
                │                                                                        │
                │ Características más importantes:                                       │
                {feature_bars}
                │                                                                        │
                │ Creando protección inteligente...                                      │
                ╰────────────────────────────────────────────────────────────────────────╯
                """
                return panel

            for epoch in range(1, 101):
                self.model.fit(X_train, y_train)
                acc = self.model.score(X_test, y_test)
                self.training_history.append(acc)
                fi = self.model.feature_importances_ if hasattr(self.model, 'feature_importances_') else [0.35, 0.25, 0.20, 0.15, 0.05]
                
                df = pd.DataFrame({'Época': range(1, epoch+1), 'Precisión': self.training_history})
                fig = px.line(df, x='Época', y='Precisión', title='Progreso del Entrenamiento')
                chart_placeholder.plotly_chart(fig)
                
                progress_bar.progress(epoch/100)
                status_text.text(f"Época: {epoch} - Precisión: {acc:.2%}")
                training_panel.code(create_training_panel(epoch, acc, fi))
                time.sleep(0.05)

            joblib.dump(self.model, "local_pass_model.pkl")
            
            features = ['Longitud', 'Mayúsculas', 'Dígitos', 'Símbolos', 'Unicidad']
            fig = px.bar(x=features, y=fi, title='Importancia de las Características')
            st.plotly_chart(fig)
            
            st.success(f"🎉 Modelo entrenado! Precisión: {acc:.2%}")
            return True
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False

def secure_folder_section():
    st.subheader("🔐 Carpeta Segura")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Subir Archivos")
        uploaded_file = st.file_uploader("Selecciona archivo a proteger:", 
                                       type=["txt", "pdf", "png", "jpg", "docx", "xlsx"])
        access_key = st.text_input("Crea una llave de acceso:", type="password")
        
        if uploaded_file and access_key:
            if len(access_key) < 8:
                st.error("La llave debe tener al menos 8 caracteres")
            else:
                file_path = os.path.join(SECURE_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                SecureVault.encrypt_file(file_path, access_key)
                st.success("Archivo protegido con éxito!")
                st.info("Guarda bien tu llave de acceso, sin ella no podrás recuperar el archivo")
    
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
                    temp_path = f"temp_{selected_file}"
                    
                    if SecureVault.decrypt_file(file_path, access_key_download):
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
    st.subheader("💬 Chat de Seguridad")
    
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            if msg['type'] == 'user':
                st.markdown(f'''
                <div class="user-message">
                    {msg["content"]}
                    <div class="timestamp">{msg["time"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="bot-message">
                    {msg["content"]}
                    <div class="timestamp">{msg["time"]}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        user_input = st.text_input("Escribe tu mensaje:", key="chat_input")
        
        if st.button("Enviar", key="send_button"):
            if user_input:
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'time': datetime.now().strftime("%H:%M")
                })
                
                respuestas = {
                    'hola': '¡Hola! Soy WildBot 🤖 ¿En qué puedo ayudarte?',
                    'ayuda': 'Puedo:\n🔸 Generar contraseñas seguras\n🔸 Analizar tu contraseña\n🔸 Entrenar el modelo IA\n🔸 Dar consejos de seguridad',
                    'seguridad': '🔒 Consejos de seguridad:\n1. 12+ caracteres\n2. Mayúsculas y minúsculas\n3. Números y símbolos\n4. Sin información personal',
                    'generar': 'Para generar:\n1. Ve a "🏠 Inicio"\n2. Click en "🔒 Generar"\n3. ¡Listo! 🎉',
                    'default': '🤖 No entendí. ¿Puedes reformularlo?'
                }
                
                respuesta = next((v for k, v in respuestas.items() if k in user_input.lower()), respuestas['default'])
                
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': respuesta,
                    'time': datetime.now().strftime("%H:%M")
                })
                
                if 'chat_input' in st.session_state:
                    del st.session_state.chat_input
                st.experimental_rerun()

def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("🔐 WildPass Pro - Gestor de Contraseñas")
    st.markdown("---")
    
    model = PasswordModel()
    
    menu = st.sidebar.radio(
        "Menú Principal",
        ["🏠 Inicio", "📊 Analizar", "🔧 Entrenar IA", "💬 Chat", "🗃️ Carpeta Segura"]
    )
    
    if menu == "🏠 Inicio":
        st.subheader("Generador de Claves")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔢 Generar PIN"):
                pin = model.generate_pin()
                st.code(pin, language="text")
                st.session_state.generated_passwords.append(pin)
        
        with col2:
            if st.button("🔑 Generar Llave"):
                access_key = model.generate_access_key()
                st.code(access_key, language="text")
                st.session_state.generated_passwords.append(access_key)
        
        with col3:
            if st.button("🔒 Generar Contraseña"):
                password = model.generate_strong_password()
                st.code(password, language="text")
                st.session_state.generated_passwords.append(password)
        
        if st.session_state.generated_passwords:
            pass_str = "\n".join(st.session_state.generated_passwords)
            st.download_button(
                label="⬇️ Descargar Contraseñas",
                data=pass_str,
                file_name="contraseñas_seguras.txt",
                mime="text/plain"
            )
        
        try:
            sample_passwords = [model.generate_strong_password() for _ in range(50)]
            strengths = [model.model.predict_proba([model.extract_features(pwd)])[0][1] for pwd in sample_passwords]
            fig = px.histogram(x=strengths, nbins=20, title='Distribución de Fortaleza de Contraseñas')
            st.plotly_chart(fig)
        except:
            pass
    
    elif menu == "📊 Analizar":
        st.subheader("Analizador de Seguridad")
        password = st.text_input("Introduce una contraseña:", type="password")
        
        if password:
            if model.model is None:
                st.error("Primero entrena el modelo!")
            else:
                try:
                    features = model.extract_features(password)
                    score = model.model.predict_proba([features])[0][1] * 100
                    
                    st.metric("Puntuación de Seguridad", f"{score:.1f}%")
                    st.progress(score/100)
                    
                    df = pd.DataFrame({
                        'Característica': ['Longitud', 'Mayúsculas', 'Dígitos', 'Símbolos', 'Unicidad'],
                        'Valor': features
                    })
                    fig = px.bar(df, x='Característica', y='Valor', title='Análisis de Características')
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif menu == "🔧 Entrenar IA":
        st.subheader("Entrenamiento del Modelo")
        if st.button("🚀 Iniciar Entrenamiento"):
            with st.spinner("Entrenando IA..."):
                if model.train_model():
                    st.balloons()
    
    elif menu == "💬 Chat":
        chat_interface()
    
    elif menu == "🗃️ Carpeta Segura":
        secure_folder_section()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
