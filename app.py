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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Configuración de modelos
DEEPSEEK_MODEL = "deepseek-ai/deepseek-chat"
SECURE_FOLDER = "secure_vault"
os.makedirs(SECURE_FOLDER, exist_ok=True)

# Inicialización del estado
if 'generated_passwords' not in st.session_state:
    st.session_state.generated_passwords = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Configuración de página
st.set_page_config(
    page_title="WildPass Pro",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown(f"""
<style>
@keyframes gradient {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

.stApp {{
    background: linear-gradient(-45deg, #1a1a1a, #2a2a2a, #3a3a3a, #4a4a4a);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}}

.main {{
    background-color: rgba(0, 0, 0, 0.8);
    padding: 30px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}}

.chat-container {{
    max-height: 500px;
    overflow-y: auto;
    padding: 20px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 15px;
    margin-bottom: 20px;
}}

.user-message {{
    background: rgba(0, 255, 157, 0.1);
    padding: 10px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 70%;
    float: right;
}}

.bot-message {{
    background: rgba(0, 184, 255, 0.1);
    padding: 10px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 70%;
    float: left;
}}
</style>
""", unsafe_allow_html=True)

class SecureVault:
    @staticmethod
    def generate_key(password: str):
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
        with open(file_path, "rb") as f:
            encrypted = fernet.encrypt(f.read())
        with open(file_path, "wb") as f:
            f.write(encrypted)

    @staticmethod
    def decrypt_file(file_path: str, password: str):
        key = SecureVault.generate_key(password)
        fernet = Fernet(key)
        try:
            with open(file_path, "rb") as f:
                decrypted = fernet.decrypt(f.read())
            with open(file_path, "wb") as f:
                f.write(decrypted)
            return True
        except:
            return False

class PasswordModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.deepseek = None
        self.load_models()
        self.training_history = []

    def load_models(self):
        try:
            # Cargar DeepSeek
            self.tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
            self.deepseek = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL)
            
            # Cargar modelo local
            try:
                self.model = joblib.load("local_pass_model.pkl")
            except:
                self.model = RandomForestClassifier()
                
        except Exception as e:
            st.error(f"Error cargando modelos: {str(e)}")

    def deepseek_generate_password(self):
        prompt = "Genera una contraseña segura de 16 caracteres con letras, números y símbolos. Solo muestra la contraseña:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.deepseek.generate(
            inputs.input_ids,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(":")[-1].strip()

    def deepseek_train_model(self):
        try:
            # Generar datos de entrenamiento con DeepSeek
            st.info("🧠 DeepSeek está generando datos de entrenamiento...")
            training_data = self.generate_training_data_with_deepseek()
            
            # Preparar datos
            X = np.array([self.extract_features(pwd) for pwd, label in training_data])
            y = np.array([label for pwd, label in training_data])
            
            # Entrenamiento
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            self.model = RandomForestClassifier(n_estimators=100)
            
            progress_bar = st.progress(0)
            status = st.empty()
            chart = st.empty()
            
            for epoch in range(1, 101):
                self.model.fit(X_train, y_train)
                acc = self.model.score(X_test, y_test)
                self.training_history.append(acc)
                
                # Actualizar interfaz
                progress_bar.progress(epoch)
                df = pd.DataFrame({'Época': range(epoch), 'Precisión': self.training_history})
                fig = px.line(df, x='Época', y='Precisión', title='Progreso del Entrenamiento')
                chart.plotly_chart(fig)
                
                # Análisis con DeepSeek
                analysis = self.deepseek_analyze_training(epoch, acc)
                status.markdown(f"""
                **Precisión actual:** {acc:.2%}
                **Análisis de DeepSeek:**  
                {analysis}
                """)
                
                time.sleep(0.05)
            
            joblib.dump(self.model, "local_pass_model.pkl")
            st.success("✅ Entrenamiento completado!")
            return True
            
        except Exception as e:
            st.error(f"Error en entrenamiento: {str(e)}")
            return False

    def generate_training_data_with_deepseek(self, samples=500):
        prompt = """Genera ejemplos de contraseñas seguras (1) e inseguras (0):
        Formato: contraseña|etiqueta
        Ejemplos:
        password123|0
        G7$kLm9!qRt4&wVz|1
        """
        generator = pipeline("text-generation", model=DEEPSEEK_MODEL)
        response = generator(prompt, max_length=2000, num_return_sequences=1)[0]['generated_text']
        
        training_data = []
        for line in response.split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    try:
                        training_data.append((parts[0].strip(), int(parts[1].strip())))
                    except:
                        continue
        return training_data

    def deepseek_analyze_training(self, epoch, accuracy):
        prompt = f"""
        Analiza el progreso del entrenamiento de un modelo de seguridad de contraseñas:
        - Época actual: {epoch}/100
        - Precisión: {accuracy:.2%}
        - Características importantes: {self.model.feature_importances_}
        Proporciona recomendaciones para mejorar el modelo:
        """
        generator = pipeline("text-generation", model=DEEPSEEK_MODEL)
        response = generator(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
        return response.split("Proporciona recomendaciones")[-1].strip()

    def extract_features(self, password):
        return [
            len(password),
            sum(c.isupper() for c in password),
            sum(c.isdigit() for c in password),
            sum(c in string.punctuation for c in password),
            len(set(password))/len(password)
        ]

def secure_folder_section():
    st.subheader("🔐 Carpeta Segura")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Subir archivo")
        password = st.text_input("Clave de acceso", type="password")
        if uploaded_file and password:
            file_path = os.path.join(SECURE_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            SecureVault.encrypt_file(file_path, password)
            st.success("Archivo protegido!")
    
    with col2:
        access_key = st.text_input("Ingresar clave", type="password")
        if access_key:
            files = [f for f in os.listdir(SECURE_FOLDER) if os.path.isfile(os.path.join(SECURE_FOLDER, f))]
            selected = st.selectbox("Archivos disponibles", files)
            if selected:
                file_path = os.path.join(SECURE_FOLDER, selected)
                if SecureVault.decrypt_file(file_path, access_key):
                    with open(file_path, "rb") as f:
                        st.download_button("Descargar", f, file_name=selected)
                    SecureVault.encrypt_file(file_path, access_key)

def chat_interface():
    st.subheader("💬 Chat de Seguridad")
    
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history[-10:]:
            css_class = "user-message" if msg['type'] == 'user' else "bot-message"
            st.markdown(f'''
            <div class="{css_class}">
                <div class="message-content">{msg["content"]}</div>
                <div class="timestamp">{msg["time"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        user_input = st.text_input("Escribe tu mensaje:", key="chat_input")
        
        if st.button("Enviar") and user_input:
            # Añadir mensaje de usuario
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input,
                'time': datetime.now().strftime("%H:%M")
            })
            
            # Generar respuesta con DeepSeek
            generator = pipeline("text-generation", model=DEEPSEEK_MODEL)
            response = generator(
                user_input,
                max_length=300,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )[0]['generated_text']
            
            # Añadir respuesta
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': response,
                'time': datetime.now().strftime("%H:%M")
            })
            
            st.experimental_rerun()

def main():
    st.title("🔐 WildPass Pro")
    model = PasswordModel()
    
    menu = st.sidebar.selectbox("Menú", [
        "🏠 Inicio", 
        "📊 Analizar", 
        "🔧 Entrenar IA", 
        "💬 Chat", 
        "🗃️ Carpeta Segura"
    ])
    
    if menu == "🏠 Inicio":
        st.subheader("Generador de Contraseñas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔢 Generar PIN"):
                pin = ''.join(random.choices(string.digits, k=6))
                st.code(pin)
                st.session_state.generated_passwords.append(pin)
        
        with col2:
            if st.button("🔑 Generar Llave"):
                key = ''.join(random.choices(string.ascii_letters + string.digits + '-_', k=24))
                st.code(key)
                st.session_state.generated_passwords.append(key)
        
        with col3:
            if st.button("🔒 Generar Contraseña"):
                password = model.deepseek_generate_password()
                st.code(password)
                st.session_state.generated_passwords.append(password)
        
        if st.session_state.generated_passwords:
            st.download_button(
                "📥 Descargar Contraseñas",
                "\n".join(st.session_state.generated_passwords),
                file_name="passwords.txt"
            )
    
    elif menu == "📊 Analizar":
        st.subheader("Analizador de Seguridad")
        password = st.text_input("Ingresa una contraseña:", type="password")
        if password and model.model:
            features = model.extract_features(password)
            proba = model.model.predict_proba([features])[0][1]
            st.metric("Fortaleza", f"{proba*100:.1f}%")
            st.progress(proba)
    
    elif menu == "🔧 Entrenar IA":
        st.subheader("Entrenamiento del Modelo")
        if st.button("🚀 Iniciar Entrenamiento con DeepSeek"):
            with st.spinner("DeepSeek está entrenando el modelo..."):
                model.deepseek_train_model()
    
    elif menu == "💬 Chat":
        chat_interface()
    
    elif menu == "🗃️ Carpeta Segura":
        secure_folder_section()

if __name__ == "__main__":
    main()
