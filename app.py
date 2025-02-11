# app.py
import streamlit as st
import os
import requests
from datetime import datetime

# Configuración de API (¡NUNCA expongas esto en tu código!)
API_KEY = "sk-9815428384464ad394db0391a8e3a33c"  # ← Configura esto en tus variables de entorno
API_URL = "https://chat.deepseek.com"  # Alternativa  # Ejemplo, verifica la URL real

# Configuración inicial
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
    layout="centered"
)

# Estilos CSS
st.markdown("""
<style>
.stApp {
    background: #1a1a1a;
    color: white;
}
.stTextInput>div>div>input {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

def deepseek_chat(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def generar_contraseña():
    prompt = "Genera una contraseña segura de 16 caracteres con letras, números y símbolos. Solo muestra la contraseña:"
    return deepseek_chat(prompt)

def analizar_contraseña(password):
    prompt = f"""Analiza la seguridad de esta contraseña: {password}
    Devuelve solo un porcentaje numérico entre 0-100 sin texto adicional."""
    return deepseek_chat(prompt)

def main():
    st.title("🔐 WildPass Pro")
    
    menu = st.sidebar.selectbox("Menú", [
        "🏠 Inicio", 
        "📊 Analizar", 
        "💬 Chat"
    ])
    
    if menu == "🏠 Inicio":
        st.subheader("Generador de Contraseñas")
        if st.button("🔒 Generar Contraseña"):
            password = generar_contraseña()
            st.session_state.generated_passwords.append(password)
            st.code(password)
            
    elif menu == "📊 Analizar":
        st.subheader("Analizador de Seguridad")
        password = st.text_input("Ingresa una contraseña:", type="password")
        if password:
            resultado = analizar_contraseña(password)
            st.metric("Fortaleza", f"{resultado}%")
            
    elif menu == "💬 Chat":
        st.subheader("Chat de Seguridad")
        user_input = st.text_input("Escribe tu mensaje:")
        
        if user_input:
            respuesta = deepseek_chat(user_input)
            st.session_state.chat_history.append({
                'usuario': user_input,
                'bot': respuesta,
                'hora': datetime.now().strftime("%H:%M")
            })
            
        for chat in st.session_state.chat_history:
            st.markdown(f"""
            <div style='margin: 10px; padding: 10px; border-radius: 5px; background: #2d2d2d;'>
                <div style='color: #00ff9d;'>Usuario ({chat['hora']}):</div>
                <div>{chat['usuario']}</div>
                <div style='color: #00b8ff; margin-top: 5px;'>WildPass ({chat['hora']}):</div>
                <div>{chat['bot']}</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
