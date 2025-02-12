import streamlit as st
import requests
import json

# Configuración de la API de Groq
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"  # URL de la API de Groq
GROQ_API_KEY = "tu_api_key_de_groq"  # Reemplaza con tu API key de Groq
MODEL_NAME = "llama-3.3-70b-versatile"  # Nombre del modelo de Groq

# Función para interactuar con Groq
def chat_with_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Interfaz de usuario en Streamlit
def main():
    st.title("Chatbot con Groq y LLaMA 3.3-70b en Streamlit")
    
    # Inicializar el historial del chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar el historial del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Entrada del usuario
    if prompt := st.chat_input("¿Qué quieres preguntar?"):
        # Añadir el mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Obtener la respuesta del chatbot
        response = chat_with_groq(prompt)
        
        # Añadir la respuesta del chatbot al historial
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
