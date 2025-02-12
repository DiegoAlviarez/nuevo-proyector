import streamlit as st
import openai
import os

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("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")
)

MODEL_NAME = "llama3-70b-8192"  # Modelo actualizado (ver modelos disponibles en Groq)

def chat_with_groq(messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("🤖 Chatbot con Groq y Llama 3")
    
    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy un chatbot impulsado por Llama 3 en Groq. ¿En qué puedo ayudarte?"}
        ]
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Manejar entrada de usuario
    if prompt := st.chat_input("Escribe tu mensaje..."):
        # Añadir mensaje de usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Obtener respuesta
        with st.spinner("Pensando..."):
            response = chat_with_groq(st.session_state.messages)
        
        # Añadir respuesta del asistente
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Actualizar la interfaz
        st.rerun()

if __name__ == "__main__":
    main()
