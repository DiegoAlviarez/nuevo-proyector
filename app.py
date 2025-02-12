import openai
import streamlit as st

# Aquí defines la clave directamente, ya que es para enseñanza
GROQ_API_KEY = "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"

# Inicializar cliente OpenAI con la API de Groq
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

MODEL_NAME = "llama3-70b-8192"

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
    st.title("🤖 Chatbot de prueba para WildPassPro")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy un chatbot impulsado por Llama 3 en Groq. ¿En qué puedo ayudarte?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe tu mensaje..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Pensando..."):
            response = chat_with_groq(st.session_state.messages)

        st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()

if __name__ == "__main__":
    main()
