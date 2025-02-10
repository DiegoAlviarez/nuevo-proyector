# app.py
import streamlit as st
import joblib
import numpy as np
import string
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PasswordModel:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            self.model = joblib.load("local_pass_model.pkl")
            st.success("Modelo de seguridad cargado!")
        except Exception as e:
            st.warning("Modelo no encontrado, por favor entrénalo primero")
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
        
        # Crear elementos de visualización
        progress_bar = st.progress(0)
        status_text = st.empty()
        training_panel = st.empty()
        
        # Nuevo: Función para generar el panel de entrenamiento
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
            │                                                                        │
            │ Árboles creados: {epoch}/100                                           
            │ Precisión actual: {accuracy:.1%}                                      
            │                                                                        │
            │ Características más importantes:                                       │
            {feature_bars}
            │                                                                        │
            │ Creando protección inteligente...                                      │
            ╰────────────────────────────────────────────────────────────────────────╯
            """
            return panel
        
        # Entrenamiento con actualización visual
        for epoch in range(1, 101):
            self.model.fit(X_train, y_train)
            acc = self.model.score(X_test, y_test)
            
            # Obtener importancia de características
            if hasattr(self.model, 'feature_importances_'):
                fi = self.model.feature_importances_
            else:
                fi = [0.35, 0.25, 0.20, 0.15, 0.05]  # Valores por defecto
            
            # Actualizar elementos visuales
            progress_bar.progress(epoch/100)
            status_text.text(f"Época: {epoch} - Precisión: {acc:.2%}")
            training_panel.code(create_training_panel(epoch, acc, fi))
            
            time.sleep(0.1)  # Pequeña pausa para visualización
            
        joblib.dump(self.model, "local_pass_model.pkl")
        st.success(f"Modelo entrenado! Precisión: {acc:.2%}")
        return True
    except Exception as e:
        st.error(f"Error en entrenamiento: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="WildPass Local",
        page_icon="🔒",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔐 WildPass Local - Generador Seguro")
    st.markdown("---")
    
    model = PasswordModel()
    
    menu = st.sidebar.selectbox(
        "Menú Principal",
        ["🏠 Inicio", "🔧 Entrenar Modelo", "📊 Analizar Contraseña"]
    )
    
    if menu == "🏠 Inicio":
        st.subheader("Generar Contraseñas")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔒 Generar Contraseña Fuerte"):
                password = model.generate_strong_password()
                st.code(password, language="text")
                
        with col2:
            if st.button("⚠ Generar Contraseña Débil"):
                password = model.generate_weak_password()
                st.code(password, language="text")
                
    elif menu == "🔧 Entrenar Modelo":
        st.subheader("Entrenamiento del Modelo IA")
        if st.button("🚀 Iniciar Entrenamiento"):
            with st.spinner("Entrenando modelo con 1000 muestras..."):
                if model.train_model():
                    st.balloons()
                else:
                    st.error("Error en el entrenamiento")
                    
    elif menu == "📊 Analizar Contraseña":
        st.subheader("Analizador de Seguridad")
        password = st.text_input("Introduce una contraseña para analizar:", type="password")
        
        if password:
            if model.model is None:
                st.error("Primero entrena el modelo!")
            else:
                try:
                    score = model.model.predict_proba([model.extract_features(password)])[0][1] * 100
                    st.metric("Puntuación de Seguridad", f"{score:.1f}%")
                    
                    features = model.extract_features(password)
                    st.write("**Detalles del Análisis:**")
                    st.json({
                        "Longitud": features[0],
                        "Mayúsculas": features[1],
                        "Dígitos": features[2],
                        "Símbolos": features[3],
                        "Unicidad": f"{features[4]*100:.1f}%"
                    })
                except Exception as e:
                    st.error(f"Error en análisis: {str(e)}")

if __name__ == "__main__":
    main()
