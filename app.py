# app.py
import streamlit as st
import joblib
import numpy as np
import string
import random
from sklearn.ensemble import RandomForestClassifier
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

class PasswordModel:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            self.model = joblib.load("local_pass_model.pkl")
            st.success("Modelo de seguridad cargado!")
        except:
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
        X, y = self.generate_training_data()
        
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)
        joblib.dump(self.model, "local_pass_model.pkl")
        
    def analyze_password(self, password):
        if self.model is None:
            return 0.0
        features = self.extract_features(password)
        return self.model.predict_proba([features])[0][1] * 100

def main():
    st.set_page_config(page_title="WildPass Local", page_icon="🔒", layout="wide")
    
    st.title("🔐 WildPass Local - Generador Seguro")
    st.markdown("---")
    
    model = PasswordModel()
    
    with st.sidebar:
        st.header("Opciones")
        menu = st.radio("Menú:", ["Generar", "Analizar", "Entrenar"])
    
    if menu == "Generar":
        st.subheader("Generador de Contraseñas")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔒 Generar Fuerte"):
                password = model.generate_strong_password()
                st.code(password, language="text")
                
        with col2:
            if st.button("⚠ Generar Débil"):
                password = model.generate_weak_password()
                st.code(password, language="text")
                
    elif menu == "Analizar":
        st.subheader("Analizador de Seguridad")
        password = st.text_input("Introduce una contraseña:")
        if password:
            score = model.analyze_password(password)
            progress = score / 100
            
            st.metric("Puntuación de Seguridad", f"{score:.1f}%")
            st.progress(progress)
            
            features = model.extract_features(password)
            st.json({
                "Longitud": features[0],
                "Mayúsculas": features[1],
                "Dígitos": features[2],
                "Símbolos": features[3],
                "Unicidad": f"{features[4]*100:.1f}%"
            })
            
    elif menu == "Entrenar":
        st.subheader("Entrenamiento del Modelo")
        if st.button("🚀 Iniciar Entrenamiento"):
            with st.spinner("Entrenando modelo local..."):
                model.train_model()
            st.success("Modelo actualizado correctamente!")
            
            if model.model is not None:
                st.plotly_chart(self.create_feature_importance_plot(model.model))

if __name__ == "__main__":
    main()
