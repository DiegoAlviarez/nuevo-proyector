import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="FIFA 21 Neural Network Analysis", layout="wide")

st.title("🧠 FIFA 21 Neural Network Analysis")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/main/FIFA-21%20Complete.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    # Select features
    features = ['Age', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic']
    target = 'Overall'
    
    # Clean and prepare data
    X = df[features].values
    y = df[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Load data
with st.spinner("Loading FIFA 21 dataset..."):
    df = load_data()
    st.success("Dataset loaded successfully!")

# Preprocess data
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = create_model()

# Training configuration
epochs = st.slider("Number of epochs", min_value=10, max_value=200, value=100, step=10)
batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)

# Training
if st.button("Start Training"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_chart = st.empty()
    
    # Lists to store metrics
    train_loss_history = []
    val_loss_history = []
    epochs_range = range(1, epochs + 1)
    
    # Training loop
    for epoch in range(epochs):
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=1,
            batch_size=batch_size,
            verbose=0
        )
        
        # Update metrics
        train_loss_history.append(history.history['loss'][0])
        val_loss_history.append(history.history['val_loss'][0])
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training Progress: {int(progress * 100)}%")
        
        # Update chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_range[:epoch+1], y=train_loss_history,
                                mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(x=epochs_range[:epoch+1], y=val_loss_history,
                                mode='lines', name='Validation Loss'))
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        metrics_chart.plotly_chart(fig, use_container_width=True)
    
    st.success("Training completed!")
    
    # Evaluate model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    st.metric("Test Mean Squared Error", f"{test_loss[0]:.4f}")
    
    # Feature importance analysis
    st.subheader("Feature Importance Analysis")
    feature_names = ['Age', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic']
    weights = model.layers[0].get_weights()[0]
    importance = np.abs(weights).mean(axis=1)
    
    fig = go.Figure(go.Bar(
        x=feature_names,
        y=importance,
        text=np.round(importance, 3),
        textposition='auto',
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Add prediction interface
st.subheader("Player Rating Prediction")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=45, value=25)
    pace = st.number_input("Pace", min_value=0, max_value=100, value=70)
    shooting = st.number_input("Shooting", min_value=0, max_value=100, value=70)
    passing = st.number_input("Passing", min_value=0, max_value=100, value=70)

with col2:
    dribbling = st.number_input("Dribbling", min_value=0, max_value=100, value=70)
    defending = st.number_input("Defending", min_value=0, max_value=100, value=70)
    physic = st.number_input("Physic", min_value=0, max_value=100, value=70)

if st.button("Predict Rating"):
    # Create input array
    input_data = np.array([[age, pace, shooting, passing, dribbling, defending, physic]])
    
    # Scale input
    scaler = StandardScaler()
    scaler.fit(X)  # Fit on all data to match training
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0][0]
    
    st.metric("Predicted Overall Rating", f"{prediction:.1f}")
