import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import requests

# Configure page
st.set_page_config(page_title="Football Analytics", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # URLs for the datasets
        base_url = "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/"
        datasets = {
            'players': 'players.csv',
            'appearances': 'appearances.csv',
            'clubs': 'clubs.csv',
            'competitions': 'competitions.csv',
            'games': 'games.csv'
        }
        
        data = {}
        for key, filename in datasets.items():
            url = base_url + filename
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            data[key] = pd.read_csv(StringIO(response.text))
        
        # Clean and preprocess data
        data['players']['market_value_eur'] = data['players']['market_value_eur'].fillna(0)
        data['players']['current_club_id'] = data['players']['current_club_id'].fillna(-1)
        
        # Add age calculation
        data['players']['birth_date'] = pd.to_datetime(data['players']['birth_date'])
        current_date = pd.Timestamp('now')
        data['players']['age'] = ((current_date - data['players']['birth_date']).dt.days / 365.25).astype(int)
        
        # Merge relevant data
        player_stats = data['appearances'].groupby('player_id').agg({
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum',
            'yellow_cards': 'sum',
            'red_cards': 'sum'
        }).reset_index()
        
        data['players'] = data['players'].merge(player_stats, on='player_id', how='left')
        data['players'] = data['players'].fillna(0)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def prepare_features(players_df):
    try:
        # Create performance metrics
        players_df['goals_per_90'] = (players_df['goals'] * 90) / players_df['minutes_played'].replace(0, 90)
        players_df['assists_per_90'] = (players_df['assists'] * 90) / players_df['minutes_played'].replace(0, 90)
        
        # Select features for the model
        features = [
            'age', 'market_value_eur', 'goals_per_90', 'assists_per_90',
            'minutes_played', 'yellow_cards', 'red_cards'
        ]
        
        return players_df[features].fillna(0)
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None

def train_prediction_model(X, y):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model (simplified for stability)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {
            'model': model,
            'scaler': scaler,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_scores': cross_val_score(model, X_train_scaled, y_train, cv=5)
        }
        
        return results
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Main app
try:
    # Load data
    data = load_data()
    
    if data is not None:
        players_df = data['players']
        
        # Prepare features and target
        X = prepare_features(players_df)
        y = players_df['market_value_eur']
        
        if X is not None:
            # Train model
            model_results = train_prediction_model(X, y)
            
            if model_results is not None:
                # Sidebar
                st.sidebar.title("Football Analytics")
                analysis_type = st.sidebar.selectbox(
                    "Select Analysis Type",
                    ["Player Statistics", "Market Value Prediction"]
                )
                
                # Main content
                st.title("Football Analytics Dashboard")
                
                if analysis_type == "Player Statistics":
                    st.header("Player Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Age distribution
                        fig_age = px.histogram(
                            players_df,
                            x="age",
                            title="Age Distribution of Players",
                            labels={"age": "Age", "count": "Number of Players"}
                        )
                        st.plotly_chart(fig_age, use_container_width=True)
                    
                    with col2:
                        # Market value distribution
                        fig_value = px.box(
                            players_df,
                            y="market_value_eur",
                            title="Market Value Distribution",
                            labels={"market_value_eur": "Market Value (EUR)"}
                        )
                        st.plotly_chart(fig_value, use_container_width=True)
                
                elif analysis_type == "Market Value Prediction":
                    st.header("Market Value Prediction")
                    
                    # Model performance metrics
                    st.subheader("Model Performance")
                    metrics_df = pd.DataFrame({
                        'Metric': ['R² Score', 'RMSE', 'MAE'],
                        'Value': [
                            f"{model_results['r2']:.3f}",
                            f"€{model_results['rmse']:,.2f}",
                            f"€{model_results['mae']:,.2f}"
                        ]
                    })
                    st.table(metrics_df)
                    
                    # User input for prediction
                    st.subheader("Predict Player Market Value")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        input_age = st.number_input("Age", min_value=15, max_value=45, value=25)
                        input_goals = st.number_input("Goals", min_value=0, value=0)
                        input_assists = st.number_input("Assists", min_value=0, value=0)
                    
                    with col2:
                        input_minutes = st.number_input("Minutes Played", min_value=0, value=900)
                        input_yellows = st.number_input("Yellow Cards", min_value=0, value=0)
                        input_reds = st.number_input("Red Cards", min_value=0, value=0)
                    
                    if st.button("Predict Market Value"):
                        # Calculate performance metrics
                        goals_per_90 = (input_goals * 90) / max(input_minutes, 90)
                        assists_per_90 = (input_assists * 90) / max(input_minutes, 90)
                        
                        # Prepare input data
                        input_data = np.array([[
                            input_age, 0, goals_per_90, assists_per_90,
                            input_minutes, input_yellows, input_reds
                        ]])
                        input_scaled = model_results['scaler'].transform(input_data)
                        
                        # Make prediction
                        prediction = model_results['model'].predict(input_scaled)[0]
                        
                        st.success(f"Predicted Market Value: €{prediction:,.2f}")
                
                # Footer
                st.markdown("---")
                st.markdown("Data source: [Transfermarkt Datasets](https://github.com/dcaribou/transfermarkt-datasets)")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please try refreshing the page. If the error persists, check the data source or contact support.")
