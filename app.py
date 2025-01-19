import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import requests
from io import StringIO

st.set_page_config(page_title="Football Analytics", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_data():
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
        data[key] = pd.read_csv(url)
    
    # Clean and preprocess data
    data['players']['market_value_eur'] = data['players']['market_value_eur'].fillna(0)
    data['players']['current_club_id'] = data['players']['current_club_id'].fillna(-1)
    
    # Add age calculation
    data['players']['birth_date'] = pd.to_datetime(data['players']['birth_date'])
    data['players']['age'] = (datetime.now() - data['players']['birth_date']).astype('<m8[Y]')
    
    # Merge relevant data
    player_stats = data['appearances'].groupby('player_id').agg({
        'goals': 'sum',
        'assists': 'sum',
        'minutes_played': 'sum',
        'yellow_cards': 'sum',
        'red_cards': 'sum'
    }).reset_index()
    
    data['players'] = data['players'].merge(player_stats, on='player_id', how='left')
    
    return data

def prepare_features(players_df):
    # Create performance metrics
    players_df['goals_per_90'] = (players_df['goals'] * 90) / players_df['minutes_played'].replace(0, 90)
    players_df['assists_per_90'] = (players_df['assists'] * 90) / players_df['minutes_played'].replace(0, 90)
    
    # Select features for the model
    features = [
        'age', 'market_value_eur', 'goals_per_90', 'assists_per_90',
        'minutes_played', 'yellow_cards', 'red_cards'
    ]
    
    return players_df[features].fillna(0)

def train_prediction_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results[name] = {
            'model': model,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_scores': cross_val_score(model, X_train_scaled, y_train, cv=5)
        }
    
    # Select best model based on R2 score
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    
    return results[best_model_name]['model'], scaler, results

# Load data
data = load_data()
players_df = data['players']

# Prepare features and target
X = prepare_features(players_df)
y = players_df['market_value_eur']

# Train model
model, scaler, model_results = train_prediction_model(X, y)

# Sidebar
st.sidebar.title("Football Analytics")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Player Statistics", "Market Value Prediction", "Team Analysis", "Performance Metrics"]
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
        st.plotly_chart(fig_age)
    
    with col2:
        # Market value distribution
        fig_value = px.box(
            players_df,
            y="market_value_eur",
            title="Market Value Distribution",
            labels={"market_value_eur": "Market Value (EUR)"}
        )
        st.plotly_chart(fig_value)
    
    # Position analysis
    st.subheader("Performance by Position")
    position_stats = players_df.groupby('sub_position').agg({
        'market_value_eur': 'mean',
        'goals_per_90': 'mean',
        'assists_per_90': 'mean'
    }).reset_index()
    
    fig_position = px.bar(
        position_stats,
        x='sub_position',
        y=['goals_per_90', 'assists_per_90'],
        title="Average Performance Metrics by Position",
        barmode='group'
    )
    st.plotly_chart(fig_position)

elif analysis_type == "Market Value Prediction":
    st.header("Market Value Prediction")
    
    # Model performance metrics
    st.subheader("Model Performance")
    metrics_df = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE', 'MAE', 'CV Score (mean ± std)'],
        'Value': [
            f"{model_results['r2']:.3f}",
            f"€{model_results['rmse']:,.2f}",
            f"€{model_results['mae']:,.2f}",
            f"{model_results['cv_scores'].mean():.3f} ± {model_results['cv_scores'].std():.3f}"
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
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"Predicted Market Value: €{prediction:,.2f}")

elif analysis_type == "Team Analysis":
    st.header("Team Analysis")
    
    # Team value analysis
    team_values = players_df.groupby(['current_club_id']).agg({
        'market_value_eur': ['mean', 'sum', 'count'],
        'age': 'mean'
    }).reset_index()
    
    team_values.columns = ['Club ID', 'Average Player Value', 'Total Team Value', 'Squad Size', 'Average Age']
    
    # Team value scatter plot
    fig_team = px.scatter(
        team_values,
        x="Average Player Value",
        y="Total Team Value",
        size="Squad Size",
        color="Average Age",
        title="Team Value Analysis",
        labels={
            "Average Player Value": "Average Player Value (EUR)",
            "Total Team Value": "Total Team Value (EUR)"
        }
    )
    st.plotly_chart(fig_team)
    
    # Top valuable teams
    st.subheader("Top 10 Most Valuable Teams")
    top_teams = team_values.nlargest(10, 'Total Team Value')
    fig_top = px.bar(
        top_teams,
        x='Club ID',
        y='Total Team Value',
        title="Top 10 Teams by Total Value"
    )
    st.plotly_chart(fig_top)

elif analysis_type == "Performance Metrics":
    st.header("Performance Metrics")
    
    # Performance correlation matrix
    performance_cols = ['age', 'market_value_eur', 'goals_per_90', 'assists_per_90', 
                       'minutes_played', 'yellow_cards', 'red_cards']
    correlation = players_df[performance_cols].corr()
    
    fig_corr = px.imshow(
        correlation,
        title="Feature Correlation Matrix",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig_corr)
    
    # Performance trends
    st.subheader("Performance Trends")
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=players_df['age'],
        y=players_df['goals_per_90'],
        mode='markers',
        name='Goals per 90'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=players_df['age'],
        y=players_df['assists_per_90'],
        mode='markers',
        name='Assists per 90'
    ))
    
    fig_trends.update_layout(
        title="Performance Metrics vs Age",
        xaxis_title="Age",
        yaxis_title="Performance Metric"
    )
    
    st.plotly_chart(fig_trends)

# Footer
st.markdown("---")
st.markdown("Data source: [Transfermarkt Datasets](https://github.com/dcaribou/transfermarkt-datasets)")