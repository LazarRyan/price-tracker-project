import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from amadeus import Client, ResponseError
import random
import logging
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage
from google.oauth2 import service_account
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Amadeus client using Streamlit secrets
try:
    amadeus = Client(
        client_id=st.secrets["AMADEUS_API_KEY"],
        client_secret=st.secrets["AMADEUS_API_SECRET"]
    )
except Exception as e:
    logging.error(f"Failed to initialize Amadeus client: {str(e)}")
    amadeus = None

# Initialize Google Cloud Storage using Streamlit secrets
try:
    # Convert Streamlit's AttrDict to regular dict
    credentials_dict = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    
    storage_client = storage.Client(credentials=credentials)
    bucket_name = st.secrets["gcs_bucket_name"]
    bucket = storage_client.bucket(bucket_name)
except Exception as e:
    logging.error(f"Failed to initialize GCS client: {str(e)}")
    storage_client = None

def save_to_gcs(df, filename):
    """
    Save DataFrame to Google Cloud Storage
    """
    try:
        if storage_client is None:
            logging.error("GCS client not initialized")
            return False
            
        csv_data = df.to_csv(index=False)
        blob = bucket.blob(filename)
        blob.upload_from_string(csv_data, content_type='text/csv')
        
        logging.info(f"Successfully saved {filename} to GCS")
        return True
    except Exception as e:
        logging.error(f"Error saving to GCS: {str(e)}")
        return False

def load_from_gcs(filename):
    """
    Load DataFrame from Google Cloud Storage
    """
    try:
        if storage_client is None:
            logging.error("GCS client not initialized")
            return None
            
        blob = bucket.blob(filename)
        data = blob.download_as_string()
        df = pd.read_csv(pd.io.common.BytesIO(data))
        return df
    except Exception as e:
        logging.error(f"Error loading from GCS: {str(e)}")
        return None

def fetch_and_process_data(origin, destination, start_date, end_date):
    """
    Fetch and process flight data with error handling and GCS integration
    """
    # Update filename format
    filename = f"flight_prices_{origin}_{destination}.csv"
    existing_df = load_from_gcs(filename)
    
    if existing_df is not None:
        return existing_df

    all_data = []
    current_date = start_date
    end_date = start_date + relativedelta(months=12)
    
    # Setup progress tracking
    total_iterations = ((end_date - current_date).days // 30) * 3
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0

    while current_date < end_date:
        month_end = current_date + relativedelta(months=1, days=-1)
        sample_dates = [current_date + timedelta(days=random.randint(0, (month_end - current_date).days)) 
                       for _ in range(3)]

        for sample_date in sample_dates:
            status_text.text(f"Analyzing flights for {sample_date.strftime('%Y-%m-%d')}...")
            
            try:
                response = amadeus.shopping.flight_offers_search.get(
                    originLocationCode=origin,
                    destinationLocationCode=destination,
                    departureDate=sample_date.strftime('%Y-%m-%d'),
                    adults=1
                )
                
                for offer in response.data:
                    price = float(offer['price']['total'])
                    departure = offer['itineraries'][0]['segments'][0]['departure']['at']
                    airline = offer['validatingAirlineCodes'][0]
                    
                    all_data.append({
                        'departure': departure,
                        'price': price,
                        'airline': airline
                    })
                    
            except ResponseError as e:
                logging.error(f"Amadeus API error: {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error in data fetching: {str(e)}")
                continue
            
            processed += 1
            progress_bar.progress(processed / total_iterations)
            time.sleep(0.1)

        current_date += relativedelta(months=1)
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['departure'] = pd.to_datetime(df['departure'])
        # Save to GCS
        save_to_gcs(df, filename)
    return df

def prepare_features(df):
    """
    Prepare features for the prediction model
    """
    if df.empty:
        return None, None
    
    df['day_of_week'] = df['departure'].dt.dayofweek
    df['month'] = df['departure'].dt.month
    df['day'] = df['departure'].dt.day
    df['days_until_flight'] = (df['departure'] - pd.Timestamp.now()).dt.days
    
    label_encoder = LabelEncoder()
    df['airline_encoded'] = label_encoder.fit_transform(df['airline'])
    
    features = ['day_of_week', 'month', 'day', 'days_until_flight', 'airline_encoded']
    X = df[features]
    y = df['price']
    
    return X, y

def train_model(X, y):
    """
    Train the prediction model
    """
    if X is None or y is None:
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def generate_future_dates(start_date, days=90):
    """
    Generate future dates for prediction
    """
    future_dates = pd.date_range(start=start_date, periods=days, freq='D')
    future_df = pd.DataFrame({'departure': future_dates})
    
    future_df['day_of_week'] = future_df['departure'].dt.dayofweek
    future_df['month'] = future_df['departure'].dt.month
    future_df['day'] = future_df['departure'].dt.day
    future_df['days_until_flight'] = (future_df['departure'] - pd.Timestamp.now()).dt.days
    
    return future_df

def create_price_trend_chart(df):
    """
    Create an interactive price trend chart
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['departure'],
        y=df['predicted_price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#1f77b4'),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Price Trend Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def main():
    st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")
    
    st.title("✈️ Flight Price Predictor")
    st.markdown("Predict future flight prices using historical data and machine learning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        origin = st.text_input("Origin Airport Code", "JFK").upper()
    with col2:
        destination = st.text_input("Destination Airport Code", "LAX").upper()
    with col3:
        start_date = st.date_input("Start Date", min_value=datetime.today())
    
    if st.button("🔍 Predict Prices"):
        if len(origin) != 3 or len(destination) != 3:
            st.error("Please enter valid 3-letter airport codes")
            return
            
        with st.spinner("Analyzing flight patterns..."):
            try:
                # Fetch historical data
                df = fetch_and_process_data(origin, destination, start_date, None)
                
                if df.empty:
                    st.error("No flight data available for the selected route")
                    return
                
                # Prepare data and train model
                X, y = prepare_features(df)
                model = train_model(X, y)
                
                if model is None:
                    st.error("Unable to train prediction model")
                    return
                
                # Generate and predict future prices
                future_df = generate_future_dates(start_date)
                future_df['airline_encoded'] = df['airline_encoded'].mode()[0]  # Use most common airline
                
                features = ['day_of_week', 'month', 'day', 'days_until_flight', 'airline_encoded']
                future_df['predicted_price'] = model.predict(future_df[features])
                
                # Display visualizations
                st.subheader("📈 Price Forecast")
                fig = create_price_trend_chart(future_df)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("💰 Best Days to Book")
                best_days = future_df.nsmallest(5, 'predicted_price')
                
                fig_best_days = go.Figure()
                fig_best_days.add_trace(go.Bar(
                    x=best_days['departure'].dt.strftime('%Y-%m-%d'),
                    y=best_days['predicted_price'],
                    text=['${:,.2f}'.format(x) for x in best_days['predicted_price']],
                    textposition='auto',
                    marker_color='#1f77b4'
                ))
                
                fig_best_days.update_layout(
                    title="Top 5 Cheapest Days to Fly",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    showlegend=False,
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_best_days, use_container_width=True)
                
                # Display statistics
                st.subheader("📊 Price Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Price", f"${future_df['predicted_price'].mean():.2f}")
                with col2:
                    st.metric("Minimum Price", f"${future_df['predicted_price'].min():.2f}")
                with col3:
                    st.metric("Maximum Price", f"${future_df['predicted_price'].max():.2f}")
                
            except Exception as e:
                logging.error(f"Error in main execution: {str(e)}")
                st.error("An error occurred while processing your request. Please try again.")

if __name__ == "__main__":
    main()
