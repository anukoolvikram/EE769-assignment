import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import random

# Title of the web app
st.title('White Wine Quality Predictor (SVR)')

@st.cache_data # Caching with corrected decorator
def load_data():
  white_wine_data = pickle.load(open('data.pkl', 'rb'))
  return white_wine_data

# Load data
white_wine_data = load_data()
df = white_wine_data['df']
scaler = white_wine_data['scaler']
model = white_wine_data['model']

# Session state to store slider values
if 'slider_dict' not in st.session_state:
  st.session_state.slider_dict = {}

# Description
st.write('Enter the characteristics of the wine to predict its quality.')

# Function to create sliders with unique keys using enumerate
def create_slider(column, idx):
  min_val = df[column].min()
  max_val = df[column].max()
  step = round((max_val - min_val) / 100, 3)

  # Combine index and column name for a unique key
  key = f"slider_{idx}_{column}"

  if column not in st.session_state.slider_dict:
    st.session_state.slider_dict[column] = st.slider(column, key=key, min_value=min_val, max_value=max_val, step=step, value=random.uniform(min_val, max_val))
  else:
    st.session_state.slider_dict[column] = st.slider(column, key=key, min_value=min_val, max_value=max_val, step=step, value=st.session_state.slider_dict[column])

# Generate and display sliders
for idx, column in enumerate(df.columns[:-1]):
  create_slider(column, idx)

# Prediction button
if st.button('Predict'):
  # Transform features and make prediction
  features = [list(st.session_state.slider_dict.values())]
  features = scaler.transform(features)
  quality = model.predict(features)[0]
  # Display predicted quality with success message
  st.success(f'The predicted wine quality is {quality}')
