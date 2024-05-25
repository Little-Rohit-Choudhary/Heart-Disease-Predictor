import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('heart.pkl')

# Define the feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def predict_heart_disease(data_list):
    input_data = np.array(data_list).reshape(1, -1)
    test = pd.DataFrame(input_data, columns=feature_names)
    Prediction = model.predict(test)
    return "Patient has heart disease" if Prediction else "Patient has no heart disease"

# Streamlit app
st.title('Heart Disease Prediction')

st.sidebar.header('Input Features')

def get_user_input():
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.sidebar.selectbox('Sex', options={'Male': 1, 'Female': 0})
    cp = st.sidebar.selectbox('Chest Pain Type', options={'Typical': 0, 'Atypical': 1, 'Non-anginal': 2, 'Asymptomatic': 3})
    trestbps = st.sidebar.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options={'True': 1, 'False': 0})
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', options={'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2})
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', options={'Yes': 1, 'No': 0})
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', options={'Upsloping': 0, 'Flat': 1, 'Downsloping': 2})
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thalassemia', options={'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3})

    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # The values are already mapped correctly by the selectbox
    return user_data

user_input = get_user_input()

if st.button('Predict'):
    data_list = [user_input[feature] for feature in feature_names]
    result = predict_heart_disease(data_list)
    st.success(result)
