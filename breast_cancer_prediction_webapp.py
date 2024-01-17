import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('D:/breast cancer/trained_model.sav', 'rb'))

scaler = StandardScaler()

def prediction_function(input_data):

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    input_data_scaled = scaler.fit_transform(input_data_as_numpy_array)

    prediction = loaded_model.predict(input_data_scaled)

    if (prediction == 0):
        return "Cell found to be Benign i.e. non-cancerous"
    else:
        return "Cell found to be Malignant i.e. cancerous"

def main():
    #heading
    st.title("Breast Cancer Prediction WebApp")

    #inputs
    radius_mean = st.text_input("Enter Radius Mean")
    texture_mean = st.text_input("Enter Texture Mean")
    perimeter_mean = st.text_input("Enter Perimeter Mean")
    area_mean = st.text_input("Enter Area Mean")
    smoothness_mean = st.text_input("Enter Smoothness Mean")
    compactness_mean = st.text_input("Enter Compactness Mean")
    concavity_mean = st.text_input("Enter Concavity Mean")
    concave_points_mean = st.text_input("Enter Concave Points Mean")
    symmetry_mean = st.text_input("Enter Symmetry Mean")
    fractal_dimension_mean = st.text_input("Enter Fractal Dimension Mean")
    radius_se = st.text_input("Enter Radius Se")
    texture_se = st.text_input("Enter Texture Se")
    perimeter_se = st.text_input("Enter Perimeter Se")
    area_se = st.text_input("Enter Area Se")
    smoothness_se = st.text_input("Enter Smoothness Se")
    compactness_se = st.text_input("Enter Compactness Se")
    concavity_se = st.text_input("Enter Concavity Se")
    concave_points_se = st.text_input("Enter Concave Points Se")
    symmetry_se = st.text_input("Enter Symmetry Se")
    fractal_dimension_se = st.text_input("Enter Fractal Dimension Se")
    radius_worst = st.text_input("Enter Radius Worst")
    texture_worst = st.text_input("Enter Texture Worst")
    perimeter_worst = st.text_input("Enter Perimeter Worst")
    area_worst = st.text_input("Enter Area Worst")
    smoothness_worst = st.text_input("Enter Smoothness Worst")
    compactness_worst = st.text_input("Enter Compactness Worst")
    concavity_worst = st.text_input("Enter Concavity Worst")
    concave_points_worst = st.text_input("Enter Concave Points Worst")
    symmetry_worst = st.text_input("Enter Symmetry Worst")
    fractal_dimension_worst = st.text_input("Enter Fractal Dimension Worst")
    


    #precdiction code
    diagnosis = ''

    if st.button("Cancer Test Result"):
        diagnosis = prediction_function([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, 
                                        concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                                         texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, 
                                        symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
                                        area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                                        symmetry_worst, fractal_dimension_worst])
    
    st. success(diagnosis)

if __name__ == '__main__':
    main()
