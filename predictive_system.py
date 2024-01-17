import pickle
import numpy as np
import sklearn.preprocessing

scaler = sklearn.preprocessing.StandardScaler()

#loading model
loaded_model = pickle.load(open('D:\\breast cancer\\trained_model.sav', 'rb'))

input_data = (9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,
              0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773)

input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

input_data_scaled = scaler.fit_transform(input_data_as_numpy_array)

prediction = loaded_model.predict(input_data_scaled)

if (prediction == 0):
    print("Cell found to be Benign i.e. non-cancerous")
else:
    print("Cell found to be Malignant i.e. cancerous")
