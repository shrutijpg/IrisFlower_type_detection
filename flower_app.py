import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
#from urllib.parse import quote_plus

# MongoDB setup
#username = quote_plus("shrutibh1001")
#password = quote_plus("shruti@123")

#mongodb setup
uri = "mongodb+srv://shrutibh1001:shruti1234@cluster0.lvxhkfu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['iris_prediction']
collection = db["iris_logs"]

# Defining the mapping for the numerical predictions of the output
species_mapping = {0:"Iris-Setosa",1:"Iris-versicolor",2:"Iris-virginica"}
#Model Loading based on our selection

def load_model(model_key):
    file_map = {
        "SVM (Binary)":"svm_binary.pkl",
        "SVM (Multi-class)":"svm_multi.pkl",
        "Logistic Regression (Binary)": "logistics_binary.pkl",
        "Logistic Regression (OvR)" : "logistics_ovr.pkl",
        "Logistic Regression (Multinomial)":"logistics_multinomial.pkl"
    }
    model = joblib.load(file_map[model_key])
    return model

def predict_iris(input_data,model):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

#Streamlit app interface
def main():
    st.title("🌸 Iris Flower Classifier ")
    st.write("Choose a model and input flower details to predict the Iris species.")

    #Input features of the iris flower
    #sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)

    sepal_length = st.number_input("sepal length (cm)",0.0,10.0,5.1)
    sepal_width = st.number_input("sepal width (cm)",0.0,10.0,3.5)
    petal_length = st.number_input("petal length (cm)",0.0,10.0,1.4)
    petal_width = st.number_input("petal length (cm)",0.0,10.0,0.2)



    # Now we will select the model
    model_options = [
        "SVM (Binary)",
        "SVM (Multi-class)",
        "Logistic Regression (Binary)",
        "Logistic Regression (OvR)" ,
        "Logistic Regression (Multinomial)"
    ]



    selected_model_name = st.selectbox("Select Model", model_options)


    if st.button("Predict"):
        input_data = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)":sepal_width,
            "petal length (cm)":petal_length,
            "petal width (cm)":petal_width
        }

        model = load_model(selected_model_name)
        prediction= predict_iris(input_data,model)

        #Mapping the numerical predicted value to the type name
        predicted_species = species_mapping.get(prediction)

        if predicted_species:
            st.success(f"🔍 Predicted Iris Species: **{predicted_species}**")
        else:
            st.error("Error:could not map")


        #Saving the prediction to the mongoDB
        input_data["Predicted Species"] = prediction
        input_data["Model Used"] = selected_model_name
        collection.insert_one(input_data)

if __name__ == "__main__":
    main()



