import pickle

import streamlit as st
# from flasgger import Swagger
# from flask import Flask

# app = Flask(__name__)
# Swagger(app)

pickle_in = open("linreg.pkl", "rb")
regressor = pickle.load(pickle_in)


# @app.route('/')
def welcome():
    return "Welcome All"



# @app.route('/predict',methods=["Get"])
def predict_rbc(age, sex, pcv, mcv, mch, mchc, rdw, tlc, plt, hgb):
    """Predict RBC for from input features
    This is using docstrings for specifications.
    ---
    parameters:
      - name: age
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: pcv
        in: query
        type: number
        required: true
      - name: mcv
        in: query
        type: number
        required: true
      - name: mch
        in: query
        type: number
        required: true
      - name: mchc
        in: query
        type: number
        required: true
      - name: rdw
        in: query
        type: number
        required: true
      - name: tlc
        in: query
        type: number
        required: true
      - name: plt
        in: query
        type: number
        required: true
      - name: hgb
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    prediction = regressor.predict([[age, sex, pcv, mcv, mch, mchc, rdw, tlc, plt, hgb]])
    print(prediction)
    return prediction


def main():
    st.title("Diagnosis Anemia with Predicting RBC levels")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Predict RBC ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.text_input("Age")
    sex = st.text_input("Sex")
    pcv = st.text_input("PCV")
    mcv = st.text_input("MCV",)
    mch = st.text_input("MCH")
    mchc = st.text_input("MCHC")
    rdw = st.text_input("RDW")
    tlc = st.text_input("TLC")
    plt = st.text_input("PLT")
    hgb = st.text_input("HGB")
    result = ""
    if st.button("Predict"):
        result = predict_rbc(age, sex, pcv, mcv, mch, mchc, rdw, tlc, plt, hgb)[0]
    st.success('RBC is {}'.format(result))
    # if st.button("About"):
    #     st.text("Lets LEarn")
    #     st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
