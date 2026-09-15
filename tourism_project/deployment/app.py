import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_Tourism-Project_model_v1.joblib",
)
CLASSIFICATION_THRESHOLD = 0.45

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️")
st.title("Tourism Package Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the tourism package.")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("The trained model file is missing. Please run the training pipeline first.")
    st.stop()

Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"],
)
Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"],
)
NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 5, 1)
Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "AVP", "VP", "Sr. Manager"],
)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= CLASSIFICATION_THRESHOLD)

    if prediction == 1:
        st.success("Prediction: Customer is likely to purchase the tourism package.")
    else:
        st.info("Prediction: Customer is unlikely to purchase the tourism package.")

    st.metric("Purchase probability", f"{probability:.1%}")
    st.caption(f"Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}")
