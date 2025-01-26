import streamlit as st
import pickle
import numpy as np

# function to save model into variable
def load_model():
    with open('saved_steps_sde.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# following function creates our streamlit application
def show_predict_page_sde():
    st.title("Software Developer Salary Prediction")

    # """### {creates h3 tag} """
    st.write("""### We need some information to predict the salary""")

    # tuple with choice of countries and following the choice of educations

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad"
    )

    # the country the user selects will be saved in the variable country

    country = st.selectbox("Country", countries)
    education = st.selectbox("Your education", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    # if we dont click on the button then it remains false, if clicked it becomes true
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience ]])
        # we then transform the info into ints which is then cast into floats
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
    
        salary = regressor.predict(X)

        # print the salary with 2 decimals, the salary variable is a np array, which is why we want to access the first value
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")