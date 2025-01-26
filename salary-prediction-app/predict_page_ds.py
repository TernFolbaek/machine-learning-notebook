import streamlit as st
import pickle
import numpy as np

# function to save model into variable
def load_model():
    with open('saved_steps_ds.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
emp_residence = data["emp_residence"]
exp_level = data["exp_level"]
job_position = data["job_position"]

# following function creates our streamlit application
def show_predict_page_ds():
    st.title("Data Scientist Positions Salary Prediction")

    # """### {creates h3 tag} """
    st.write("""### We need some information to predict the salary""")

    # tuple with choice of countries and following the choice of educations

    residences = (
        "US",
        "GB",
        "CA",
        "ES",
        "IN",
    )

    levels = (
        "MI",
        "SE",
        "EN",
        "EX"
    )

    positions = (
        'Data Engineer',  'Analytics Engineer',
       'Machine Learning Engineer', 'Data Scientist', 'Data Analyst'
    )

    # the country the user selects will be saved in the variable country

    residence = st.selectbox("residence", residences)
    job_level = st.selectbox("Your job rank", levels)
    position = st.selectbox("Job Position", positions)

    # if we dont click on the button then it remains false, if clicked it becomes true
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[position, job_level, residence ]])
        # we then transform the info into ints which is then cast into floats
        X[:, 0] = job_position.transform(X[:,0])
        X[:, 1] = exp_level.transform(X[:,1])
        X[:, 2] = emp_residence.transform(X[:,2])


        X = X.astype(float)
    
        salary = regressor.predict(X)

        # print the salary with 2 decimals, the salary variable is a np array, which is why we want to access the first value
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")