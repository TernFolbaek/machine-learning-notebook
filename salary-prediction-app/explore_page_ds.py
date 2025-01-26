import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


#executed once, it will be saved since the output will be the same, better performance
@st.cache_data

# this function cleans the data
def load_data():
    df = pd.read_csv("ds_salaries.csv")
    df = df[["job_title", "experience_level", "employee_residence","salary_in_usd"]]
    df = df.dropna()
    df = df.rename({"salary_in_usd": "salary"}, axis=1)


    residence_map = shorten_categories(df.employee_residence.value_counts(), 70)
    # change data frame country, where it only keeps the countries with sufficient 
    # data which is now in the variable "country_map"
    df['employee_residence'] = df['employee_residence'].map(residence_map)
    df = df[df["salary"] <= 320000]
    df = df[df["salary"] >= 10000]
    # save new data-frame where employee residence has no value of other
    df = df[df['employee_residence'] != 'Other']

    desired_job_titles = ["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer", "Analytics Engineer "]
    df = df[df['job_title'].isin(desired_job_titles)]
    return df

df = load_data()

def show_explore_page_ds():
    st.title("Explore Data Scientist Salaries")

    st.write(
        """### Stack Overflow Developer Survey 2020"""
    )

    # accessing values of country category in data-frame
    data = df["employee_residence"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal") # Equal aspect ration ensures that pie is drawn as a circle
    st.write("""#### Number of data from different residences""")
    st.pyplot(fig1)

    st.write("""#### Mean Salary Based On Residence""")
    # group the df by the country, then access the salary for the country, and get the mean value from all salaries of that country 
    # and at last sort it in ascending order
    data = df.groupby(["employee_residence"])["salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""#### Mean Salary Based On The Position""")
    data = df.groupby(["job_title"])["salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

