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


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

#executed once, it will be saved since the output will be the same, better performance
@st.cache_data

# this function cleans the data
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)


    country_map = shorten_categories(df.Country.value_counts(), 400)
    # change data frame country, where it only keeps the countries with sufficient 
    # data which is now in the variable "country_map"
    df['Country'] = df['Country'].map(country_map)
    df = df[df["Salary"] <= 250000]
    df = df[df["Salary"] >= 10000]
    df = df[df['Country'] != 'Other']

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)

    return df

df = load_data()

def show_explore_page_sde():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """### Stack Overflow Developer Survey 2020"""
    )

    # accessing values of country category in data-frame
    data = df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal") # Equal aspect ration ensures that pie is drawn as a circle
    st.write("""#### Number of data from different countries""")
    st.pyplot(fig1)

    st.write("""#### Mean Salary Based On Country""")
    # group the df by the country, then access the salary for the country, and get the mean value from all salaries of that country 
    # and at last sort it in ascending order
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""#### Mean Salary Based On The Experience""")
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

