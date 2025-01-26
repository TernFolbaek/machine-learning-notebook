import streamlit as st
from predict_page_sde import show_predict_page_sde
from explore_page_sde import show_explore_page_sde
from explore_page_ds import show_explore_page_ds
from predict_page_ds import show_predict_page_ds
# Select boz where "Predict", "Explore" are the options
page = st.sidebar.selectbox("Explore Or Predict", ("Predict SDE Salary", "Predict DS Salary", "Explore SDE", "Explore DS"))

print(page)

if page == "Predict SDE Salary":
    show_predict_page_sde()
elif page == "Predict DS Salary":
    show_predict_page_ds()
elif page == "Explore SDE":
    show_explore_page_sde()
else:
    show_explore_page_ds()

