import pickle 
import string
import streamlit as st
import webbrowser

global Lrdetect_Model

pickle_file = open('model.pckl', 'rb')
lr_model = pickle.load(pickle_file)
pickle_file.close()
st.title("Language detection Tool")
input_test = st.text_input("provide your text input here", "Hello my name is Jay")

button_clicked = st.button("Get Language Name")
if button_clicked:
    text = lr_model.predict([input_test])
    no_punct = ''.join(char for char in text if char not in string.punctuation)
    st.write(f"""### {no_punct} """)

