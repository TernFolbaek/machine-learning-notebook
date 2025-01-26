import streamlit as st
import pickle
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.title("Resentful Bot")

def load_model():
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

answer_count = st.slider("Word Count Reply", 25, 100)
input_text = st.text_input("Enter your text:")

if input_text and answer_count > 0:
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    generated_text = model.generate(
        input_ids, max_length=answer_count, num_return_sequences=1, do_sample=True)
    decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    st.markdown(f"### {decoded_text} ###")
else:
    st.write("Please enter some text.")


st.markdown(f"""
    <div 
    style="
    color:  lightblue;
    margin-left: 20px;
    font-style: italic;
    text-align: justify;">
        <h4>This chatbot is currently under development, seeking a more suitable dataset. We employed 
        a dataset to build a model that discerns negative/positive/neutral connotations in messages. 
        Leveraging this model, we extracted negative messages from a larger dataset, dividing it into 
        input and output halves. While this approach results in an AI with seemingly erratic responses, 
        it offers amusement through its diverse outputs.
        </h4>
    </div>
""", unsafe_allow_html=True)