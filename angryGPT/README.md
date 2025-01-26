<<<<<<< HEAD
# angryGPT
GPT that responds back aggressively 
=======
## Resentful Bot

The objective of this project is to investigate the retort of a chatbot that is trained on a pessimistic dataset to respond to daily inquiries and questions.

### Files & their Purpose

#### `sentiment_detection.py`:
This file is used to train the model that detects the polarity of a textual message. It utilizes `twitter_training.csv` as its data, which has a column dedicated to its text and the corresponding value of either positive, negative, or neutral.

#### `angryGPT.py`:
This file is dedicated to training the chatbot itself. It imports the previous model from `sentiment_detection.py` using "model.pkl". It also imports the data-set `utterances.jsonl` and converts it to a CSV file named `provoking_file.ipynb`. In the file, we import certain Hugging Face Transformers to deal with the NLP training. Finally, we export our new-found model to `my_model.pkl`.

#### `app.py`:
This file runs the project in the browser using the well-known machine-learning/data-science library named Streamlit. The reason we have a "venv" is that the versions of certain libraries were not compatible with others. If you clone this repository and run it yourself, keep in mind that we used Google Colab to run `angryGPT.py` since our personal computers did not have the adequate power to run it. To run the project, prompt the terminal with "streamlit run app.py".
>>>>>>> origin/main
