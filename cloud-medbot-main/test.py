import json
import numpy as np
from tensorflow import keras
import pickle
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load necessary files
with open("cloud-medbot-main/intents.json") as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model('cloud-medbot-main/chat_model_2')

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 200

def preprocess_text(text):
    # Load stop words
    stop_words = set(stopwords.words('english'))

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Lemmatize and remove stop words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    
    return ' '.join(words)

def get_response(user_input):
    is_available=False
    preprocessed_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=max_len)
    result = model.predict(padded_sequence)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]
    probability = np.max(result)

    for i in data['intents']:
        if i['tags'][0] == tag:
            is_available=True
            break
        else:
            is_available=False
    if is_available :
        return {'response': i['answer'], 'score': str(probability)}
    else:
        return {'response' : "sorry i cant" , 'score':str(0)}

# Streamlit interface
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

def display_chat(chat_messages):
    for sender, message in chat_messages:
        if sender == 'user':
            st.write(
                f'<div style="display:flex; align-items:center; flex-direction:row-reverse;">'
                f'<div style="margin-right: 10px;">'
                f'👤'
                f'</div>'
                f'<div style="background-color:#25D366; padding:10px; border-radius:10px;">'
                f'{message}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.write(
                f'<div style="display:flex; align-items:center;">'
                f'<div style="margin-right: 10px;">'
                f'🤖'
                f'</div>'
                f'<div style="background-color:#075E54; padding:10px; border-radius:10px;">'
                f'{message}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

def chat_interface(chat_messages):
    user_input = st.text_input('Query')
    if st.button('Ask'):
        if user_input:
            response = get_response(user_input)
            bot_response = response['response']
            bot_score = float(response['score'])
            # st.session_state.query=''

            if bot_score > 0.72:
                chat_messages.append(('user', user_input))
                chat_messages.append(('bot', bot_response))
            else:
                chat_messages.append(('user', user_input))
                chat_messages.append(('bot', bot_response))
        else:
            st.error("Please enter a query!")

        display_chat(chat_messages)

    if st.button('Clear chat'):
        st.session_state["chat_messages"] = []

    # display_chat(chat_messages)

if __name__ == '__main__':
    st.title("MedIQ ChatBot")
    chat_interface(st.session_state["chat_messages"])
