import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('next_word.h5')

with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)



def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    
    token_list = pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.set_page_config(
    page_title="Next Word Prediction", page_icon="🪄", layout="wide"
)

st.title('🪄 Next Word Prediction with LSTM')

input_text = st.text_input('Enter the sequence of words')

if st.button('Predict next word'):
    max_seq_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_seq_len)

    st.write(f'Predicted word: {next_word}')

st.write('| Note:')
st.write('This model is trained with the shakespeare dataset.So, it will predict words from the shakespeare dataset.')

st.markdown("---")

st.subheader('| Github:')
st.write('If you want some ideas regarding what text to enter, check out my github repo which has a text file of the shakespeare dataset.')