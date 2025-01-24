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


st.title('Next Word Prediction with LSTM')

input_text = st.text_input('Enter the sequence of words','To be or not to be')

if st.button('Predict next word'):
    max_seq_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_seq_len)

    st.write(f'Predicted word: {next_word}')