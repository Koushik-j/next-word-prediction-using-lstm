# Next word prediction using LSTM

## Project Description: Next Word Prediction Using LSTM
#### Project Overview:

This project aims to develop a deep learning model for predicting the next word in a given sequence of words. The model is built using Long Short-Term Memory (LSTM) networks, which are well-suited for sequence prediction tasks. The project includes the following steps:

1- Data Collection: We use the text of Shakespeare's "Hamlet" as our dataset. This rich, complex text provides a good challenge for our model.

2- Data Preprocessing: The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. The sequences are then split into training and testing sets.

3- Model Building: An LSTM model is constructed with an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the probability of the next word.

4- Model Training: The model is trained using the prepared sequences, with early stopping implemented to prevent overfitting. Early stopping monitors the validation loss and stops training when the loss stops improving.

5- Model Evaluation: The model is evaluated using a set of example sentences to test its ability to predict the next word accurately.

6- Deployment: A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.

## Required tools and libraries 
1. Python 
2. Pandas
3. Numpy
4. Scikit-learn
5. Streamlit
6. Tensorflow/keras
7. Scikeras
8. nltk

## Setting up and installation

1. Setup a virtual environment (optional but recommended)
```cmd
 pip install virtualenv
 virtualenv -p python3 file_name
 file_name\Scripts\activate
```

2. Install required dependencies
```cmd
pip install -r requirements.txt
```

3. Run the streamlit app
```cmd
streamlit run app.py
```

## View my Results

```
prediction-next-word.streamlit.app
```

## Screenshot 

![image](https://github.com/user-attachments/assets/7ab46930-2e71-4b98-bf1f-2ed331f5eea7)

