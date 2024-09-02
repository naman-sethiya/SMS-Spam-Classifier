import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    text = nltk.word_tokenize(text)

    # Remove punctuation and stopwords
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stop_words]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


# Load pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.title("SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # Preprocess input
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
