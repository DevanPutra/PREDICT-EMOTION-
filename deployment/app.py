import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
from wordcloud import WordCloud
from wordcloud import STOPWORDS

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up app title and header image
st.set_page_config(page_title='Emotion Prediction')
st.title('Emotion Prediction')
st.image('https://i.imgur.com/LV6X1ji.png', use_column_width=True)

# read CSV untuk memanggil data csv, dan missing value untuk memastikan bahwa tidak ada lagi uniq missing value
missing_values = ["n/a", "na", "--",' ?', 'inf']
data=pd.read_csv("NLP_Dataset.csv", na_values = missing_values, skipinitialspace=True)
df= data.copy()
df.head(10) #viewing the first 10 data

# "To remove duplicate columns, and keep only one column to prevent duplication."
df.drop_duplicates(subset='text', keep='first', inplace=True)

#"To create a word cloud for the words emotion."

st.title('Word Cloud')
# define emotions
emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# create subplots
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 18))

# loop through emotions and plot wordcloud in corresponding subplot
for i, emotion in enumerate(emotions):
    row = i // 2
    col = i % 2
    wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=STOPWORDS).generate(" ".join(df[df['emotion'] == emotion]['text']))
    axs[row, col].imshow(wc, interpolation='bilinear')
    axs[row, col].set_title(emotion.upper(), fontsize=30)
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()

st.pyplot(fig)


st.title('Lets Predict')
# Load the model
loaded_model = tf.keras.models.load_model('my_model')

# Initialize the WordNetLemmatizer
wnl = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing stop words, and removing non-alphabetic characters.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token in stop_words]
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Define a function to lemmatize each word in a sentence
def lemmatize_sentence(sentence):
    """
    Lemmatize each word in a sentence and return the lemmatized sentence.
    """
    tokenized_sentence = nltk.word_tokenize(sentence)
    lemmatized_sentence = [wnl.lemmatize(word, pos='v') for word in tokenized_sentence]
    return lemmatized_sentence

# Define a function to preprocess and lemmatize the text
def preprocess_and_lemmatize_text(text):
    """
    Preprocess the text by converting to lowercase, removing stop words, and removing non-alphabetic characters.
    Then, lemmatize each word in the preprocessed text and return the resulting string.
    """
    preprocessed_text = preprocess_text(text)
    lemmatized_text = ' '.join(lemmatize_sentence(preprocessed_text))
    return lemmatized_text

# Define the Streamlit app
def app():
    st.title('Emotion Detection')
    text = st.text_input('Enter some text:')
    if text:
        processed_text = preprocess_and_lemmatize_text(text)
        processed_text = [[processed_text]]
        predicted_class = loaded_model.predict(processed_text)
        label_dict = {0: 'sadness', 1: 'joy', 2: 'fear', 3: 'anger', 4: 'love', 5: 'surprise'}
        predicted_labels = np.argmax(predicted_class, axis=1)
        predicted_emotion = label_dict[predicted_labels[0]]
        # Add emoticons for each predicted emotion
        if predicted_emotion == 'sadness':
            emoticon = ':('
            color = 'blue'
        elif predicted_emotion == 'joy':
            emoticon = ':)'
            color = 'yellow'
        elif predicted_emotion == 'fear':
            emoticon = ':O'
            color = 'purple'
        elif predicted_emotion == 'anger':
            emoticon = '>_<'
            color = 'red'
        elif predicted_emotion == 'love':
            emoticon = '<3'
            color = 'pink'
        elif predicted_emotion == 'surprise':
            emoticon = ':o'
            color = 'green'
        
        st.markdown(f'<h1 style="color:{color};text-align:center">{predicted_emotion} {emoticon}</h1>', unsafe_allow_html=True)



        
if __name__ == '__main__':
    app()
