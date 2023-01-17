import re
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
import spacy
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpaperaccess.com/full/1325192.jpg");
             background-attachment: fixed;
	     background-position: 25% 75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title('Model Deployment: Classification')
st.header('Input Product Description and Product Type')


url = 'https://github.com/Suraj-23/Sentiment-Analysis-on-Amazon-Products/blob/main/Product_details.csv?raw=true'
train = pd.read_csv(url)

inputpd = st.text_input('Product Description', 'Umm. Hello!! What about Android?! RT @mention Awesome new version of @mention for iPhone and new web site coming after #sxsw.')
inputpc = st.selectbox("Select any 1 from list",(0,1,2,3,4,5,6,7,8,9))

train.loc[len(train.index)] = [1234,inputpd, inputpc, 2]

def depure_data(data):
    
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    data = re.sub('@\S*\s?', '', data)
    data = re.sub('\S*#\S*\s?', '', data)
    data = re.sub('#\S*\s?', '', data)
    data = re.sub('&\S*\s?', '', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
        
    return data

temp = []
#Splitting pd.Series to list
data_to_list = train['Product_Description'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        

data_words = list(sent_to_words(temp))

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))

data = np.array(data)

y = np.array(train['Sentiment'])
labels = tf.keras.utils.to_categorical(y, 4, dtype="float32")
del y

x = np.array(train['Product_Type'])
category = tf.keras.utils.to_categorical(x, 10, dtype="float32")
del x

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
max_words = 5000
max_len = 200


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
reviews = pad_sequences(sequences, maxlen=max_len)

# add category to reviews here
reviews = np.append(reviews,category, axis=1)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(reviews,labels,test_size=1, train_size=6364, random_state=42)

#Let's load the best model obtained during training
best_model = keras.models.load_model("best_model1.hdf5")

predictions = best_model.predict(X_test)


st.header('The predicted sentiment is:  {}'.format(np.argmax(predictions)))
