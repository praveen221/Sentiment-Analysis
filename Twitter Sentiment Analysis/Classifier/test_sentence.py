from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,SimpleRNN
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation
from keras.layers.convolutional import MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import imdb
from keras.models import model_from_json
import cPickle as pkl
import pickle
import dill
import os
dill.dumps(os)

import cPickle as pkl

from collections import OrderedDict
from nltk.corpus import stopwords

import glob
import re
import string


def extract_words(sentences):
    result = []
    stop = stopwords.words('english')
    trash_characters = '?.,!:;"$%^&*()#@+/0123456789<>=\\[]_~{}|`'
    trans = string.maketrans(trash_characters, ' '*len(trash_characters))

    for text in sentences:
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = text.replace('<br />', ' ')
        text = text.replace('--', ' ').replace('\'s', '')
        text = text.translate(trans)
        text = ' '.join([w for w in text.split() if w not in stop])

        words = []
        for word in text.split():
            word = word.lstrip('-\'\"').rstrip('-\'\"')
            if len(word)>2:
                words.append(word.lower())
        text = ' '.join(words)
        result.append(text.strip())
    return result

def grab_sentence(tweet, dictionary):
    sentences = [tweet]
    sentences = extract_words(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    #path = dataset_path
    
    pickle_in = open("dict_feat.pickle","rb")
    dictionary = pickle.load(pickle_in)
    

    #train_x_pos = grab_data('pos/matrix_train4_unigram', dictionary)
    #train_x_neg = grab_data('neg/matrix_train4_unigram', dictionary)
    #train_x_neu = grab_data('neu/matrix_train4_unigram', dictionary)
    #train_x = train_x_pos + train_x_neg + train_x_neu
    #train_y = [[1,0,0]] * len(train_x_pos) + [[0,1,0]] * len(train_x_neg) + [[0,0,1]] * len(train_x_neu)

    max_features = 20000
    maxlen = 100
    batch_size = 32
    
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    
    # load json and create model
    json_file = open('model_CNN_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_CNN_v2.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    result=[]
    x_test = grab_sentence('I am with Narendra Modi', dictionary)
    x_test += grab_sentence('Narendra is feku', dictionary)
    result=loaded_model.predict(x_test,verbose=0)
    for ii,i in enumerate(result):
        if i[0] > i [1]:
            if i[0] > i[2]:
                print("Positive")
            else:
                print("Neutral")
        else:
            if i[1] > i[2]:
                print("Negative")
            else:
                print("Neutral")
    

if __name__ == '__main__':
    main()
