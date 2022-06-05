# (1) Imports
import os
import sys
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Model

##Phase 1

#(2) Load dataset
print('Processing text dataset')
articles = [] # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
Topic_ID = [] # list of label ids

#(2) Load dataset and giving every topic id & datafram
dataframe = pd.read_csv("news-article-categories.csv")
X_data=dataframe.iloc[:,[2]]
Y_data=dataframe.iloc[:,[0]]
news_labels = dataframe['category'].unique()

for i in news_labels:
    label_id = len(labels_index)
    labels_index[i] = label_id

def condition(x):
    if x == "BUSINESS":
        return 0
    elif x == "POLITICS":
        return 1
    elif x == "SPORTS":
        return 2
    else:
        return 3
dataframe["Topic_ID"] = dataframe["category"].apply(
    condition
)
Topic_ID = list(dataframe["Topic_ID"])

#(3) Process Text samples
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 40000

#converting to list of articles
texts_list = X_data.values.tolist()
texts=[]
for i in texts_list:
    texts.append(i[0])
#print(texts[0])

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index # the dictionary
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# (4) format output of the CNN (the shape of labels)
labels_matrix = to_categorical(np.asarray(Topic_ID)) #matrix of every article corspnding to its topic column

#(5) Split samples and labels to training and testing sets
TEST_SPLIT = 0.2
indices = np.arange(data.shape[0]) #array of indices [0, 1, .....,1502]
np.random.shuffle(indices)
data_shuffled = data[indices]
labels_shuffled = labels_matrix[indices]
nb_test_samples = int(TEST_SPLIT * data_shuffled.shape[0])
x_train = data_shuffled[:-nb_test_samples]
y_train = labels_shuffled[:-nb_test_samples]
x_test = data_shuffled[-nb_test_samples:]
y_test = labels_shuffled[-nb_test_samples:]


# (6) Read Glove Word Embeddings
EMBEDDING_DIM = 100
embeddings_index = {}
glove_file_path = os.path.join('glove.6B\\', 'glove.6B.100d.txt')
with open(glove_file_path, encoding='UTF8') as f:
    for line in f:
        values = line.split(sep=' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# (7) Map the dataset dictionary of (words,IDs) to a matrix of the
# embeddings of each word in the dictionary
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))#+1 to include the zeros vector for non-existing words
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# (8.1) Embedding Layer
embedding_layer = Embedding(len(word_index) + 1, #vocab size
EMBEDDING_DIM, #embedding vector size
weights=[embedding_matrix],
#weights matrix
input_length=MAX_SEQUENCE_LENGTH, #padded sequence length
trainable=False)

# (8.2) Build 1D CNN Layers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D(35)(x) # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

# (8.3) Build, Compile, and Run the model
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 5)

# (8.4) Evaluate the model
print('Acuracy on testing set:')
model.evaluate(x_test,y_test)

#(9) Use the model for prediction
model.predict(x_test)

"""------------------------------------------------------------------------------"""
##Phase 2
n = int(input("Enter number of tests: "))
for i in range(n):
    test_str = input("Enter the post: ")
    texts = []
    texts.append(test_str)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_vec = model.predict(data.reshape(1,-1))
    label_id = np.argmax(label_vec)
    label_name = ''
    for name, ID in labels_index.items(): # for name, age in dictionary.iteritems():
        if label_id == ID:
            label_name = name
            break
    print ('The category of article is %s' %(label_name))