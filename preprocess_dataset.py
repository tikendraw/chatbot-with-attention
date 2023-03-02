import os
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding
import tensorflow_text as tf_text
import pickle



# Load data
data = pd.read_csv('./dataset.csv',sep = '\t',encoding='latin1', names = ['col1','col2'])
data1 = pd.read_csv('./human_chat_dataset/human_dataset.csv',encoding='latin1')

# preprocessing text
def tf_lower_and_split_punct_en(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿|]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


# vectorization
vectorizer = TextVectorization(output_sequence_length=MAX_OUTPUT_LENGTH, standardize=tf_lower_and_split_punct_en)


# adapt to vectorizer
all_words_here = pd.concat([data['col1'],data['col2']], axis = 0)

vectorizer.adapt(all_words_here)

# Embedding

# Reading glove embedding
zi = zipfile.ZipFile('./embedding/glove.6B.50d.zip')
zi.extractall('./embedding/')
zi.close()

embeddings_index = {}
with open('./embedding/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print("Glove Loded!")

word_dict = {word:i for i, word in enumerate(vectorizer.get_vocabulary())}


# Create matrix that holds words that occour together

embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index), embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


embedding_matrix = embedding_matrix_creater(50, word_index=word_dict)    

VOCAB_SIZE = vectorizer.vocabulary_size()

embedding = Embedding(VOCAB_SIZE, 
                  50, 
                  
                  input_length=13,
                  trainable=True)

embedding.build((None,))
embedding.set_weights([embedding_matrix])


# saving embedding_matrix for further use
np.save('./embedding/embedding_matrix.npy',embedding_matrix, allow_pickle=True)



# Pickle the config and weights

os.makedirs('components', exist_ok=True)
pickle.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}
            , open("./components/vectorizer.pkl", "wb"))

# make vector
def make_vector(x, y):
    x = vectorizer(x)
    y = vectorizer(y)

    x = x[:-1]
    y_in = y[:-1]
    y_out = y[1:]
    return (x,y_in),y_out

train_data = train_data.map(make_vector).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(make_vector).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


save_train_data_path = './dataset/train/'
save_test_data_path = './dataset/test/'

# # save the train_data and test_data
train_data.save(save_train_data_path, compression='GZIP')
test_data.save(save_test_data_path, compression='GZIP')

