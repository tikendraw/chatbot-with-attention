import argparse
import os
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import tensorflow_text as tf_text
import pickle
from sklearn.model_selection import train_test_split

# preprocessing text
def tf_lower_and_split_punct_en(text) -> str:
    # Split accented characters.
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, "[.?!,¿|]", r" \0 ")
    # Strip whitespace.
    text = tf.strings.strip(text)
    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
    return text


# Embedding
def glove_embedding() -> dict:
    # Reading glove embedding
    zi = zipfile.ZipFile("./embedding/glove.6B.50d.zip")
    zi.extractall("./embedding/")
    zi.close()

    embeddings_index = {}
    with open("./embedding/glove.6B.50d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        f.close()

    os.remove("./embedding/glove.6B.50d.txt")
    return embeddings_index


# Create matrix that holds words that occour together
def embedding_matrix_creater(EMBEDDING_DIMENSION, word_index):
    embeddings_index = glove_embedding()
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIMENSION))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix




def main(
    directory: str,
    batch_size: int = 32,
    max_output_length: int = 100,
    sep: str = ","
):
    # Load data
    data = pd.read_csv(directory, sep=sep, encoding="latin1")
    # data.columns = ["col1", "col2"]

    # Vectorize the data
    vectorizer = TextVectorization(
        output_sequence_length=max_output_length,
        standardize=tf_lower_and_split_punct_en,
    )
    # Adapt
    all_words_here = pd.concat([data["col1"], data["col2"]], axis=0)
    vectorizer.adapt(all_words_here)

    # make vector
    def make_vector(x, y):
        x = vectorizer(x)
        y = vectorizer(y)

        x = x[:-1]
        y_in = y[:-1]
        y_out = y[1:]
        return (x, y_in), y_out

    # Loading Embedding weights
    word_dict = {word: i for i, word in enumerate(vectorizer.get_vocabulary())}

    # Creating embedding matrix
    embedding_matrix = embedding_matrix_creater(
        EMBEDDING_DIMENSION, word_index=word_dict
    )

    # Saving embedding_matrix for further use
    np.save(
        "./embedding/embedding_matrix.npy",
        embedding_matrix,
        allow_pickle=True
        )
    # compressing
    zipfile.ZipFile("embedding_matrix.zip", mode="w").write(
        "./embedding/embedding_matrix.npy"
    )

    # Pickle the config and weights

    os.makedirs("components", exist_ok=True)
    pickle.dump({
        "config": vectorizer.get_config(),
        "weights": vectorizer.get_weights()
        },
        open("./components/vectorizer.pkl", "wb"),
    )
    xtrain, xtest, ytrain, ytest = train_test_split(data["col1"], data["col2"], test_size=.01, random_state= 44)
    train_data = tf.data.Dataset.from_tensor_slices((data["col1"], data["col2"]))
    #NOTE: purpose of these is data is not to test but just to predict some examples
    # i am aware that is has been included in training
    test_data = tf.data.Dataset.from_tensor_slices((xtest, ytest))

    train_data = train_data.map(make_vector).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.map(make_vector).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # save paths
    save_train_data_path = './dataset/train/'
    save_test_data_path = './dataset/test/'

    # # save the train_data and test_data
    train_data.save(save_train_data_path, compression='GZIP')
    test_data.save(save_test_data_path, compression='GZIP')


script_description = """
This Script creates training data from csv file
(csv file must have 2 columns only)

Embedding dimension = 50
"""


if __name__ == "__main__":

    EMBEDDING_DIMENSION = 50

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to csv dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_output_length", type=int, default=100, help="Max output length"
    )
    parser.add_argument("--separator", default=",", help="Separator")
    args = parser.parse_args()

    # Call main function with arguments
    main(args.directory, args.batch_size, args.max_output_length, args.separator)
