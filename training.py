import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
from model.chatbot import ChatBot
from model.utils import masked_acc, masked_loss
from tensorflow.keras.layers import TextVectorization

import tensorflow_text as tf_text
import pickle
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger
import zipfile


print('GPU Avaliable: ', gpu:=len(tf.config.list_physical_devices('GPU')))


MAX_OUTPUT_LENGTH = 200
BATCH_SIZE = 32
UNITS = 64
EMBEDDING_DIMS = 50

# Vectorizer
# Loading vectorizer
from_disk = pickle.load(open("./components/vectorizer.pkl", "rb"))
vectorizer = TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorizer.set_weights(from_disk['weights'])


# Embedding


zipf = zipfile.ZipFile('./embedding/embedding_matrix.zip')
zipf.extractall('./embedding/')
zipf.close()

# Loading embedding_matrix
embedding_matrix = np.load('./embedding/embedding_matrix.npy')


# Dataset

save_train_data_path = './dataset/train/'
save_test_data_path = './dataset/test/'

#loading the data
train_data = tf.data.Dataset.load(save_train_data_path, compression='GZIP')
test_data = tf.data.Dataset.load(save_test_data_path, compression='GZIP')




# initiate the model
model = ChatBot(vectorizer, UNITS, EMBEDDING_DIMS)
# compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005),
              loss=masked_loss, 
              metrics=[masked_acc, masked_loss])

EPOCHS = 40

CKPT_DIR = './model_checkpoint'
# CKPT_DIR = '/content/drive/MyDrive/tf_model/chatbot'
os.makedirs(CKPT_DIR, exist_ok = True)
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(CKPT_DIR,  f"{datetime.now().strftime('%m:%d:%Y, %H:%M:%S')}"),
    monitor= 'masked_acc',
    verbose= 0,
    save_best_only = True,
    save_weights_only = True,
    mode= 'auto',
    save_freq='epoch'
)

os.makedirs('log', exist_ok = True)
csv_logger = CSVLogger('./log/training.log')

# Train
history = model.fit(
    train_data.repeat(), 
    epochs=EPOCHS,
    steps_per_epoch = 50,
    validation_data=test_data,
    validation_steps = 2,
    callbacks=[
                # tf.keras.callbacks.EarlyStopping(patience=5),
                model_ckpt,
                csv_logger]
                )


class Export(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def reply(self, inputs):
        return self.model.reply(inputs)
    

inputs = [
    "It's really cold here.",
    "This is my life.",
    "His room is a mess"
]


export = Export(model)

_ = export.reply(tf.constant(inputs))


result = export.reply(tf.constant(inputs))

print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()

tf.saved_model.save(export, 'chatbot',
                    signatures={'serving_default': export.reply})


# reloaded = tf.saved_model.load('chatbot')
# _ = reloaded.reply(tf.constant(inputs)) #warmup