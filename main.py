import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    LSTM,
    GRU,
    Bidirectional,
    Dense,
)
import tensorflow_text as tf_text
import pickle
from datetime import datetime

MAX_OUTPUT_LENGTH = 102
BATCH_SIZE = 32
UNITS = 64
EMBEDDING_DIMS = 128

print("GPU Avaliable: ", gpu := len(tf.config.list_physical_devices("GPU")))


# preprocessing text function
def tf_lower_and_split_punct_en(text):
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


# Load the vectorizer
from_disk = pickle.load(open("./components/vectorizer.pkl", "rb"))
vectorizer = TextVectorization.from_config(from_disk["config"])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorizer.set_weights(from_disk["weights"])

# loading the data
save_train_data_path = "./dataset/train/"
save_test_data_path = "./dataset/test/"

train_data = tf.data.Dataset.load(save_train_data_path, compression="GZIP")
test_data = tf.data.Dataset.load(save_test_data_path, compression="GZIP")


# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_vectorizer, units, embed_dims):
        super(Encoder, self).__init__()
        self.text_vectorizer = text_vectorizer
        self.units = units
        self.embed_dims = embed_dims
        self.vocab_size = text_vectorizer.vocabulary_size()
        self.embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dims,
            mask_zero=True,
        )
        self.rnn = Bidirectional(
            merge_mode="concat",
            layer=GRU(self.units, return_sequences=True, return_state=True),
        )

    def call(self, x, y=None, return_state=False):

        x = self.embedding(x)
        encoder_output, encoder_fw_state, encoder_bw_state = self.rnn(x)
        encoder_state = [encoder_fw_state, encoder_bw_state]
        if return_state:
            return (encoder_output, encoder_state)
        else:
            encoder_output

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_vectorizer(texts)
        context = self(context)
        return context


# Context
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=units, num_heads=1, **kwargs
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):

        attn_output, attn_scores = self.mha(
            query=x, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


# Decoder
class Decoder(keras.layers.Layer):
    def __init__(self, text_vectorizer, units, embed_dims):
        super(Decoder, self).__init__()
        self.text_vectorizer = text_vectorizer
        self.units = units
        self.embed_dims = embed_dims
        self.vocab_size = text_vectorizer.vocabulary_size()

        self.embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dims,
            mask_zero=True,
        )
        self.rnn = LSTM(self.units, return_sequences=True, return_state=True)

        # self.attention =  tf.keras.layers.Attention()
        self.attention = CrossAttention(units)

        self.output_dense = Dense(self.vocab_size)

        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_vectorizer.get_vocabulary(),
            mask_token="",
            oov_token="[UNK]",
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_vectorizer.get_vocabulary(),
            mask_token="",
            oov_token="[UNK]",
            invert=True,
        )

        self.start_token = self.word_to_id("[START]")
        self.end_token = self.word_to_id("[END]")

    def call(self, x, context, state=None, return_state=False):
        """x, context, state=None, return_sequence=False"""

        x = self.embedding(x)

        decoder_output, decoder_state_h, decoder_state_c = self.rnn(
            x, initial_state=state
        )

        decoder_state = [decoder_state_h, decoder_state_c]
        x = self.attention(decoder_output, context)
        self.last_attention_weights = self.attention.last_attention_weights

        logits = self.output_dense(x)

        if return_state:
            return logits, decoder_state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=" ")
        result = tf.strings.regex_replace(result, "^ *\[START\] *", "")
        result = tf.strings.regex_replace(result, " *\[END\] *$", "")
        return result

    def get_next_token(self,
                       next_token,
                       context,
                       done,
                       state,
                       temperature=0.0):

        logits, state = self(
            next_token,
            context,
            state=state,
            return_state=True
            )

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        # If a sequence produces an `end_token`, set it `done`
        done = done | (next_token == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state


# Model
class ChatBot(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units, embed_dims):
        super().__init__()
        self.text_processor = text_processor
        self.units = units
        self.embed_dims = embed_dims

        # Build the encoder and decoder
        encoder = Encoder(text_processor, units, embed_dims)
        decoder = Decoder(text_processor, units, embed_dims)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(x, context)

        try:
            # Delete the keras mask, so keras doesn't scale the loss+accuracy.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


# Masked loss
def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# Masked Accuracy
def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)


# Intanciate a bot
model = ChatBot(vectorizer, UNITS, EMBEDDING_DIMS)

# Compile
model.compile(
    optimizer="adam",
    loss=masked_loss,
    metrics=[masked_acc, masked_loss]
    )


EPOCHS = 50

CKPT_DIR = "./model_checkpoint"

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(CKPT_DIR, f"{datetime.now().strftime('%m:%d:%Y, %H:%M:%S')}"),
    monitor="masked_acc",
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch",
)

# Train

# Train
history = model.fit(
    train_data.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=80,
    validation_data=test_data,
    validation_steps=5,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3), model_ckpt],
)


# Saving the Model
os.makedirs("saved_model")
model.save(
    f"./saved_model/model_{datetime.now().strftime('%m:%d:%Y, %H:%M:%S')}.h5"
    )
