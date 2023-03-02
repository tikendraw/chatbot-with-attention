import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Embedding, 
    LSTM, 
    Dense, 
    Concatenate
)
from attention import AttentionLayer

# Decoder

class Decoder(keras.layers.Layer):
    def __init__(self, text_vectorizer, units,  embed_dims, embedding_matrix=None) :
        super(Decoder, self).__init__()
        self.text_vectorizer =  text_vectorizer
        self.units = units * 2
        self.embed_dims = embed_dims
        self.vocab_size = text_vectorizer.vocabulary_size()
        
        self.embedding = Embedding(input_dim=self.vocab_size , output_dim=self.embed_dims, mask_zero=True, trainable=False)
        self.embedding.build((None,))
        self.embedding.set_weights([embedding_matrix])
        
        self.rnn = LSTM(self.units, return_sequences=True, return_state=True)
        
        # self.attention =  tf.keras.layers.Attention()
        self.attention = AttentionLayer()
        
        self.output_dense = Dense(self.vocab_size)
        
        self.word_to_id = tf.keras.layers.StringLookup(vocabulary=text_vectorizer.get_vocabulary(), mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(vocabulary=text_vectorizer.get_vocabulary(), mask_token='', oov_token='[UNK]', invert=True)
        
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

    def call(self, x, context, state=None, return_state = False, training=False):
        ''' x, context, state=None, return_sequence=False '''
        
        x = self.embedding(x)
        
        decoder_output, decoder_state_h, decoder_state_c = self.rnn(x, initial_state=state)
        decoder_state = [decoder_state_h, decoder_state_c]
        
        # attention
        # attn_op= self.attention([context, decoder_output])
        attn_op, attn_state = self.attention([context, decoder_output]) # this is for custom AttentionLayer()

        x = Concatenate(axis=-1)([decoder_output, attn_op])
        # x = tf.multiply(decoder_output, attn_op)
        # x = self.attention(decoder_output, context)
        # self.last_attention_weights = self.attention.last_attention_weights

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
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result
    
    def get_next_token(self, next_token, context,  done, state, temperature = 0.0):
        
        logits, state = self(next_token, context, state = state, return_state=True, training = False) 

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :]/temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        # If a sequence produces an `end_token`, set it `done`
        done = done | (next_token == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state





























def main():
	...

if __name__=="__main__":
	
	main()
