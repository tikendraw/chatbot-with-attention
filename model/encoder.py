import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, 
    LSTM, 
    GRU, 
    Bidirectional, 

)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_vectorizer, units, embed_dims):
        super(Encoder, self).__init__()
        self.text_vectorizer =  text_vectorizer
        self.units = units
        self.embed_dims = embed_dims
        self.vocab_size = text_vectorizer.vocabulary_size()
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dims, mask_zero=True, )
        self.rnn = Bidirectional(merge_mode='concat', layer = GRU(self.units, return_sequences=True, return_state=True))
        
    def call(self, x, y=None, return_state=False):
        
        x = self.embedding(x)
        encoder_output, encoder_fw_state, encoder_bw_state = self.rnn(x)
        encoder_state = [encoder_fw_state, encoder_bw_state]
        
        if return_state:
            return encoder_output, encoder_state
        else:
            return encoder_output
        
    def convert_input(self, texts, return_state=False):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_vectorizer(texts)
        
        context = self(context, return_state = return_state)
        
        return context


def main():
	...

if __name__=="__main__":
	
	main()
