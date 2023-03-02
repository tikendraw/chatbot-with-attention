from model.decoder import Decoder
from model.encoder import Encoder
import tensorflow as tf


# Model
class ChatBot(tf.keras.Model):
    
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units, embed_dims, embedding_matrix):
        super().__init__()
        self.text_processor = text_processor
        self.units = units
        self.embed_dims = embed_dims
        
        # Build the encoder and decoder
        encoder = Encoder(text_processor, units, embed_dims, embedding_matrix)
        decoder = Decoder(text_processor, units, embed_dims, embedding_matrix)
        
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(x, context, training = True)

        #TODO(b/250038731): remove this
        try:
          # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


    def reply(self,
                texts, *,
                max_length=50,
                temperature=0.0):
        # Process the input texts
        context = self.encoder.convert_input(texts, return_state = True)

        context, state = context

        batch_size = tf.shape(texts)[0]

        # Setup the loop inputs
        tokens = []
        # attention_weights = []
        next_token, done, state_zero = self.decoder.get_initial_state(context)
        # state = state
        state =[state_zero,state_zero]
        for _ in range(max_length):
            # Generate the next token
            next_token, done, state = self.decoder.get_next_token(
                    next_token, context, done,  state, temperature)

            # Collect the generated tokens
            tokens.append(next_token)
            # attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Stack the lists of tokens and attention weights.
        tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
        return self.decoder.tokens_to_text(tokens)
