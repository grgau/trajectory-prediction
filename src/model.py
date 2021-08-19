import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, number_of_codes, encoder_units, dropout):
    super(Encoder, self).__init__()
    self.number_of_codes = number_of_codes
    self.encoder_units = encoder_units
    self.dropout = dropout

    self.lstm = tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_initializer='glorot_uniform')
    self.apply_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x[0], [1,0,2]) * tf.expand_dims(x[1], axis=-1))

  def call(self, input, mask, state=None):
    lstm_output, state_h, state_c = self.lstm(input, initial_state=state)
    output = self.apply_mask([lstm_output, mask])
    return output, state_h, state_c

class Decoder(tf.keras.layers.Layer):
  def __init__(self, number_of_codes, decoder_units, dropout, epsilon):
    super(Decoder, self).__init__()
    self.number_of_codes = number_of_codes
    self.decoder_units = decoder_units
    self.dropout = dropout

    self.lstm = tf.keras.layers.LSTM(self.decoder_units, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_initializer='glorot_uniform')
    self.apply_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x[0], [1,0,2]) * tf.expand_dims(x[1], axis=-1))
    self.dense = tf.keras.layers.Dense(self.number_of_codes, activation='relu', activity_regularizer=tf.keras.regularizers.l2(epsilon))

  def call(self, input, mask, state):
    lstm_output, state_h, state_c = self.lstm(input, initial_state=state)
    output = self.apply_mask([lstm_output, mask])
    # output = tf.reshape(output, (-1, output.shape[2])) # For sparse loss?
    output = self.dense(output)
    return output, state_h, state_c

class EncoderDecoder(tf.keras.layers.Layer):
  def __init__(self, encoder, decoder):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def call(self, input, mask):
    input = tf.transpose(input, [1,0,2]) # Make input to be (batch_size x sequence_length x number_of_codes)
    enc_output = self.encoder(input, mask)
    enc_states = enc_output[1:]
    dec_states = enc_states
    output = self.decoder(input, mask, dec_states)
    output = tf.nn.softmax(output)

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    return output, variables


# class BahdanauAttention(tf.keras.layers.Layer):
#   def __init__(self, attention_units):
#     super().__init__()
#     self.W1 = tf.keras.layers.Dense(attention_units, use_bias=False)
#     self.W2 = tf.keras.layers.Dense(attention_units, use_bias=False)

#     self.attention = tf.keras.layers.AdditiveAttention()

#   def call(self, query, value, mask):
#     w1_query = self.W1(query)
#     w2_key = self.W2(value)

#     query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
#     value_mask = mask

#     context_vector, attention_weights = self.attention(
#         inputs = [w1_query, value, w2_key],
#         mask=[query_mask, value_mask],
#         return_attention_scores = True,
#     )
#     return context_vector, attention_weights

