import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, number_of_codes, embedding_dim, encoder_units):
    super(Encoder, self).__init__()
    self.number_of_codes = number_of_codes
    self.encoder_units = encoder_units
    self.embedding_dim = embedding_dim

    # self.model_input = tf.keras.layers.Input((None, self.number_of_codes))
    # self.embedding = tf.keras.layers.Embedding(self.number_of_codes, self.embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, return_state=False, recurrent_initializer='glorot_uniform')
    self.dense = tf.keras.layers.Dense(self.number_of_codes, activation='relu')

  def call(self, input, mask, state=None):
    mask = tf.transpose(mask, [1,0]) # Make mask to be (batch_size x sequence_length)
    mask = tf.expand_dims(mask, axis = -1) # Make mask to be (batch_size x sequence_length x 1)
    input = tf.transpose(input, [1,0,2]) # Make input to be (batch_size x sequence_length x number_of_codes)
    input = tf.math.multiply(input, mask)

    output = self.lstm(input, initial_state=state)
    output = self.dense(output)
    output = tf.nn.softmax(output)

    return output
    # return output, state