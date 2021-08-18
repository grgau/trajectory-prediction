import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, number_of_codes, encoder_units, dropout):
    super(Encoder, self).__init__()
    self.number_of_codes = number_of_codes
    self.encoder_units = encoder_units
    self.dropout = dropout

    # self.embedding = tf.keras.layers.Embedding(self.number_of_codes, self.embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_initializer='glorot_uniform')
    self.dense = tf.keras.layers.Dense(self.number_of_codes, activation='relu')
    self.apply_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x[0], [1,0,2]) * tf.expand_dims(x[1], axis=-1))

  def call(self, input, mask, state=None):
    input = tf.transpose(input, [1,0,2]) # Make input to be (batch_size x sequence_length x number_of_codes)

    lstm_output, _, _ = self.lstm(input, initial_state=state)
    output = self.apply_mask([lstm_output, mask])
    output = self.dense(output)
    output = tf.nn.softmax(output)

    return output
    # return output, state