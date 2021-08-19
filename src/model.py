import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, number_of_codes, encoder_units, dropout):
    super(Encoder, self).__init__()
    self.number_of_codes = number_of_codes
    self.encoder_units = encoder_units
    self.dropout = dropout

    self.lstm = tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_initializer='glorot_uniform')
    self.attention = BahdanauAttention(self.encoder_units)
    self.apply_mask = tf.keras.layers.Lambda(lambda x: tf.transpose(x[0], [1,0,2]) * tf.expand_dims(x[1], axis=-1))
    self.dense = tf.keras.layers.Dense(self.number_of_codes, activation='relu')

  def call(self, input, mask, state=None):
    input = tf.transpose(input, [1,0,2]) # Make input to be (batch_size x sequence_length x number_of_codes)

    lstm_output, _, _ = self.lstm(input, initial_state=state)
    output = self.apply_mask([lstm_output, mask])
    output = self.dense(output)
    output = tf.nn.softmax(output)

    return output

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, attention_units):
    super().__init__()
    self.W1 = tf.keras.layers.Dense(attention_units, use_bias=False)
    self.W2 = tf.keras.layers.Dense(attention_units, use_bias=False)

    self.attention = tf.keras.layers.AdditiveAttention()

  def call(self, query, value, mask):
    w1_query = self.W1(query)
    w2_key = self.W2(value)

    query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
    value_mask = mask

    context_vector, attention_weights = self.attention(
        inputs = [w1_query, value, w2_key],
        mask=[query_mask, value_mask],
        return_attention_scores = True,
    )
    return context_vector, attention_weights

