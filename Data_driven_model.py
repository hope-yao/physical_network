import tensorflow as tf
slim = tf.contrib.slim
lstm = tf.contrib.rnn.LSTMCell

def my_rnn(y_true, num_layers, n_times_latentdim=1, training=True):
    batch_size, time_steps, y_dim = y_true.get_shape().as_list()
    latent_dim = y_dim *n_times_latentdim
    y_pred = y_true[:, 0:1, :]  # initial predictions
    for t in range(time_steps - 1):

        if training:
            y_t = y_true[:, t, :]
            # y_t = y_pred[:, -1, :]
        else:
            y_t = y_pred[:, -1, :]

        if training and t==0:
            flag = None
        else:
            flag = True

        if t==0:
            h_t = tf.zeros((batch_size,latent_dim))

        with tf.variable_scope("RNN_block", reuse=flag) as testing:
            h_t = slim.fully_connected(tf.concat([y_t,h_t],1), latent_dim, activation_fn=tf.nn.relu)
            y_t = slim.fully_connected(h_t, y_dim, activation_fn=None)
        y_pred = tf.concat([y_pred, tf.expand_dims(y_t, 1)], 1)

        # with tf.variable_scope("RNN_block", reuse=flag) as testing:
        #     for t in range(num_layers - 1):
        #         y_t = slim.fully_connected(y_t, latent_dim * n_times_latentdim, activation_fn=tf.nn.relu)
        #     y_tp1 = slim.fully_connected(y_t, latent_dim, activation_fn=None)
        # y_pred = tf.concat([y_pred, tf.expand_dims(y_tp1, 1)], 1)
    return y_pred


def my_lstm(y_true, num_layers=1, n_times_latentdim=1, training=True):
    batch_size, time_steps, latent_dim = y_true.get_shape().as_list()

    def lstm_cell(lstm_size):
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(latent_dim*n_times_latentdim) for _ in range(num_layers)])

    initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
    y_pred = []
    for t in range(time_steps):
        if training:
            y_t = y_true[:, t, :]
        else:
            y_t = y_pred[:, -1, :]
        if training and t==0:
            flag = None
        else:
            flag = True
        with tf.variable_scope("LSTM_block", reuse=flag) as testing:
            y_t = y_pred[:, -1, :]
            y_tp1, state = stacked_lstm(y_t, state)
            y_pred = tf.concat([y_pred, tf.expand_dims(y_tp1, 1)], 1)

    return y_pred