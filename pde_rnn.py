import tensorflow as tf
slim = tf.contrib.slim
lstm = tf.contrib.rnn.LSTMCell

def my_rnn(x, y_0):
    batch_size, time_steps, y_dim = x.get_shape().as_list()
    state_dim = y_0.get_shape().as_list()[1]
    latent_dim = 4
    y_pred = [tf.expand_dims(y_0,1)]
    h_t_old = y_0#tf.zeros((batch_size, latent_dim))

    w0 = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev=0.1), name='w0')
    b0 = tf.Variable(tf.truncated_normal([latent_dim], stddev=0.1), name='b0')
    w1 = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.truncated_normal([latent_dim], stddev=0.1), name='b1')
    w2 = tf.Variable(tf.truncated_normal([state_dim + latent_dim, state_dim], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.truncated_normal([state_dim], stddev=0.1), name='b2')
    w3 = tf.Variable(tf.truncated_normal([state_dim, state_dim], stddev=0.1), name='w3')
    b3 = tf.Variable(tf.truncated_normal([state_dim], stddev=0.1), name='b3')

    for t in range(time_steps):
        h_t = tf.nn.relu( tf.nn.xw_plus_b(x[:, t, :], w0, b0) )
        h_t = tf.nn.relu( tf.nn.xw_plus_b(h_t, w1, b1) )
        y_t = tf.nn.relu( tf.nn.xw_plus_b(tf.concat([h_t, h_t_old],1), w2, b2) )
        y_t = tf.nn.relu( tf.nn.xw_plus_b(y_t, w3, b3))
        # y_t = slim.fully_connected(tf.concat([h_t, h_t_old],1), state_dim, activation_fn=None)
        # h_t = slim.fully_connected(x[:, t, :], latent_dim, activation_fn=tf.nn.relu)
        h_t_old = y_t
        y_pred += [tf.expand_dims(y_t,1)]

    return tf.concat(y_pred,1)

batch_size = 64
num_time_steps = 100
dim_control = 4
dim_state = 9
control_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps, dim_control])
y_true_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps, dim_state])
y_pred = my_rnn(control_pl, y_true_pl[:, 0, :])
loss = tf.reduce_mean(tf.abs(y_pred[:, :-1, :] - y_true_pl[:, :, :]))

## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
grads = optimizer.compute_gradients(loss, tf.all_variables())
train_op = optimizer.apply_gradients(grads)

## training starts ###
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
init = tf.global_variables_initializer()
sess.run(init)



import scipy.io as sio
import numpy as np
aa = sio.loadmat('./data/Boeing747_step_20s_ts01_normalized.mat')
data = aa['y']
if 1:
    new_data = []
    for data_i in data:
        new_data += [data_i[:-1, :] - data_i[1:, :]]
    new_data = np.asarray(new_data)
    new_data_normalized = (new_data - np.expand_dims(np.min(new_data, 1), 1)) / np.expand_dims(
        np.max(new_data, 1) - np.min(new_data, 1), 1)
    data = new_data_normalized
y_train = np.asarray([ data[i,:num_time_steps] for i in range(0, data.shape[0], 2) ])
y_train_ex = np.asarray([ data[i,num_time_steps:num_time_steps*2] for i in range(0, data.shape[0], 2) ])
y_test = np.asarray([ data[i, :num_time_steps] for i in range(0, data.shape[0], 2) ])

y_train_max = np.asarray([np.max(y_train[:,:,i]) for i in range(9)])
y_train_min = np.asarray([np.min(y_train[:,:,i]) for i in range(9)])
y_trin = (y_train - np.expand_dims(np.expand_dims(y_train_min,0),0)) / np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0)
y_test = (y_test - np.expand_dims(np.expand_dims(y_train_min,0),0)) / np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0)
y_train_ex = (y_train_ex - np.expand_dims(np.expand_dims(y_train_min,0),0)) / np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0)
control = [[[1 / 57.2958, 0, 0, 1 / 57.2958]]*(num_time_steps)] * batch_size

max_epoch = 20000
train_loss_val_hist = []
test_loss_val_hist = []
train_loss_val_hist_ex = []
test_loss_val_hist_ex = []
for eq_i in range(max_epoch):
    # training data, for optimization
    ave_loss_val_train = []
    num_itr = y_train.shape[0] / batch_size
    for itr_i in range(num_itr):
        feed_dict = {control_pl: control,
                     y_true_pl: y_train[itr_i*batch_size : (itr_i+1)*batch_size]}
        loss_val, _ = sess.run([loss, train_op], feed_dict)
        ave_loss_val_train += [loss_val]
    train_loss_val_hist += [np.mean(ave_loss_val_train)]
    # extrapolated training data
    ave_loss_val_train_ex = []
    num_itr = y_train_ex.shape[0] / batch_size
    for itr_i in range(num_itr):
        feed_dict = {control_pl: control,
                     y_true_pl: y_train_ex[itr_i*batch_size : (itr_i+1)*batch_size]}
        loss_val = sess.run(loss, feed_dict)
        ave_loss_val_train_ex += [loss_val]
    train_loss_val_hist_ex += [np.mean(ave_loss_val_train_ex)]
    # testing data
    ave_loss_val_test = []
    num_itr = y_test.shape[0] / batch_size
    for itr_i in range(num_itr):
        feed_dict = {control_pl: control,
                     y_true_pl: y_test[itr_i * batch_size: (itr_i + 1) * batch_size]}
        loss_val = sess.run(loss, feed_dict)
        ave_loss_val_test += [loss_val]
    test_loss_val_hist += [np.mean(ave_loss_val_test)]
    # # extrapolated testing data
    # ave_loss_val_test_ex = []
    # num_itr = y_test.shape[0] / batch_size
    # for itr_i in range(num_itr):
    #     feed_dict = {control_pl: control,
    #                  y_true_pl: y_test_ex[itr_i * batch_size: (itr_i + 1) * batch_size]}
    #     loss_val = sess.run(loss, feed_dict)
    #     ave_loss_val_test_ex += [loss_val]
    # test_loss_val_hist_ex += [np.mean(ave_loss_val_test_ex)]

    print(np.mean(ave_loss_val_train), np.mean(ave_loss_val_test))




import matplotlib.pyplot as plt
plt.plot(train_loss_val_hist, label='train')
plt.plot(test_loss_val_hist, label='test')
#plt.plot(train_loss_val_hist_ex, label='train_ex')
#plt.plot(test_loss_val_hist_ex, label='test_ex')
plt.legend()
print('done')
import matplotlib.pyplot as plt
import numpy as np
feed_dict = {control_pl: control,
             y_true_pl: y_test[itr_i*batch_size : (itr_i+1)*batch_size]}
y_pred_val = sess.run(y_pred, feed_dict)
y_pred_val = y_pred_val * np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0) + np.expand_dims(np.expand_dims(y_train_min,0),0)
y_true_val = sess.run(y_true_pl, feed_dict)
y_true_val = y_true_val * np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0) + np.expand_dims(np.expand_dims(y_train_min,0),0)
for i in range(10):
    plt.figure()
    for j in range(9):
        plt.subplot(3,3,j+1)
        plt.plot(y_pred_val[i,:,j], label='pred')
        plt.plot(y_true_val[i,:,j], label='true')
        plt.legend()
        #plt.axis([0, num_time_steps, y_train_min[j], y_train_max[j]])
plt.show()