import tensorflow as tf

def my_lstm(x, y_0):
    batch_size, time_steps, y_dim = x.get_shape().as_list()

    dim_out = y_0.get_shape().as_list()[1]
    dim_hid = dim_out + x.get_shape().as_list()[2]
    w_f = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w1')
    b_f = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b1')
    w_i = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w2')
    b_i = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b2')
    w_c = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w3')
    b_c = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b3')
    w_o = tf.Variable(tf.truncated_normal([dim_hid, dim_out], stddev=0.1), name='w4')
    b_o = tf.Variable(tf.truncated_normal([dim_out], stddev=0.1), name='b4')

    c_t_0 = 0
    h_t_0 = y_0
    x_t = x[:,0,:]
    y_pred = [tf.expand_dims(y_0, 1)]
    for t in range(time_steps-1):
        h_x = tf.concat([h_t_0, x_t], 1)
        f_t = tf.sigmoid( tf.nn.xw_plus_b(h_x, w_f, b_f) )
        i_t = tf.sigmoid( tf.nn.xw_plus_b(h_x, w_i, b_i) )
        c_t_hat = tf.nn.tanh( tf.nn.xw_plus_b(h_x, w_c, b_c) )
        c_t = f_t * c_t_0 + i_t * c_t_hat
        o_t = tf.sigmoid( tf.nn.xw_plus_b(h_x, w_o, b_o) )
        h_t = o_t * tf.nn.tanh(c_t)

        h_t_0 = h_t
        c_t_0 = c_t
        x_t = x[:, t, :]
        y_pred += [tf.expand_dims(h_t, 1)]

    return tf.concat(y_pred,1)

def my_lstm_stack(x, y0):
    with tf.name_scope("stack0") as scope:
        y_pred_stack0 = my_lstm(x, y0)
    with tf.name_scope("stack1") as scope:
        y_pred_stack1 = my_lstm(y_pred_stack0, y0)
    with tf.name_scope("stack1") as scope:
        y_pred_stack2 = my_lstm(y_pred_stack1, y0)
    return y_pred_stack2

max_epoch = 2000
batch_size = 64
num_time_steps = 100
dim_control = 2
dim_state = 5
control_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps, dim_control])
y_true_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps, dim_state])
y_pred = my_lstm_stack(control_pl, y_true_pl[:, 0, :])
loss = tf.reduce_mean(tf.abs(y_pred - y_true_pl))

## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
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
if  1:
    new_data = []
    for data_i in data:
        new_data += [data_i[:-1,:] - data_i[1:,:]]
    new_data = np.asarray(new_data)
    new_data_normalized =(new_data - np.expand_dims(np.min(new_data,1),1) ) / np.expand_dims(np.max(new_data,1) - np.min(new_data,1), 1)
    data = new_data_normalized
y_train = []
y_train_ex = []
y_test = []
for i in range(data.shape[0]):
    if i%10!=0:
        y_train += [ data[i,:num_time_steps, 4:] ]
        y_train_ex += [ data[i,num_time_steps:num_time_steps*2, 4:] ]
    else:
        y_test += [ data[i, :num_time_steps, 4:]]
y_train = np.asarray(y_train)
y_train_ex = np.asarray(y_train_ex)
y_test = np.asarray(y_test)
NEW_TEST_SET = 1
if NEW_TEST_SET:
    new_y_test = sio.loadmat('/home/hope-yao/Documents/physical_network/data/Boeing747_step_200s_ts01_normalized_case4.mat')['y'][:,:num_time_steps+1, 4:]
    new_data = []
    for data_i in new_y_test:
        new_data += [data_i[:-1,:] - data_i[1:,:]]
    new_data = np.asarray(new_data)
    new_data_normalized =(new_data - np.expand_dims(np.min(new_data,1),1) ) / np.expand_dims(np.max(new_data,1) - np.min(new_data,1), 1)
    new_y_test = np.tile(new_data_normalized,(batch_size, 1, 1))


y_train_max = np.asarray([np.max(y_train[:,:,i]) for i in range(5)])
y_train_min = np.asarray([np.min(y_train[:,:,i]) for i in range(5)])
y_trin = (y_train - np.expand_dims(np.expand_dims(y_train_min,0),0)) / np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0)
y_test = (y_test - np.expand_dims(np.expand_dims(y_train_min,0),0)) / np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0)
y_train_ex = (y_train_ex - np.expand_dims(np.expand_dims(y_train_min,0),0)) / np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0)
train_control = [[[ 0, 1 / 57.2958]] *(num_time_steps)] * batch_size
test_control = [[[ 0, 1 / 57.2958]] *(num_time_steps)] * batch_size
if NEW_TEST_SET:
    new_test_control = [[[1/57.2958, 0]] * 20 + [[0, 0]] * 80] * batch_size

train_loss_val_hist = []
test_loss_val_hist = []
train_loss_val_hist_ex = []
test_loss_val_hist_ex = []
new_test_loss_val_hist = []
for ep_i in range(max_epoch):
    # training data, for optimization
    ave_loss_val_train = []
    num_itr = y_train.shape[0] / batch_size
    for itr_i in range(num_itr):
        feed_dict = {control_pl: train_control,
                     y_true_pl: y_train[itr_i*batch_size : (itr_i+1)*batch_size]}
        loss_val, _ = sess.run([loss, train_op], feed_dict)
        ave_loss_val_train += [loss_val]
    train_loss_val_hist += [np.mean(ave_loss_val_train)]
    # extrapolated training data
    ave_loss_val_train_ex = []
    num_itr = y_train_ex.shape[0] / batch_size
    for itr_i in range(num_itr):
        feed_dict = {control_pl: train_control,
                     y_true_pl: y_train_ex[itr_i*batch_size : (itr_i+1)*batch_size]}
        loss_val = sess.run(loss, feed_dict)
        ave_loss_val_train_ex += [loss_val]
    train_loss_val_hist_ex += [np.mean(ave_loss_val_train_ex)]
    # testing data
    ave_loss_val_test = []
    num_itr = y_test.shape[0] / batch_size
    for itr_i in range(num_itr):
        feed_dict = {control_pl: test_control,
                     y_true_pl: y_test[itr_i * batch_size: (itr_i + 1) * batch_size]}
        loss_val = sess.run(loss, feed_dict)
        ave_loss_val_test += [loss_val]
    test_loss_val_hist += [np.mean(ave_loss_val_test)]

    feed_dict = {control_pl: new_test_control,
                     y_true_pl: new_y_test}
    new_loss_val = sess.run(loss, feed_dict)

    # # extrapolated testing data
    # ave_loss_val_test_ex = []
    # num_itr = y_test.shape[0] / batch_size
    # for itr_i in range(num_itr):
    #     feed_dict = {control_pl: control,
    #                  y_true_pl: y_test_ex[itr_i * batch_size: (itr_i + 1) * batch_size]}
    #     loss_val = sess.run(loss, feed_dict)
    #     ave_loss_val_test_ex += [loss_val]
    # test_loss_val_hist_ex += [np.mean(ave_loss_val_test_ex)]

    print(ep_i, np.mean(ave_loss_val_train), np.mean(ave_loss_val_test), new_loss_val)


import matplotlib.pyplot as plt
plt.plot(train_loss_val_hist, label='train')
plt.plot(test_loss_val_hist, label='test')
#plt.plot(train_loss_val_hist_ex, label='train_ex')
#plt.plot(test_loss_val_hist_ex, label='test_ex')
plt.legend()
print('done')
import matplotlib.pyplot as plt
import numpy as np

feed_dict = {control_pl: new_test_control,
             y_true_pl: new_y_test[: batch_size]}
y_pred_val = sess.run(y_pred, feed_dict)
y_pred_val = y_pred_val * np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0) + np.expand_dims(np.expand_dims(y_train_min,0),0)
for i in range(3):
    plt.figure()
    for j in range(5):
        plt.subplot(3,3,j+1)
        plt.plot(y_pred_val[i,:,j], label='new test pred')
        plt.plot(new_y_test[i,:,j], label='new test true')
        plt.legend()
        #plt.axis([0, num_time_steps, y_train_min[j], y_train_max[j]])
plt.show()



feed_dict = {control_pl: test_control,
             y_true_pl: y_test[: batch_size]}
y_pred_val = sess.run(y_pred, feed_dict)
y_pred_val = y_pred_val * np.expand_dims(np.expand_dims(y_train_max-y_train_min,0),0) + np.expand_dims(np.expand_dims(y_train_min,0),0)
for i in range(3):
    plt.figure()
    for j in range(5):
        plt.subplot(3,3,j+1)
        plt.plot(y_pred_val[i,:,j], label='test pred')
        plt.plot(y_test[i,:,j], label='test true')
        plt.legend()
        #plt.axis([0, num_time_steps, y_train_min[j], y_train_max[j]])
plt.show()



feed_dict = {control_pl: train_control,
             y_true_pl: y_train[: batch_size]}
y_pred_val = sess.run(y_pred, feed_dict)
for i in range(3):
    plt.figure()
    for j in range(5):
        plt.subplot(3,3,j+1)
        plt.plot(y_pred_val[i,:,j], label='train pred')
        plt.plot(y_train[i,:,j], label='train true')
        plt.legend()
        #plt.axis([0, num_time_steps, y_train_min[j], y_train_max[j]])
plt.show()





























