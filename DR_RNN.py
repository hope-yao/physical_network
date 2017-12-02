import tensorflow as tf
from tqdm import tqdm
import numpy as np
from utils import *
import scipy.io as sio

delta_t = 1e-1
time_start = 0
time_end = 10
num_time_steps = int((time_end-time_start)/delta_t) + 1
num_y = 3
num_layers = 4
gamma = 0.1
zeta = 0.9
eps = 1e-8
lr = 0.2 # looks good for DR_RNN_4, which is not specified in the paper
# lr = 1. # looks good for DR_RNN_1,DR_RNN_2, which is not specified in the paper
num_epochs = 15
batch_size = 15

def get_residual(y_tp1, y_t, delta_t):
    er1 = y_tp1[:,0] - y_t[:,0] - delta_t * y_tp1[:,0] * y_tp1[:,2]
    er2 = y_tp1[:,1] - y_t[:,1] + delta_t * y_tp1[:,1] * y_tp1[:,2]
    er3 = y_tp1[:,2] - y_t[:,2] - delta_t * (-y_tp1[:,0] ** 2 + y_tp1[:,1]** 2)
    return tf.stack([er1, er2, er3],1)

def load_data(data_fn):
    y = sio.loadmat(data_fn)['y']
    # idx = np.random.choice(len(y), len(y), replace=False)
    # return y[idx[:500]], y[idx[500:]]
    return y[:500], y[500:]

def main(cfg):

    weight_w = tf.Variable(tf.truncated_normal([3,], stddev=0.1),name='weight_w')
    weight_u = tf.constant(1.,name='weight_u')#tf.truncated_normal([1,], stddev=0.1)
    eta = tf.Variable(tf.random_uniform([num_layers-1,]),name='eta')
    y_true = tf.placeholder(tf.float32,shape=(batch_size,num_time_steps,num_y))

    y_pred = y_true[:,0:1,:]# initial predictions
    with tf.variable_scope("DR_RNN_testing", reuse=True) as training:
        ## Training
        for t in range(num_time_steps-1):
            y_t = y_pred[:, -1, :]
            y_tp1 = y_t #initial guess for the value in next time step
            r_tp1 = get_residual(y_tp1, y_true[:,t,:], delta_t)
            # first layer
            y_tp1 = y_tp1 - weight_w * tf.nn.tanh(weight_u * r_tp1)
            # following layers
            G =  tf.norm(r_tp1,axis=1)  #which is not specified in the paper
            for k in range(num_layers-1):
                r_tp1 = get_residual(y_tp1, y_true[:,t,:], delta_t)
                G = gamma * tf.norm(r_tp1,axis=1) + zeta * G
                y_tp1 = y_tp1 - tf.expand_dims(eta[k]/tf.sqrt(G+eps),1) * r_tp1
            y_pred = tf.concat([y_pred,tf.expand_dims(y_tp1,1)],1)

    y_pred_testing = y_true[:,0:1,:]# initial predictions
    with tf.variable_scope("DR_RNN_testing", reuse=True) as testing:
        ## Testing
        for t in range(num_time_steps-1):
            y_t_testing = y_pred_testing[:, -1, :]
            y_tp1_testing = y_t_testing  # initial guess for the value in next time step
            r_tp1_testing = get_residual(y_tp1_testing, y_t_testing, delta_t)
            # first layer
            y_tp1_testing = y_tp1_testing - weight_w * tf.nn.tanh(weight_u * r_tp1_testing)
            # following layers
            G_testing =  tf.norm(r_tp1_testing,axis=1)  #which is not specified in the paper
            for k in range(num_layers-1):
                r_tp1_testing = get_residual(y_tp1_testing, y_t_testing, delta_t)
                G_testing = gamma * tf.norm(r_tp1,axis=1) + zeta * G_testing
                y_tp1_testing = y_tp1_testing - tf.expand_dims(eta[k]/tf.sqrt(G_testing+eps),1) * r_tp1_testing
            y_pred_testing = tf.concat([y_pred_testing,tf.expand_dims(y_tp1_testing,1)],1)


    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    loss_testing = tf.reduce_mean(tf.square(y_true - y_pred_testing))

    ## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(loss,[weight_w,eta])
    # for i,(g,v) in enumerate(grads):
    #     if g is not None:
    #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)

    ## Monitor ##
    logdir, modeldir = creat_dir("DR-RNN_K{}".format(num_layers))
    summary_writer = tf.summary.FileWriter(logdir)
    summary_op = tf.summary.merge([
        tf.summary.scalar("loss/loss", loss),
        tf.summary.scalar("lr/lr", learning_rate),
    ])

    ## graph initialization ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.write_graph(sess.graph, logdir, 'train.pbtxt')

    ## training starts ###
    # data_fn = './data/problem1.npz'
    data_fn = './data/problem1_1129.mat'
    y_train, y_test = load_data(data_fn)#500,1000 split as in the paper
    count = 0
    for epoch in range(num_epochs):
        it_per_ep = len(y_train)/batch_size
        for i in tqdm(range(it_per_ep)):
            y_input = y_train[i*batch_size:(i + 1)*batch_size]
            sess.run(train_op,{y_true:y_input})

            if count%10==0:
                train_result = sess.run(loss, {y_true:y_input})
                rand_idx = np.random.random_integers(0,len(y_test)-1,size=batch_size)
                test_result = sess.run(loss_testing, {y_true: y_test[rand_idx]})
                print("iter:{}  train_cost: {}  test_cost: {} ".format(count, train_result, test_result))
                summary = sess.run(summary_op, {y_true:y_test[rand_idx]})
                summary_writer.add_summary(summary, count)
                summary_writer.flush()

            if count%1000==1:
                sess.run( tf.assign(learning_rate, learning_rate * 0.5) )
            count += 1

    print('done')
    # ## Monitor ##
    # # saver = tf.train.Saver() # saves variables learned during training
    # logdir, modeldir = creat_dir("VoxNet_T{}_n{}".format(num_glimpse, glimpse_size))
    # saver = tf.train.Saver()
    # #saver.restore(sess, "*.ckpt")
    # summary_writer = tf.summary.FileWriter(logdir)
    # grad1 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Variable:0")[0])[0]))
    # grad2 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Variable_2:0")[0])[0]))
    # grad3 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="VoxNet/fc3/weights:0")[0])[0]))
    # grad4 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="VoxNet/fc4/weights:0")[0])[0]))
    # grad5 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reader/fc1/weights:0")[0])[0]))
    # grad6 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reader/fc2/weights:0")[0])[0]))
    # summary_op = tf.summary.merge([
    #     tf.summary.scalar("loss/loss", cost),
    #     tf.summary.scalar("err/err_last", err[-1]),
    #     tf.summary.scalar("lr/lr", learning_rate),
    #     tf.summary.scalar("grad/grad1", grad1),
    #     tf.summary.scalar("grad/grad2", grad2),
    #     tf.summary.scalar("grad/grad3", grad3),
    #     tf.summary.scalar("grad/grad4", grad4),
    #     tf.summary.scalar("grad/grad5", grad5),
    #     tf.summary.scalar("grad/grad6", grad6),
    # ])

if __name__ == "__main__":


    cfg = {'batch_size': 128,
           'img_dim': 3,
           'vox_size': 32,
           'n_class': 10,
           'num_glimpse': 8,
           'glimpse_size': 3,
           'data_path': '../data/ModelNet',
           'lr': 1e-3,
           }
    main(cfg)