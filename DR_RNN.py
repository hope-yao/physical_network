import tensorflow as tf
from tqdm import tqdm
import numpy as np
from utils import *
import scipy.io as sio
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from Data_driven_model import *
import sys

class DR_RNN:
    def __init__(self, cfg):
        self.delta_t = cfg['delta_t']
        self.time_start = cfg['time_start']
        self.time_end = cfg['time_end']
        self.num_time_steps = int((self.time_end - self.time_start) / self.delta_t) + 1
        self.num_y = cfg['num_y']
        self.num_layers = cfg['num_layers']
        self.gamma = cfg['gamma']
        self.zeta = cfg['zeta']
        self.eps = cfg['eps']
        self.lr = cfg['lr']
        self.num_epochs = cfg['num_epochs']
        self.batch_size = cfg['batch_size']
        self.logdir = cfg['logdir']
        self.modeldir = cfg['modeldir']
        self.data_fn = cfg['data_fn']
        self.model_name = cfg['model']
        self.n_times_latentdim = cfg['n_times_latentdim']

    def load_data(self):
        self.raw_data = sio.loadmat(self.data_fn)
        self.y = self.raw_data['y']
        # self.y_do1t = self.raw_data['Y_dot']
        rand_idx = np.random.random_integers(0, len(self.y) - 1, len(self.y) - 1)
        # idx = np.random.choice(len(y), len(y), replace=False)
        # return y[idx[:500]], y[idx[500:]]
        return self.y[rand_idx[:500]], self.y[rand_idx[500:]] # 500,1000 split as in the paper

    def get_residual(self, y_tp1, y_t, delta_t):
        k1 = 1
        k2 = 1
        k3 = 1
        k4 = 1
        m1 = 1
        m2 = 1
        m3 = 1
        c1 = 0.02
        c2 = 0.02
        A = np.asarray([ [     0       ,        1     ,      0     ,       0    ,       0    ,     0],
                         [  -(k1+k2+k4)/m1 ,  -c1/m1  ,    k2/m1   ,      0     ,    k4/m1   ,    0],
                         [     0        ,       0     ,      0      ,      1    ,       0    ,     0],
                         [   k2/m2       ,      0     , -(k2+k3)/m2 ,   -c2/m2  ,      k3/m2 ,    c2/m2],
                         [     0        ,       0     ,      0       ,     0    ,       0    ,     1],
                         [   k4/m3      ,       0     ,     k3/m3    ,   c2/m3  , -(k3+k4)/m3 , -c2/m3]], dtype='float32')

        er = y_tp1 - y_t - delta_t * tf.transpose(tf.matmul(A,tf.transpose(y_tp1,(1,0))),(1,0))
        return er

    def build_training_model(self):
        # self.ode_para = tf.Variable([0., 0., 0.], name='ode_para')
        # self.ode_para = tf.Variable(tf.truncated_normal([3, ], stddev=0.1), name='ode_para')

        self.load_data()

        self.ode_para_mean = tf.Variable(tf.truncated_normal([3, ], stddev=0.1), name='ode_para_mean')
        self.ode_para_var = tf.Variable(tf.truncated_normal([3, ], stddev=0.1), name='ode_para_var')
        eps = tf.random_normal((3,), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.ode_para = self.ode_para_mean + self.ode_para_var * eps

        self.weight_w = tf.Variable(tf.truncated_normal([self.num_y, ], mean=1, stddev=0.1), name='weight_w')
        self.weight_u = tf.Variable(tf.truncated_normal([self.num_y, self.num_y], stddev=0.1), name='weight_u')
        self.eta = tf.Variable(tf.random_uniform([self.num_layers - 1, ], minval=0.1,maxval=0.4), name='eta')
        self.y_true = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_time_steps, self.num_y))

        self.y_pred = self.y_true[:, 0:1, :]  # initial predictions
        if cfg['model'] == 'DR_RNN':
            with tf.variable_scope("DR_RNN", reuse=True) as training:
                for t in range(self.num_time_steps - 1):
                    y_t = self.y_pred[:, -1, :]
                    y_tp1 = y_t  # initial guess for the value in next time step
                    r_tp1 = self.get_residual(y_tp1, self.y_true[:, t, :], self.delta_t)
                    # first layer
                    y_tp1 = y_tp1 - tf.multiply(self.weight_w, tf.nn.tanh(
                        tf.transpose(tf.matmul(self.weight_u, tf.transpose(r_tp1, (1, 0))), (1, 0)))) #corrected
                    # following layers
                    G = tf.square(tf.norm(r_tp1, axis=1))  # which is not specified in the paper
                    for k in range(self.num_layers - 1):
                        r_tp1 = self.get_residual(y_tp1, self.y_true[:, t, :], self.delta_t)
                        G = self.gamma * tf.norm(r_tp1, axis=1) + self.zeta * G #missing square
                        y_tp1 = y_tp1 - tf.expand_dims(self.eta[k] / tf.sqrt(G + self.eps), 1) * r_tp1
                    self.y_pred = tf.concat([self.y_pred, tf.expand_dims(y_tp1, 1)], 1)

        elif cfg['model'] == 'RNN':
            with tf.variable_scope("RNN", reuse=None) as training:
                self.y_pred = my_rnn(self.y_true, self.num_layers, n_times_latentdim=self.n_times_latentdim, training=True)

        elif cfg['model'] == 'LSTM':
            with tf.variable_scope("LSTM", reuse=None) as training:
                self.y_pred = my_lstm(self.y_true, self.num_layers, n_times_latentdim=self.n_times_latentdim, training=True)

        decay = 0
        if decay:
            time_decay = []
            w_i = 0.9
            for i in range(self.num_time_steps):
                w_i = w_i * 0.9
                time_decay += [w_i]
            self.training_loss = tf.reduce_mean(
                tf.reduce_mean(tf.abs(self.y_true - self.y_pred), (0, 2)) * np.asarray(time_decay)[::-1])
        else:
            self.training_loss = tf.reduce_mean(tf.square(self.y_true - self.y_pred))

    def build_test_model(self):
        self.delta_t_testing = tf.placeholder(tf.float32, shape=())
        self.y_pred_testing = self.y_true[:, 0:1, :]  # initial predictions
        if cfg['model'] == 'DR_RNN':
            with tf.variable_scope("DR_RNN", reuse=True) as testing:
                for t in range(self.num_time_steps - 1):
                    y_t_testing = self.y_pred_testing[:, -1, :]
                    y_tp1_testing = y_t_testing  # initial guess for the value in next time step
                    r_tp1_testing = self.get_residual(y_tp1_testing, y_t_testing, self.delta_t_testing)
                    # first layer
                    y_tp1_testing = y_tp1_testing - tf.multiply(self.weight_w, tf.nn.tanh(
                        tf.transpose(tf.matmul(self.weight_u, tf.transpose(r_tp1_testing, (1, 0))), (1, 0))))
                    # following layers
                    G_testing = tf.square(tf.norm(r_tp1_testing, axis=1))  # which is not specified in the paper
                    for k in range(self.num_layers - 1):
                        r_tp1_testing = self.get_residual(y_tp1_testing, y_t_testing, self.delta_t_testing)
                        G_testing = self.gamma * tf.square(tf.norm(r_tp1_testing, axis=1)) + self.zeta * G_testing
                        y_tp1_testing = y_tp1_testing - tf.expand_dims(self.eta[k] / tf.sqrt(G_testing + self.eps),
                                                                       1) * r_tp1_testing
                    self.y_pred_testing = tf.concat([self.y_pred_testing, tf.expand_dims(y_tp1_testing, 1)], 1)
                self.testing_loss = tf.reduce_mean(tf.abs(self.y_true - self.y_pred_testing))

        elif cfg['model'] == 'RNN':
            with tf.variable_scope("RNN", reuse=True) as testing:
                self.y_pred_testing = my_rnn(self.y_true, self.num_layers, n_times_latentdim=self.n_times_latentdim, training=False)

        elif cfg['model'] == 'LSTM':
            with tf.variable_scope("LSTM", reuse=True) as testing:
                self.y_pred_testing = my_lstm(self.y_true, self.num_layers, n_times_latentdim=self.n_times_latentdim, training=False)

        self.testing_loss = tf.reduce_mean(tf.square(self.y_true - self.y_pred_testing))

    def training_init(self):
        '''initial training graph and monitor'''

        ## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
        self.learning_rate = tf.Variable(self.lr)  # learning rate for optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        if cfg['model'] == 'DR_RNN':
            para_list = [self.weight_w, self.eta, self.ode_para_mean, self.ode_para_var]
        else:
            para_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cfg['model'])
        grads = optimizer.compute_gradients(self.training_loss, para_list)

            # for i,(g,v) in enumerate(grads):
        #     if g is not None:
        #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
        self.train_op = optimizer.apply_gradients(grads)

        ## Monitor ##
        self.summary_writer = tf.summary.FileWriter(self.logdir)
        self.summary_op_training = tf.summary.merge([
            tf.summary.scalar("loss/training_loss", self.training_loss),
            tf.summary.scalar("lr/lr", self.learning_rate),
        ])
        self.summary_op_testing = tf.summary.merge([
            tf.summary.scalar("loss/testing_loss", self.testing_loss),
        ])

        ## graph initialization ###
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.train.write_graph(self.sess.graph, logdir, 'train.pbtxt')
        self.saver = tf.train.Saver()

        ## Monitor ##
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.summary_op_training = tf.summary.merge([
            tf.summary.scalar("loss/training_loss", self.training_loss),
            # tf.summary.scalar("ODE/training_ode_para_mean_err", self.ode_para_mean_err),
            # tf.summary.scalar("ODE/training_ode_para_var_err", self.ode_para_var_err),
            tf.summary.scalar("lr/lr", self.learning_rate),
        ])
        self.summary_op_testing = tf.summary.merge([
            tf.summary.scalar("loss/testing_loss", self.testing_loss),
            # tf.summary.scalar("ODE/testing_ode_para_mean_err", self.ode_para_mean_err),
            # tf.summary.scalar("ODE/testing_ode_para_var_err", self.ode_para_var_err),
        ])


    def restore_model(self):
        file_path = './saved_models/DR-RNN_K2/DR-RNN_K2_2018_01_04_22_04_56/experiment_50000.ckpt'
        self.saver.restore(self.sess, file_path)

    def train_model(self):
        '''training starts '''

        self.y_train_all, self.y_test_all = self.load_data()
        self.y_train_data = self.y_train_all[:, :self.num_time_steps, :]
        self.y_test_data = self.y_test_all[:, :self.num_time_steps, :]

        count = 0
        for epoch in tqdm(range(self.num_epochs)):
            it_per_ep = len(self.y_train_data) / self.batch_size
            for i in range(it_per_ep):
                y_input = self.y_train_data[i * self.batch_size:(i + 1) * self.batch_size]
                self.sess.run(self.train_op, {self.y_true: y_input})

                if count % 10 == 0:
                    self.training_loss_value = self.sess.run(self.training_loss, {self.y_true: y_input})
                    # rand_idx = np.random.random_integers(0, len(y_test) - 1, size=self.batch_size)
                    rand_idx = np.arange(0,self.batch_size,1)
                    self.testing_loss_value = self.sess.run(self.testing_loss, {self.y_true: self.y_test_data[rand_idx], self.delta_t_testing:self.delta_t})
                    print("iter:{}  train_cost: {}  test_cost: {} ".format(count, self.training_loss_value, self.testing_loss_value))
                    training_summary, testing_summary = self.sess.run([self.summary_op_training,self.summary_op_testing],
                                                                      {self.y_true: self.y_test_data[rand_idx], self.delta_t_testing:self.delta_t})
                    self.summary_writer.add_summary(training_summary, count)
                    self.summary_writer.add_summary(testing_summary, count)
                    self.summary_writer.flush()

                if count % 50000 == 0:
                    self.sess.run(tf.assign(self.learning_rate, self.learning_rate * 0.8))
                    snapshot_name = "%s_%s" % ('experiment', str(count))
                    self.saver.save(self.sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
                count += 1
            self.visualization(self.delta_t, self.delta_t, count, dist_viz=0)
            # self.extrapolation_in_time()
            # self.save_data()
            print('done')

    def sensitivity(self):
        '''sensitivity analysis'''
        it_per_ep = len(self.y) / self.batch_size
        rand_idx = np.random.random_integers(0, len(self.y) - 1, len(self.y) - 1)
        for i in range(it_per_ep):
            y_in = self.y[rand_idx[i * self.batch_size:(i + 1) * self.batch_size],:self.num_time_steps]
            pred = self.sess.run(self.y_pred_testing,
                                 {self.y_true: y_in, self.delta_t_testing: self.delta_t})
            if i == 0:
                tt_x = y_in[:, 0, 1] * 10
                tt_pred = pred[:, -1, :]
            else:
                tt_pred = np.append(tt_pred, pred[:, -1, :], 0)
                tt_x = np.append(tt_x, y_in[:, 0, 1] * 10)
        for idx in range(1,3,1):
            plt.figure()
            plt.plot(self.y[:, 0, 1] * 10, self.y[:, self.num_time_steps, idx], 'b')
            plt.plot(tt_x, tt_pred[:, idx], 'r.')
            plt.show()

    def extrapolation_in_time(self):
        delta_t = self.delta_t
        y_test_new = self.sess.run(self.y_pred_testing,
                                   {self.y_true: self.y_test_data[:self.batch_size], self.delta_t_testing: delta_t})
        y_test_new_new = self.sess.run(self.y_pred_testing, {self.y_true: y_test_new[:self.batch_size, ::-1, :],
                                                             self.delta_t_testing: delta_t})
        tt_y = np.concatenate([y_test_new, y_test_new_new], 1)
        idx = 1
        plt.plot(np.arange(0, (tt_y.shape[1] - 0.5) * self.delta_t, self.delta_t), tt_y[0, :, idx], 'r',
                 label='k{}_test_'.format(self.num_layers)+self.model_name)
        plt.plot(np.arange(0, (tt_y.shape[1] - 0.5) * self.delta_t, self.delta_t), self.y_test_all[0, :tt_y.shape[1], idx],
                 'b', label='ODE45')
        plt.legend()
        plt.axvspan(self.num_time_steps*self.delta_t, self.num_time_steps*self.delta_t*2, facecolor='g', alpha=0.5)
        plt.show()

    def save_data(self,dist_viz=0):

        k = self.num_layers
        np.save(self.logdir + '/k{}_train_{}.npy'.format(k,self.model_name),
                self.sess.run(self.y_pred, {self.y_true: self.y_test_data[:self.batch_size]}))
        np.save(self.logdir + '/k{}_test_{}.npy'.format(k,self.model_name),
                self.sess.run(self.y_pred_testing, {self.y_true: self.y_test_data[:self.batch_size],self.delta_t_testing:self.delta_t}))
        np.save(self.logdir + '/k{}_numerical.npy'.format(k), self.y_test_data[:self.batch_size])

        if dist_viz:
            test_pred = []
            for i in range(60):
                test_pred += [self.sess.run(self.y_pred_testing, {
                    self.y_true: self.y_test_data[i * self.batch_size:(i + 1) * self.batch_size],
                    self.delta_t_testing: self.delta_t})]
            for i in range(self.num_y):
                np.save(self.logdir + '/k{}_test_dist_y{}.npy'.format(self.num_layers, i),
                        np.reshape(np.asarray(test_pred)[:, :, -1, i], 60 * self.batch_size))

    def visualization(self, delta_t_training, delta_t_testing, count, dist_viz=0):
        # aa = np.load(logdir + '/k{}_numerical.npy'.format(k))
        # bb = np.load(logdir + '/k{}_train_drrnn.npy'.format(k))
        # cc = np.load(logdir + '/k{}_test_drrnn.npy'.format(k))
        idx = 1
        numerical_var = self.y_test_data[:self.batch_size]
        training_var = self.sess.run(self.y_pred, {self.y_true: numerical_var})[0, :, idx]
        testing_var = self.sess.run(self.y_pred_testing, {self.y_true: numerical_var,self.delta_t_testing:delta_t_testing})[0, :, idx]
        numerical_var = numerical_var[0,:, idx]
        fig  = plt.figure()
        plt.plot(np.arange(0,self.num_time_steps,1)*delta_t_training, numerical_var, 'k', label='k{}_numerical'.format(self.num_layers))
        plt.plot(np.arange(0,self.num_time_steps,1)*delta_t_training, training_var, 'b', label='k{}_train_{}'.format(self.num_layers,self.model_name))
        plt.plot(np.arange(0,self.num_time_steps,1)*delta_t_testing, testing_var, 'r', label='k{}_test_{}'.format(self.num_layers,self.model_name))
        plt.legend()
        plt.xlabel('physical time')
        plt.ylabel('y{}'.format(idx))
        ttl_name = 'itr:{:d}  '.format(count)+'train_err:{0:.4e}  '.format(self.training_loss_value) + 'test_err:{0:4e}'.format(self.testing_loss_value)
        plt.title(ttl_name)
        if np.min(numerical_var)<0:
            y_l = np.min(numerical_var)*2
        else:
            y_l = np.min(numerical_var)/2
        if np.max(numerical_var)<0:
            y_u = np.max(numerical_var)*2
        else:
            y_u = np.max(numerical_var)/2
        x_u = self.num_time_steps * max(delta_t_training,delta_t_testing)
        # plt.axis([0, x_u, y_l, y_u])
        # plt.show()
        fig.savefig(self.logdir+'/iter{}.png'.format(count))

        if dist_viz:
            plt.figure()
            y = sio.loadmat('./data/3dof_sys_l.mat')['y']

            y2_end = y[:, -1, 1]
            sns.distplot(y2_end, label='numerical', hist=False, kde_kws={"color": "k"})
            yy = np.load(logdir + '/k{}_test_dist_y1.npy'.format(self.num_layers))
            sns.distplot(yy, label='{}_{}'.format(self.model_name, self.num_layers), hist=False)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    cfg = {
           'model': 'DR_RNN',#RNN, 'LSTM','DR_RNN',
           'delta_t': 1e-1,
           'time_start': 0,
           'time_end': 10,
           'num_y': 6,
           'num_layers': 4,
           'gamma': 0.1,
           'zeta': 0.9,
           'eps': 1e-8,
           'lr': 0.01,  # 0.2 for DR_RNN_1, 0.1 for DR_RNN_2 and 3, ??? for DR_RNN_4,
           'num_epochs': 15*200,
           'batch_size': 64,
           'data_fn': './data/3dof_sys_l.mat',  # './data/problem1.npz'
           'n_times_latentdim': 10,
           }
    logdir, modeldir = creat_dir(cfg['model']+"_K{}".format(cfg['num_layers']))
    copyfile('DR_RNN.py', modeldir + '/' + 'DR_RNN.py')
    cfg['logdir'] = logdir
    cfg['modeldir'] = modeldir

    drrnn = DR_RNN(cfg)
    drrnn.build_training_model()
    drrnn.build_test_model()
    drrnn.training_init()

    drrnn.train_model()
    print('done')
