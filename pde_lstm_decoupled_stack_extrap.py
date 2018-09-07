import numpy as np
import tensorflow as tf

class PDE_LSTM():
    def __init__(self):
        self.num_time_steps = 100
        self.max_epoch = 2000
        self.batch_size = 64
        self.dim_control = 2
        self.dim_state = 5

    def my_lstm(self, x, y_0):
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

    def my_lstm_stack(self, x, y0):
        with tf.name_scope("stack0") as scope:
            y_pred_stack0 = self.my_lstm(x, y0)
        with tf.name_scope("stack1") as scope:
            y_pred_stack1 = self.my_lstm(y_pred_stack0, y0)
        with tf.name_scope("stack2") as scope:
            y_pred_stack2 = self.my_lstm(y_pred_stack1, y0)
        with tf.name_scope("stack3") as scope:
            y_pred_stack3 = self.my_lstm(y_pred_stack2, y0)
        return y_pred_stack0


    def get_data(self):
        import scipy.io as sio
        import numpy as np
        aa = sio.loadmat('./data/Boeing747_step_20s_ts01_normalized.mat')
        data = aa['y']
        if 1:
            new_data = []
            for data_i in data:
                new_data += [data_i[:-1, :] - data_i[1:, :]]
            new_data = np.asarray(new_data)
            range_dict = {
                'min' : np.min(new_data, 1),
                'max' : np.max(new_data, 1),
                'init': np.copy(aa['y'][:, 0, :]),
                'train_min': [],
                'train_max': [],
                'train_init': [],
                'test_min': [],
                'test_max': [],
                'test_init': [],
                'train_ex_min': [],
                'train_ex_max': [],
                'train_ex_init': [],
                'test_ex_min': [],
                'test_ex_max': [],
                'test_ex_init': [],
                'new_min': [],
                'new_max': [],
                'new_init': [],
                'new_ex_min': [],
                'new_ex_max': [],
                'new_ex_init': [],
            }
            new_data_normalized = (new_data - np.expand_dims(np.min(new_data, 1), 1)) / np.expand_dims(
                np.max(new_data, 1) - np.min(new_data, 1), 1)
            data = new_data_normalized
        y_train = []
        y_test = []
        for i in range(data.shape[0]):
            if i % 10 != 0:
                y_train += [data[i, :self.num_time_steps, 4:]]
                range_dict['train_min'] += [range_dict['min'][i, 4:]]
                range_dict['train_max'] += [range_dict['max'][i, 4:]]
                range_dict['train_init'] += [range_dict['init'][i, 4:]]
                # y_train_ex += [data[i, self.num_time_steps:, 4:]]
            else:
                y_test += [data[i, :self.num_time_steps, 4:]]
                range_dict['test_min'] += [range_dict['min'][i, 4:]]
                range_dict['test_max'] += [range_dict['max'][i, 4:]]
                range_dict['test_init'] += [range_dict['init'][i, 4:]]
                # y_test_ex += [data[i, self.num_time_steps:, 4:]]
        y_train = np.asarray(y_train)
        # y_train_ex = np.asarray(y_train_ex)
        y_test = np.asarray(y_test)
        # y_test_ex = np.asarray(y_test_ex)
        data_dict = {
            'y_train': np.copy(y_train),
            'y_test': np.copy(y_test),
        }

        # extrapolation data
        y_train_ex = []
        y_test_ex = []
        aa = sio.loadmat('./data/Boeing747_step_200s_ts01_normalized.mat')
        data = aa['y']
        new_data = []
        for data_i in data:
            new_data += [data_i[:-1, :] - data_i[1:, :]]
        new_data = np.asarray(new_data)
        range_dict['min'] = np.min(new_data, 1)
        range_dict['max'] = np.max(new_data, 1)
        range_dict['init'] = np.copy(aa['y'][:, 0, :])
        new_data_normalized = (new_data - np.expand_dims(np.min(new_data, 1), 1)) / np.expand_dims(
            np.max(new_data, 1) - np.min(new_data, 1), 1)
        data = new_data_normalized
        for i in range(data.shape[0]):
            if i % 2 != 0:
                y_train += [data[i, :self.num_time_steps, 4:]]
                range_dict['train_min'] += [range_dict['min'][i, 4:]]
                range_dict['train_max'] += [range_dict['max'][i, 4:]]
                range_dict['train_init'] += [range_dict['init'][i, 4:]]
                y_train_ex += [data[i, :, 4:]]
                range_dict['train_ex_min'] += [range_dict['min'][i, 4:]]
                range_dict['train_ex_max'] += [range_dict['max'][i, 4:]]
                range_dict['train_ex_init'] += [range_dict['init'][i, 4:]]
            else:
                y_test += [data[i, :self.num_time_steps, 4:]]
                range_dict['test_min'] += [range_dict['min'][i, 4:]]
                range_dict['test_max'] += [range_dict['max'][i, 4:]]
                range_dict['test_init'] += [range_dict['init'][i, 4:]]
                y_test_ex += [data[i, :, 4:]]
                range_dict['test_ex_min'] += [range_dict['min'][i, 4:]]
                range_dict['test_ex_max'] += [range_dict['max'][i, 4:]]
                range_dict['test_ex_init'] += [range_dict['init'][i, 4:]]
        y_train_ex = np.concatenate([np.tile(np.expand_dims(y_train_ex[i],0),(self.batch_size,1,1)) for i in range(5)],0)
        y_test_ex = np.concatenate([np.tile(np.expand_dims(y_test_ex[i],0),(self.batch_size,1,1)) for i in range(5)],0)
        data_dict['y_train_ex'] = np.copy(y_train_ex)
        data_dict['y_test_ex'] = np.copy(y_test_ex)
        if 1:
            new_y_test = \
                sio.loadmat('./data/Boeing747_step_200s_ts01_normalized_case4.mat')[
                    'y'][:, :, 4:]
            new_y_test = np.tile(new_y_test, (self.batch_size, 1, 1))
            new_data = []
            for data_i in new_y_test:
                new_data += [data_i[:-1, :] - data_i[1:, :]]
            new_data = np.asarray(new_data)
            range_dict['min'] = np.copy(np.min(new_y_test, 1))
            range_dict['max'] = np.copy(np.max(new_y_test, 1))
            range_dict['init'] = np.copy(new_y_test[:, 0, :])
            new_data_normalized = (new_data - np.expand_dims(np.min(new_data, 1), 1)) / np.expand_dims(
                np.max(new_data, 1) - np.min(new_data, 1), 1)
            # new_y_test = np.tile(new_data_normalized, (self.batch_size, 1, 1))
        data_dict['new_y_test'] = np.copy(new_y_test[:, :self.num_time_steps, :])
        range_dict['new_min'] += [range_dict['min']]
        range_dict['new_max'] += [range_dict['max']]
        range_dict['new_init'] += [range_dict['init']]
        data_dict['new_y_test_ex'] = np.copy(new_y_test[:, :])
        range_dict['new_ex_min'] += [range_dict['min']]
        range_dict['new_ex_max'] += [range_dict['max']]
        range_dict['new_ex_init'] += [range_dict['init']]

        # y_train_max = np.asarray([np.max(y_train[:, :, i]) for i in range(5)])
        # y_train_min = np.asarray([np.min(y_train[:, :, i]) for i in range(5)])
        # y_train = (y_train - np.expand_dims(np.expand_dims(y_train_min, 0), 0)) / np.expand_dims(
        #     np.expand_dims(y_train_max - y_train_min, 0), 0)
        # y_test = (y_test - np.expand_dims(np.expand_dims(y_train_min, 0), 0)) / np.expand_dims(
        #     np.expand_dims(y_train_max - y_train_min, 0), 0)
        # y_train_ex = (y_train_ex - np.expand_dims(np.expand_dims(y_train_min, 0), 0)) / np.expand_dims(
        #     np.expand_dims(y_train_max - y_train_min, 0), 0)

        train_control = [[[0, 1 / 57.2958]] * (self.num_time_steps)] * data_dict['y_train'].shape[0]
        train_control_ex = [[[0, 1 / 57.2958]] * 2000] * data_dict['y_train_ex'].shape[0]
        test_control = [[[0, 1 / 57.2958]] * (self.num_time_steps)] * data_dict['y_test'].shape[0]
        test_control_ex = [[[0, 1 / 57.2958]] * 2000] * data_dict['y_test_ex'].shape[0]
        data_dict['train_control'] = np.copy(np.stack(train_control, 0))
        data_dict['test_control'] = np.copy(np.stack(test_control, 0))
        data_dict['train_control_ex'] = np.copy(np.stack(train_control_ex, 0))
        data_dict['test_control_ex'] = np.copy(np.stack(test_control_ex, 0))

        if 1:
            new_test_control = [[[1 / 57.2958, 0]] * 20 + [[0, 0]] * 80] * self.batch_size
            new_test_control_ex = [[[0, 0]] * 2000] * self.batch_size
            data_dict['new_test_control'] = np.copy(np.stack(new_test_control, 0))
            data_dict['new_test_control_ex'] = np.copy(np.stack(new_test_control_ex, 0))

        # data_dict = {
        #     'y_train': y_train,
        #     'train_control': np.stack(train_control,0),
        #     'y_train_ex': y_train_ex,
        #     'train_control_ex': np.stack(train_control_ex,0),
        #     'y_test': y_test,
        #     'test_control': np.stack(test_control,0),
        #     'y_test_ex': y_test_ex,
        #     'test_control_ex': np.stack(test_control_ex,0),
        #     'new_y_test': new_y_test[:, :self.num_time_steps, :],
        #     'new_test_control': np.stack(new_test_control,0),
        #     'new_y_test_ex': new_y_test[:, self.num_time_steps:, :],
        #     'new_test_control_ex': np.stack(new_test_control_ex,0),
        # }
        range_dict_new = {}
        for i, j in range_dict.iteritems():
            range_dict_new[i] = np.squeeze(np.asarray(j))
        return data_dict, range_dict_new


    def train_init(self):
        self.control_pl = tf.placeholder(tf.float32, [self.batch_size, self.num_time_steps, self.dim_control])
        self.y_true_pl = tf.placeholder(tf.float32, [self.batch_size, self.num_time_steps, self.dim_state])
        self.y_pred = self.my_lstm_stack(self.control_pl, self.y_true_pl[:, 0, :])
        self.loss = tf.reduce_mean(tf.abs(self.y_pred - self.y_true_pl))

        ## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.loss, tf.all_variables())
        self.train_op = optimizer.apply_gradients(grads)

        ## training starts ###
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def run_sess(self, control, y, is_train=False):
        ave_loss_val_train_ex = []
        num_itr = y.shape[0] / self.batch_size
        round = y.shape[1] / self.num_time_steps
        assert(control.shape[0] == y.shape[0])
        y_pred_tt = []
        for itr_i in range(num_itr):
            y_batch = y[itr_i * self.batch_size: (itr_i + 1) * self.batch_size]
            c_batch = control[itr_i * self.batch_size: (itr_i + 1) * self.batch_size]
            y0 = y_batch[:, 0:1, :]
            for round_i in range(round):
                yy = np.concatenate([y0, y_batch[:, 1+round_i*self.num_time_steps: (1+round_i)*self.num_time_steps, :]],1)
                cc = c_batch[:, round_i * self.num_time_steps: (1 + round_i) * self.num_time_steps, :]
                feed_dict = {self.control_pl: cc,
                             self.y_true_pl: yy}
                if is_train:
                    _, loss_val, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred], feed_dict)
                else:
                    loss_val, y_pred = self.sess.run([self.loss, self.y_pred], feed_dict)
                if itr_i == 0:
                    y_pred_tt += [y_pred]
                y0 = y_pred[:, -1:, :]
                ave_loss_val_train_ex += [loss_val]
        return np.mean(ave_loss_val_train_ex), np.concatenate(y_pred_tt, 1)

    def train(self):
        self.data_dict, self.range_dict = self.get_data()
        self.y_train =  self.data_dict['y_train']
        self.train_control =  self.data_dict['train_control']
        self.y_train_ex =  self.data_dict['y_train_ex']
        self.train_control_ex =  self.data_dict['train_control_ex']
        self.y_test =  self.data_dict['y_test']
        self.test_control =  self.data_dict['test_control']
        self.y_test_ex =  self.data_dict['y_test_ex']
        self.test_control_ex =  self.data_dict['test_control_ex']
        self.new_y_test =  self.data_dict['new_y_test']
        self.new_test_control =  self.data_dict['new_test_control']
        self.new_y_test_ex = self.data_dict['new_y_test_ex']
        self.new_test_control_ex =  self.data_dict['new_test_control_ex']

        train_loss_val_hist = []
        test_loss_val_hist = []
        train_loss_val_hist_ex = []
        test_loss_val_hist_ex = []
        new_test_loss_val_hist = []
        new_test_loss_val_hist_ex = []

        for ep_i in range(self.max_epoch):

            # training data, for optimization
            loss, y_pred_train = self.run_sess(self.train_control, self.y_train, is_train=True)
            train_loss_val_hist += [loss]

            # extrapolated training data
            loss, y_pred_train_ex = self.run_sess(self.train_control_ex, self.y_train_ex)
            train_loss_val_hist_ex += [loss]

            # testing data
            loss, y_pred_test = self.run_sess(self.test_control, self.y_test)
            test_loss_val_hist += [loss]

            # extrapolated testing data
            loss, y_pred_test_ex = self.run_sess(self.test_control_ex, self.y_test_ex)
            test_loss_val_hist_ex += [loss]

            # new testing data
            loss, new_y_pred = self.run_sess(self.new_test_control, self.new_y_test)
            new_test_loss_val_hist += [loss]

            # new testing data extrapolated
            loss, new_y_pred_ex = self.run_sess(self.new_test_control_ex, self.new_y_test_ex)
            new_test_loss_val_hist_ex += [loss]

            loss_dict = {
                'train': train_loss_val_hist,
                'train_ex': train_loss_val_hist_ex,
                'test': test_loss_val_hist,
                'test_ex': test_loss_val_hist_ex,
                'new': new_test_loss_val_hist,
                'new_ex': new_test_loss_val_hist_ex,
            }
            pred_dict_orig = {
                'train': y_pred_train,
                'train_ex': y_pred_train_ex,
                'test': y_pred_test,
                'test_ex': y_pred_test_ex,
                'new': new_y_pred,
                'new_ex': new_y_pred_ex,
            }
            out_num = 3
            pred_dict = {}
            gt_dict = {}
            range_max = np.expand_dims(self.range_dict['train_max'][:out_num], 1)
            range_min = np.expand_dims(self.range_dict['train_min'][:out_num], 1)
            # pred
            train_gain = y_pred_train[:out_num] * (range_max - range_min) + range_min
            pred_dict['train'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                pred_dict['train'] += [pred_dict['train'][-1] + gain_i]
            pred_dict['train'] = np.transpose(np.squeeze(np.asarray(pred_dict['train'])), (1, 0, 2))
            # gt
            train_gain = self.y_train[:out_num] * (range_max - range_min) + range_min
            gt_dict['train'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                gt_dict['train'] += [gt_dict['train'][-1] + gain_i]
            gt_dict['train'] = np.transpose(np.squeeze(np.asarray(gt_dict['train'])), (1, 0, 2))

            range_max = np.expand_dims(self.range_dict['train_ex_max'][:out_num], 1)
            range_min = np.expand_dims(self.range_dict['train_ex_min'][:out_num], 1)
            # pred
            train_gain = y_pred_train_ex[:out_num] * (range_max - range_min) + range_min
            pred_dict['train_ex'] = [np.expand_dims(self.range_dict['train_ex_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                pred_dict['train_ex'] += [pred_dict['train_ex'][-1] + gain_i]
            pred_dict['train_ex'] = np.transpose(np.squeeze(np.asarray(pred_dict['train_ex'])), (1, 0, 2))
            # gt
            train_gain = self.y_train_ex[:out_num] * (range_max - range_min) + range_min
            gt_dict['train_ex'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                gt_dict['train_ex'] += [gt_dict['train_ex'][-1] + gain_i]
            gt_dict['train_ex'] = np.transpose(np.squeeze(np.asarray(gt_dict['train_ex'])), (1, 0, 2))

            range_max = np.expand_dims(self.range_dict['test_max'][:out_num], 1)
            range_min = np.expand_dims(self.range_dict['test_min'][:out_num], 1)
            # pred
            train_gain = y_pred_test[:out_num] * (range_max-range_min) + range_min
            pred_dict['test'] = [np.expand_dims(self.range_dict['test_init'][:out_num],1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i+1, :]
                pred_dict['test'] += [pred_dict['test'][-1]+gain_i]
            pred_dict['test'] = np.transpose(np.squeeze(np.asarray(pred_dict['test'])), (1,0,2))
            # gt
            train_gain = self.y_test[:out_num] * (range_max - range_min) + range_min
            gt_dict['test'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                gt_dict['test'] += [gt_dict['test'][-1] + gain_i]
            gt_dict['test'] = np.transpose(np.squeeze(np.asarray(gt_dict['test'])), (1, 0, 2))

            range_max = np.expand_dims(self.range_dict['test_ex_max'][:out_num], 1)
            range_min = np.expand_dims(self.range_dict['test_ex_min'][:out_num], 1)
            # pred
            train_gain = y_pred_test_ex[:out_num] * (range_max-range_min) + range_min
            pred_dict['test_ex'] = [np.expand_dims(self.range_dict['test_ex_init'][:out_num],1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i+1, :]
                pred_dict['test_ex'] += [pred_dict['test_ex'][-1]+gain_i]
            pred_dict['test_ex'] = np.transpose(np.squeeze(np.asarray(pred_dict['test_ex'])), (1,0,2))
            # gt
            train_gain = self.y_test_ex[:out_num] * (range_max - range_min) + range_min
            gt_dict['test_ex'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                gt_dict['test_ex'] += [gt_dict['test_ex'][-1] + gain_i]
            gt_dict['test_ex'] = np.transpose(np.squeeze(np.asarray(gt_dict['test_ex'])), (1, 0, 2))

            range_max = np.expand_dims(self.range_dict['new_max'][:out_num], 1)
            range_min = np.expand_dims(self.range_dict['new_min'][:out_num], 1)
            # pred
            train_gain = new_y_pred[:out_num] * (range_max-range_min) + range_min
            pred_dict['new'] = [np.expand_dims(self.range_dict['new_init'][:out_num],1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i+1, :]
                pred_dict['new'] += [pred_dict['new'][-1]+gain_i]
            pred_dict['new'] = np.transpose(np.squeeze(np.asarray(pred_dict['new'])), (1,0,2))
            # gt
            train_gain = self.new_y_test[:out_num] * (range_max - range_min) + range_min
            gt_dict['new'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                gt_dict['new'] += [gt_dict['new'][-1] + gain_i]
            gt_dict['new'] = np.transpose(np.squeeze(np.asarray(gt_dict['new'])), (1, 0, 2))

            range_max = np.expand_dims(self.range_dict['new_ex_max'][:out_num], 1)
            range_min = np.expand_dims(self.range_dict['new_ex_min'][:out_num], 1)
            # pred
            train_gain = new_y_pred_ex[:out_num] * (range_max-range_min) + range_min
            pred_dict['new_ex'] = [np.expand_dims(self.range_dict['new_ex_init'][:out_num],1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i+1, :]
                pred_dict['new_ex'] += [pred_dict['new_ex'][-1]+gain_i]
            pred_dict['new_ex'] = np.transpose(np.squeeze(np.asarray(pred_dict['new_ex'])), (1,0,2))
            # gt
            train_gain = self.new_y_test_ex[:out_num] * (range_max - range_min) + range_min
            gt_dict['new_ex'] = [np.expand_dims(self.range_dict['train_init'][:out_num], 1)]
            for i in range(train_gain.shape[1]):
                gain_i = train_gain[:, i:i + 1, :]
                gt_dict['new_ex'] += [gt_dict['new_ex'][-1] + gain_i]
            gt_dict['new_ex'] = np.transpose(np.squeeze(np.asarray(gt_dict['new_ex'])), (1, 0, 2))



            if ep_i % 10 == 0:
                self.loss_visualization(ep_i, loss_dict, pred_dict, gt_dict)

                import scipy.io as sio
                sio.savemat('beoing_fig1.mat', {
                    'loss': loss_dict['train']
                })
                fig2_data = {}
                for i in range(out_num):
                    fig2_data['train_pred_{}'.format(i)] = pred_dict['train'][i, :, 0]
                    fig2_data['train_gt_{}'.format(i)] = self.y_train[i, :, 0]
                sio.savemat('beoing_fig2.mat', fig2_data)
                fig3_data = {}
                for i in range(out_num):
                    fig3_data['test_extra_pred_{}'.format(i)] = pred_dict['test_ex'][i, :, 0]
                    fig3_data['test_extra_gt_{}'.format(i)] = self.y_test_ex[i, :, 0]
                sio.savemat('beoing_fig3.mat', fig3_data)
                fig4_data = {}
                for i in range(out_num):
                    fig4_data['new_extra_pred_{}'.format(i)] = pred_dict['new_ex'][i, :, 0]
                    fig4_data['new_extra_gt_{}'.format(i)] = self.new_y_test_ex[i, :, 0]
                sio.savemat('beoing_fig4.mat', fig4_data)

            if ep_i % 200 == 0:
                saver = tf.train.Saver()
                saver.save(self.sess, 'saved_models/boeing_lstm1_ep_{}.ckpt'.format(ep_i))

    def loss_visualization(self, ep_i, loss_dict, pred_dict, gt_dict):
        import matplotlib.pyplot as plt
        plt.plot(loss_dict['train'], 'k', label='training loss')
        plt.plot(loss_dict['train_ex'], 'k--', label='training extrap loss')
        plt.plot(loss_dict['test'], 'b', label='testing loss')
        plt.plot(loss_dict['test_ex'], 'b--', label='testing extrap loss')
        plt.plot(loss_dict['new'], 'r', label='new testing loss')
        plt.plot(loss_dict['new_ex'], 'r--', label='new testing extrap loss')
        plt.legend()
        plt.savefig('test.png')
        plt.close()

        for idx in range(3):
            plt.figure(figsize=(15, 8))
            for j in range(5):
                plt.subplot(6, 5, 5 * 0 + j + 1)
                plt.plot(gt_dict['train'][idx, :, j], label='train gt')
                plt.plot(pred_dict['train'][idx, :, j], label='train pred')
                plt.legend()
                plt.subplot(6, 5, 5 * 1 + j + 1)
                plt.plot(gt_dict['train_ex'][idx, :, j], label='train_ex gt')
                plt.plot(pred_dict['train_ex'][idx, :, j], label='train_ex pred')
                plt.legend()
                plt.subplot(6, 5, 5 * 2 + j + 1)
                plt.plot(gt_dict['test'][idx, :, j], label='test gt')
                plt.plot(pred_dict['test'][idx, :, j], label='test pred')
                plt.legend()
                plt.subplot(6, 5, 5 * 3 + j + 1)
                plt.plot(gt_dict['test_ex'][idx, :, j], label='test_ex gt')
                plt.plot(pred_dict['test_ex'][idx, :, j], label='test_ex pred')
                plt.legend()
                plt.subplot(6, 5, 5 * 4 + j + 1)
                plt.plot(gt_dict['new'][idx, :, j], label='new gt')
                plt.plot(pred_dict['new'][idx, :, j], label='new pred')
                plt.legend()
                plt.subplot(6, 5, 5 * 5 + j + 1)
                plt.plot(gt_dict['new_ex'][idx, :, j], label='new_ex gt')
                plt.plot(pred_dict['new_ex'][idx, :, j], label='new_ex pred')
                plt.legend()
            plt.savefig('dp_{}_ep_{}.png'.format(idx, ep_i))
            plt.close()


if __name__ == "__main__":
    pde_lstm = PDE_LSTM()
    pde_lstm.train_init()
    pde_lstm.train()
