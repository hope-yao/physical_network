import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio


aa=np.load('./temp/k1_numerical.npy')
bb=np.load('./temp/k1_train_drrnn.npy')
cc=np.load('./temp/k1_test_drrnn.npy')
plt.plot(aa[0,:,1],'k',label='k1_numerical')
plt.plot(bb[0,:,1],'b',label='k1_train_drrnn')
plt.plot(cc[0,:,1],'r',label='k1_test_drrnn')
plt.legend()
plt.show()


aa=np.load('./temp/k2_numerical.npy')
bb=np.load('./temp/k2_train_drrnn.npy')
cc=np.load('./temp/k2_test_drrnn.npy')
plt.plot(aa[0,:,1],'k',label='k2_numerical')
plt.plot(bb[0,:,1],'b',label='k2_train_drrnn')
plt.plot(cc[0,:,1],'r',label='k2_test_drrnn')
plt.legend()
plt.show()


aa=np.load('./temp/k4_numerical.npy')
bb=np.load('./temp/k4_train_drrnn.npy')
cc=np.load('./temp/k4_test_drrnn.npy')
plt.plot(aa[0,:,1],'k',label='k4_numerical')
plt.plot(bb[0,:,1],'b',label='k4_train_drrnn')
plt.plot(cc[0,:,1],'r',label='k4_test_drrnn')
plt.legend()
plt.show()


y = sio.loadmat('./data/problem1_1129.mat')['y']
y2_end = y[:,-1,1]
sns.distplot(y2_end,label='numerical', hist=False,kde_kws={"color": "k"})
yy=np.load('./temp/k1_test_dist_y2.npy')
sns.distplot(yy,label='DR-RNN_1', hist=False)
yy=np.load('./temp/k2_test_dist_y2.npy')
sns.distplot(yy,label='DR-RNN_2', hist=False)
yy=np.load('./temp/k4_test_dist_y2.npy')
sns.distplot(yy,label='DR-RNN_4', hist=False)
plt.legend()
plt.show()



y = sio.loadmat('./data/problem1_1129.mat')['y']
y3_end = y[:,-1,2]
sns.distplot(y3_end,label='numerical', hist=False,kde_kws={"color": "k"})
yy=np.load('./temp/k1_test_dist_y3.npy')
sns.distplot(yy,label='DR-RNN_1', hist=False)
yy=np.load('./temp/k2_test_dist_y3.npy')
sns.distplot(yy,label='DR-RNN_2', hist=False)
yy=np.load('./temp/k4_test_dist_y3.npy')
sns.distplot(yy,label='DR-RNN_4', hist=False)
plt.legend()
plt.show()


# # Insert in to training code
# np.save('./temp/k2_train_drrnn.npy', sess.run(y_pred, {y_true: y_test[:batch_size]}))
# np.save('./temp/k2_test_drrnn.npy', sess.run(y_pred_testing, {y_true: y_test[:batch_size]}))
# np.save('./temp/k2_numerical.npy', y_test[:batch_size])
#
# test_pred = []
# for i in range(60):
#     test_pred += [sess.run(y_pred_testing, {y_true: y_test[15*i:15*(i+1)]})]
# np.save('./temp/k4_test_dist_y2.npy',np.reshape(np.asarray(test_pred)[:,:,-1,1],60*15))
# np.save('./temp/k4_test_dist_y3.npy',np.reshape(np.asarray(test_pred)[:,:,-1,2],60*15))


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
