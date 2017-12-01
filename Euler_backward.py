from numpy.linalg import norm
import numpy as np
delta_t = 1e-1

def dev(y):
    return np.asarray([y[0] * y[2], -y[1] * y[2], -y[0] ** 2 + y[1] ** 2])


def f_newton(y):
    y1, y2, y3 = y
    yy1 = y1 - delta_t * y1 * y3
    yy2 = y2 + delta_t * y2 * y3
    yy3 = y3 - delta_t * (-y1 ** 2 + y2 ** 2)
    return np.asarray([yy1, yy2, yy3])


def df_newton(y):
    y1, y2, y3 = y
    yy1 = 1 - delta_t * 1 * y3
    yy2 = 1 + delta_t * 1 * y3
    yy3 = 1 - delta_t * (-y1 ** 2 * 2 * y2)
    return np.asarray([yy1, yy2, yy3])


def newtons_method(f, df, y_init, eps=1e-6):
    y = y_init  # initial guess of newtown's method
    cnt = 0
    while norm(f_newton(y) - y) > eps:
        y = y - ( f_newton(y)-y ) / (df_newton(y)+1e-10)
        cnt += 1
        if cnt > 500:
            break
    print('y: ', cnt, y, norm(f_newton(y) - y))
    return y

def main(x):
    y_init = np.asarray([0., 0.1 * x, 0.])
    delta_t = 0.1
    t_final = 10
    t = 0
    y_hist = []
    while (t < t_final):
        y_init = newtons_method(f_newton, df_newton, y_init)
        y_hist += [y_init]
        t += delta_t
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(0,10.1,0.1),np.asarray(y_hist)[:,2])
    # plt.show()
    return np.asarray(y_hist)

if __name__ == '__main__':
    time_series = []
    for i in range(1500):
        time_series += [main(-1+2/1500.*i)]
    time_series = np.asarray(time_series)
    idx = np.random.choice(len(time_series), len(time_series), replace=False)
    np.savez('./data/problem1.npz',y_train=time_series[idx[:500]],y_test=time_series[idx[500:]])
    print('done')

