# This is copy pasted from https://github.com/Roll920/ThiNet_Code/blob/master/ThiNet_TPAMI/VGG16/compress_model.py#L152
# Could not import the entire thing because the caffe library does not work on non-CUDA computers as far as I can read.
import numpy as np
import time


# use greedy method to select index
# x:N*64 matrix, N is the instance number, 64 is channel number
def value_sum(x, y, compress_rate):
    # 1. set parameters
    x = np.mat(x)
    y = np.mat(y)
    goal_num = int(x.shape[1] * compress_rate)
    index = []

    # 2. select
    y_tmp = y
    for i in range(goal_num):
        min_value = float("inf")
        s = time.time()
        for j in range(x.shape[1]):
            if j not in index:
                tmp_w = (x[:, j].T*y_tmp)[0, 0]/(x[:,j].T*x[:,j])[0,0]
                tmp_value = np.linalg.norm(y_tmp-tmp_w*x[:,j])
                if tmp_value < min_value:
                    min_value = tmp_value
                    min_index = j
        index.append(min_index)
        selected_x = x[:, index]
        w = np.linalg.pinv(selected_x.T * selected_x) * selected_x.T * y
        y_tmp = y - selected_x * w
        print('goal num={0}, channel num={1}, i={2}, loss={3:.3f}, time={4:.3f}'.format(goal_num, x.shape[1], i,
                                                                                        min_value, time.time() - s))

    # 3. return index
    index = np.array(list(index))
    index = np.sort(index)

    # 4.least square
    selected_x = x[:, index]
    w = (selected_x.T * selected_x).I * (selected_x.T * y)
    w = np.array(w)

    loss = np.linalg.norm(y - selected_x * w)
    print('loss with w={0:.3f}'.format(loss))
    return index, w


# use greedy method to select index
# x:N*64 matrix, N is the instance number, 64 is channel number
def value_sum_another(x, y, compress_rate):
    # 1. set parameters
    goal_num = int(x.shape[1] * compress_rate)
    index = []

    # 2. select
    for i in range(goal_num):
        min_value = float("inf")
        print('goal num=%d, channel num=%d, i=%d') % (
            goal_num, x.shape[1], i)
        old_sum = np.sum(np.take(x, index, axis=1), axis=1)
        for j in range(x.shape[1]):
            if j not in index:
                tmp_value = np.sum((old_sum + np.take(x, j, axis=1) - y.reshape(-1)) ** 2)
                if tmp_value < min_value:
                    min_value = tmp_value
                    min_index = index[:]
                    min_index.append(j)
        index = min_index

    # 3. return index
    index = np.array(list(index))
    index = np.sort(index)

    # 4.least square
    selected_x = np.mat(x[:, index])
    w = (selected_x.T * selected_x).I * (selected_x.T * y)
    w = np.array(w)
    return index, w
