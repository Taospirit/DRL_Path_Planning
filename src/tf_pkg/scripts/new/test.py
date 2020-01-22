from gazebo_env import gazebo_env
import tensorflow as tf
import numpy as np
import time
import os

class a():
    def __init__(self):
        self.b = 1
        self.c = 2

    def b_c(self):
        print ('euql b = c')
        self.b = self.c
    def c_b(self):
        print ('euql c = b')
        self.c = self.b

    def get_b(self):
        print ('b = ', self.b)
    
    def set_b(self, b_):
        print ('set b = ', b_)
        self.b = b_
    
    def set_c(self, c_):
        print ('set c = ', c_)
        self.c = c_

    def get_c(self):
        print ('c = ', self.c)


def conv2d(x, w, s, b):
    return tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME')+b)


def get_totoal_reward(r_l, gamma):
    if len(r_l) == 1:
        return r_l[0]
    else:
        return r_l.pop(0) + gamma * get_totoal_reward(r_l, gamma)

if __name__ == "__main__":
    # env = gazebo_env()
    # a = a()
    # a.get_b()
    # a.get_c()

    # a.b_c()
    # a.get_b()
    # a.set_b(555)
    # a.get_b()
    # a.get_c()
    # a.set_c(222)
    # a.get_c()
    # a.get_b()

    # a.c_b()
    # a.get_c()
    # a.get_b()
    
    # a.set_b(111)
    # a.get_c()

    #=====test tf
    # data = np.zeros((4, 10))
    # print (data.shape)
    # c = tf.unstack(data, axis=1)
    # # print (c.get_shape().as_list())

    # input_ = tf.Variable(tf.random_normal([1, 80, 80, 4]))
    # filter1 = tf.Variable(tf.random_normal([8, 8, 4, 32]))
    # filter2 = tf.Variable(tf.random_normal([4, 4, 32, 64]))
    # filter3 = tf.Variable(tf.random_normal([3, 3, 64, 64]))

    # b1 = tf.Variable(tf.random_normal([32]))
    # b2 = tf.Variable(tf.random_normal([64]))
    # b3 = tf.Variable(tf.random_normal([64]))
    
    # op1 = conv2d(input_, filter1, 4, b1)
    # op2 = conv2d(op1, filter2, 2, b2)
    # op3 = conv2d(op2, filter3, 1, b3)
    # # op1 = tf.nn.relu(tf.nn.conv2d(input_, filter1, strides=[1, 4, 4, 1], padding='SAME')+b1)
    # # op2 = 
    # img_data = tf.reshape(op3, [-1, 10*10*64])
    # laser_data  = [tf.constant(1.0, shape=(1, 360))] * 4
    # # laser_data = tf.Variable(tf.random_normal([1, 360, 4]))
    # # laser_data = tf.unstack(laser, axis=1)
    # cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
    # laser_out, state = tf.nn.static_rnn(inputs=laser_data, cell=cell, dtype=tf.float32)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     print (sess.run(tf.shape(img_data)))
    #     print (sess.run(tf.shape(laser_out[-1])))
    #     print (sess.run(tf.shape(state)))

    #====test total_reward() ====
    # start = time.time()
    # r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # ans = get_totoal_reward(r, 0.5)
    # during =  time.time() - start
    # print ('ans is {}, during {:.3f}'.format(ans, during))

    #=====test file I/O
    dir_path = os.path.dirname(__file__)
    print (dir_path)
    # file_name = os.path.join(dir_path, 'dqn_no_bos.txt')
    # fo = open(file_name, 'w')
    # fo.write('ep:123, a_n:15, epsilon:0.100, total_reward:172, step:11111\n')
    # fo.write('ep:123, a_n:15, epsilon:0.100, total_reward:172, step:11111\n')
    # fo.close()
    # with open(file_name) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         data = line.split(', ')
    #         data_ = []
    #         for item in data:
    #             data_.append(item.split(':')[1])
    #         print (data_)