import numpy as np
import matplotlib.pyplot as plt
import os

num = 30
r = 1.0
# for i in range(30):
#     start_x, start_y = [], []
#     goal_x, goal_y = [], []
def main():
    plt.figure(figsize=(10, 10))

    dir_path = os.path.dirname(os.path.abspath(__file__))
    # filename = 't.txt'
    s_x_file = '14_static_obstacle_start_point_x.txt'
    s_y_file = '14_static_obstacle_start_point_y.txt'

    g_x_file = '14_static_obstacle_goal_point_x.txt'
    g_y_file = '14_static_obstacle_goal_point_y.txt'
    
    s_x_path = os.path.join(dir_path, s_x_file)
    s_y_path = os.path.join(dir_path, s_y_file)
    g_x_path = os.path.join(dir_path, g_x_file)
    g_y_path = os.path.join(dir_path, g_y_file)

    s_x, s_y = read_file(s_x_path), read_file(s_y_path)
    g_x, g_y = read_file(g_x_path), read_file(g_y_path)

    # plt.plot(s_x, s_y, )
    plt.plot(s_x, s_y, marker='o',color='lightgreen', markersize=10)
    plt.plot(g_x, g_y, marker='*',color='deeppink', markersize=10)
    # a, b = (g_x, g_y)
    # theta = np.arange(0, 2*np.pi, 0.01)
    # x = a + r * np.cos(theta)
    # y = b + r * np.sin(theta)
    # plt.plot(x, y, label='line', color='dodgerblue', linewidth=2.0)
    # 设置坐标轴范围
    plt.xlim((-18, 18))
    plt.ylim((-18, 18))
    # 设置坐标轴刻度
    plt.xticks(np.arange(-18, 24, 6))
    plt.yticks(np.arange(-18, 24, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(dir_path+'/1.png')
# print (filepath)
# print (os.path.dirname(cur_path))
def read_file(f_path):
    store_list = []
    with open(f_path, 'r') as f:
        lines = f.readlines()
        print (len(lines))
        for l in lines:
            # print (l)
            store_list.append(l)
    return store_list

if __name__ == "__main__":
    main()