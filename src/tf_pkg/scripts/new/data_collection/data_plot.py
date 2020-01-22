import matplotlib.pyplot as plt
import os

class data_plot():
    def __init__(self):
        pass
    def get_data(self):
        pass
    def plot_data(self):
        pass
    def show(self):
        pass


if __name__ == "__main__":
    # plot = data_plot()
    # plot.show()
    # f = 'dqn_test'
    f = 'dqn_no_obs'
    f = os.path.join(os.path.dirname(__file__), f) + '.txt'
    with open(f) as f:
        lines = f.readlines()
        ep = []
        r = []
        for line in lines:
            data = line.split(', ')
            for item in data:
                data_ = item.split(' ')
                if data_[0] == 'ep':
                    ep.append(int(data_[-1]))
                if data_[0] == 'total_reward':
                    r.append(float(data_[-1]))
            # print (data)

    # print (ep)
    # print (r)
    plt.plot(ep, r)
    plt.xlabel('num_episode')
    plt.ylabel('total_reward')
    plt.show()