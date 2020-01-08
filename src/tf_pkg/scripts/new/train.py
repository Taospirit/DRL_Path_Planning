import rospy
from gazebo_env import gazebo_env
from DQN_tf import DeepQNetwork
num_episode = 100000
num_action = 200

def train():
    step = 0
    for episode in range(num_episode):
        # initial observation
        state = env.reset()
        # for n_action in range(num_action):
        action_n = 0
        while True:
            # action_n = 0
            # agent choose action by DQN law
            action = agent.choose_action(state)
            # agent take action and get next observation and reward
            state_, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, state_)

            if (step > 2000) and (step % 5 == 0):
                agent.learn()
                
            # agent.learn()
            # swap state
            state = state_
            # break while loop when end of this episode
            action_n += 1
            # print ('action do {} times'.format(action_n))
            
            if done or action_n > num_action:
                print ('ep {}, action_do {}, epsilon {:.3f}, step {}'.format(episode, action_n, agent.epsilon, step))
                break

            step += 1
    # end of game
    # print('game over')
    # env.destroy()
    print ('======learn_over======')
    rospy.spin()


if __name__ == "__main__":
    env = gazebo_env()
    agent = DeepQNetwork(env.n_actions,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      replace_target_iter=200,
                      memory_size=50000,
                      num_episode=num_episode,
                      # output_graph=True
                      )
    # RL.plot_cost()
    train()
