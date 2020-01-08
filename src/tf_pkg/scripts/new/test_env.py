import rospy, rospkg
import math
from gazebo_msgs.msg import ModelStates



class gazebo_env():
    def __init__(self):
        rospy.init_node('teste_env', anonymous=True)
        self.agent_goal = {'x': 0, 'y': 0}
        self.agent_position = {'x': 0, 'y': 0}
        self.dist = lambda p1, p2: math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
        gazebo_topic_name = '/gazebo/model_states'
        rospy.Subscriber(gazebo_topic_name, ModelStates, self.gazebo_states_callback)

    def gazebo_states_callback(self, data):
        self.gazebo_obs_states = [{'x':0, 'y':0} for name in data.name if 'obs' in name]

        for i in range(len(data.name)):
            p_x = data.pose[i].position.x
            p_y = data.pose[i].position.y
            name = str(data.name[i])

            if 'obs' in name:
                self.gazebo_obs_states[int(data.name[i][-1])] = {'x':p_x, 'y':p_y}
            elif name == 'agent_point_goal':
                self.agent_goal['x'] = p_x
                self.agent_goal['y'] = p_y
            elif name == 'agent':
                self.agent_position['x'] = p_x
                self.agent_position['y'] = p_y

    # def get_reward(self, dist_rate=0.1, cmd_rate=0.01):
    #     reward = 0
    #     # print (type(self.cmd_vel_last['v']), type(self.cmd_vel.linear.x))
    #     cmd_vel_reward = abs(self.cmd_vel_last['v'] - self.cmd_vel.linear.x) + abs(self.cmd_vel_last['w'] - self.cmd_vel.angular.z)
    #     reward += -cmd_rate * cmd_vel_reward
    #     reward += dist_rate * (self.goal_dist_last - self.goal_dist)

    #     if self.info == 1: # coll
    #         reward += self.reward_near_obs
    #     if self.info == 2: # arrive at goal
    #         reward += self.reward_near_goal

        # return reward

if __name__ == "__main__":
    env = gazebo_env()
    while not rospy.is_shutdown():
        print (env.dist(env.agent_goal, env.agent_position))

    rospy.spin()