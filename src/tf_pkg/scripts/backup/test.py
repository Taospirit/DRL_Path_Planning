#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge, CvBridgeError
import datetime
import random
import shutil
import os
import matplotlib.pyplot as plt
from pathplaner import *
from Models import *
import time
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
import rospy
import cv2
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


# import tensorflow as tf

topic = '/' + 'jackal10' + '/front/left/image_raw'
topic1 = '/' + 'jackal10' + '/front/scan'

# class test():
#     def __init__(self):
# rospy.init_node('gazebo_info_test', anonymous=True)
#         print ('sub left image')
#         rospy.Subscriber('/' + 'jackal10' +'/front/left/image_raw', Image, self.image_callback)
#         print ('after sub')
#         # rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)

#     # def gazebo_states_callback(self, data):
#     #     pass


#     def image_callback(self, data):
#         print ('callback')
#         try:
#             a = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
#             print ('ok')
#         except CvBridgeError as e:
#             # print(e)
#             print ('not ok')
def image_callback(data):
    print('callback')
    try:
        a = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
        print('ok')
    except CvBridgeError as e:
        # print(e)
        print('not ok')


def main():
    rospy.init_node('gazebo_image_info', anonymous=True)
    rospy.Subscriber(topic, Image, image_callback)
    rospy.Subscriber(topic1, LaserScan, laser_states_callback)
    rospy.spin()


def laser_states_callback(data):
    a = 1
    b = 2

    print(type(data))
    # print (np.array(data).shape)
    print(len(data.ranges))


if __name__ == "__main__":
    main()
