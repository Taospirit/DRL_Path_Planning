#!/bin/sh

# sudo apt-get install -y ros-kinetic-robot-localization
# sudo apt-get install -y ros-kinetic-move-base
# sudo apt-get install -y ros-kinetic-interactive-marker-twist-server
# sudo apt-get install -y ros-kinetic-joint-state-controller
# sudo apt-get install -y ros-kinetic-diff-drive-controller
# sudo apt-get install -y ros-kinetic-lms1xx 
# sudo apt-get install -y ros-kinetic-controller-manager
# sudo apt-get install -y ros-kinetic-gazebo-ros-control

sudo apt-get install -y ros-kinetic-robot-localization \ 
                        ros-kinetic-move-base \ 
                        ros-kinetic-interactive-marker-twist-server \
                        ros-kinetic-joint-state-controller \
                        ros-kinetic-diff-drive-controller \
                        ros-kinetic-lms1xx

# cd ./src
# git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo
# git clone https://github.com/ros-drivers/pointgrey_camera_driver