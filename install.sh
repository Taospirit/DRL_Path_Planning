#!/bin/sh

# sudo apt install ros-kinetic-robot-localization \\
# ros-kinetic-move-base \\
# ros-kinetic-interactive-marker-twist-server \\ 
# ros-kinetic-hector-gazebo-plugins \\ -y


sudo apt-get install -y ros-kinetic-robot-localization
sudo apt-get install -y ros-kinetic-move-base
sudo apt-get install -y ros-kinetic-interactive-marker-twist-server
sudo apt-get install -y ros-kinetic-joint-state-controller
sudo apt-get install -y ros-kinetic-diff-drive-controller
sudo apt-get install -y ros-kinetic-lms1xx 
sudo apt-get install -y ros-kinetic-controller-manager
sudo apt-get install -y ros-kinetic-gazebo-ros-control