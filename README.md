# DRL_Path_Planning

This is a DRL(Deep Reinforcement Learning) platform built with Gazebo for the purpose of robot's adaptive path planning.

# Environment

## Software

    Ubuntu 16.04
    ROS Kinect
    Python 2.7.12
    tensorflow 1.12.0

---    
## 1. 知乎专栏：<https://zhuanlan.zhihu.com/p/79712897>
## 2. ros-pkg 依赖：  
    sudo apt-get install -y \
        ros-kinetic-robot-localization \ 
        ros-kinetic-move-base \ 
        ros-kinetic-interactive-marker-twist-server \
        ros-kinetic-joint-state-controller \
        ros-kinetic-diff-drive-controller \
        ros-kinetic-lms1xx \ # gazebo中激光雷达模型


## 3. Problem & Solution
### 1. 编译问题:
#### P1: pointgrey_camera_driver编译不过的问题
缺乏pointgrey_camera_driver这个包, 需要额外安装
#### S1: 在src目录下下载该包:
```shell
cd src/
git clone https://github.com/ros-drivers/pointgrey_camera_driver
```
完后继续用 `catkin_make` 编译
#### P2: 编译时卡在pointgrey_camera_driver
#### S2: 移除pointgrey_camera_driver, 完后再移动回来(不放回会有新错误)
相关文件已经生成了, 所以这里先把这个包移动到~/目录, 编译后再放回
```shell
cd src
mv -r ./pointgrey_camera_driver ~/
cd .. && catkin_make
mv -r ~/pointgrey_camera_driver ./src
```
#### P3: 编译时fatal error: flycapture/FlyCapture2.h: 
#### S4: 

#### P4: ..

### 2. 代码运行问题
#### P3: 启动10个动态障碍物时终端读取xarco参数命令报错
#### S3: 可能python环境问题, 建议注释掉~/.bashrc中的conda环境配置, 保存后重启终端再尝试, python2.7使用正常.

#### P4: 激光雷达维度参数问题
gazebo仿真环境中采用的是SICK LMS1xx雷达, 它的xarco中的激光点云采样周期是720, 所以比代码里预设的360大了一倍. 具体可见[lms1xx_git_repo](https://github.com/clearpathrobotics/LMS1xx/blob/melodic-devel/urdf/sick_lms1xx.urdf.xacro), 其中sample_size=720.
#### S4: 用函数将激光数据压缩一半到360
```python
def laser_resize(self, laser_data):
    laser_resized = []
    for i in range(0, len(laser_data), 2):
        tmp = laser_data[i] + laser_data[i+1] 
        laser_resized.append(tmp/2)
    return laser_resized
```
#### P4: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize

#### S4: 在前面添加一下代码, 指定gpu设备?
```shell
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
```
