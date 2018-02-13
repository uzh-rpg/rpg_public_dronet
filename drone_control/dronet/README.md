# DroNet: Learning to fly by driving

This folder contains code and instruction to run DroNet code on a Bebop drone.

## Introduction

The bridge between the DroNet Keras code and the Bebop control is implemented in ROS.

## Installation and Setup

### Step 1: Install ROS

It is necessary for you to install [ROS](http://wiki.ros.org/ROS/Installation) to have the basic tools available. The project was tested under ROS [indigo](http://wiki.ros.org/indigo/Installation/Ubuntu), but you can use any other version without problems.

### Step 2: Build your workspace

The folder containing all the related code for a project is usually defined as `workspace'.
Create your own workspace following [these instructions](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) and call it ```bebop_ws'''.

### Step 3: Setup Bebpop Autonomy

In this step, we will bridge our ROS workspace with the bebop drone.
To do it, just follow [these instructions](http://bebop-autonomy.readthedocs.io/en/latest/installation.html).

___Be sure to read properly all the instructions about how to run the driver, how to send commands and how to read data from the drone available on the website!___

### Step 4: Install DroNet perception package

After we're done with the bridge between ROS and Bebop, we now need to connect the DroNet Keras code to ROS. This is again very easy to do, since ROS has both a cpp and python interface.

You can find code to do this step in the folder [dronet_perception](./dronet_perception).
Add this folder to your workspace:

```
mkdir ~/bebop_ws/dronet

cp -r YOUR_PATH/dronet_perception ~/bebop_ws/dronet
```

Now build the package:

```
cd ~/bebop_ws/dronet/dronet_perception

catkin build --this
```

### Step 5: Install DroNet control package

It is now time to have an interface that converts the output of DroNet to control commands for the Bebop drone.
This is implemented in the folder [dronet_control](./dronet_control).
Again, build this folder into your workspace.

```
cp -r YOUR_PATH/dronet_control ~/bebop_ws/dronet

cd ~/bebop_ws/dronet/dronet_perception

catkin build --this
```

### Step 6: Some tests

To make sure that everything is as expected, try to run some qualitative tests. Example:

1) See if you can connect to the drone
2) See if you can receive images from it with [rqt_img_view](http://wiki.ros.org/rqt_image_view)
3) See if you can publish control commands through the terminal

