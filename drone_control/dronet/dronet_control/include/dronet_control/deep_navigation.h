#pragma once

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <geometry_msgs/Twist.h>
#include "dronet_perception/CNN_out.h"

namespace deep_navigation
{

class deepNavigation final
{

public:
  deepNavigation(const ros::NodeHandle& nh,
                 const ros::NodeHandle& nh_private);
  deepNavigation() : deepNavigation(ros::NodeHandle(), ros::NodeHandle("~") ) {}

  void run();

private:

  // ROS 
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber deep_network_sub_;
  ros::Subscriber state_change_sub_;
  ros::Publisher desired_velocity_pub_;

  // Callback for networks outputs
  void deepNetworkCallback(const dronet_perception::CNN_out::ConstPtr& msg);
  void stateChangeCallback(const std_msgs::Bool& msg);
  double probability_of_collision_;
  double steering_angle_;
  bool use_network_out_;  // If True, it will use the network out, else will use zero


  // Parameters
  void loadParameters();
  double alpha_velocity_, alpha_yaw_; // Smoothers for CNN outs
  double critical_prob_coll_;
  double max_forward_index_;
  std::string name_;

  // Internal variables

  double desired_forward_velocity_;
  double desired_angular_velocity_;
  geometry_msgs::Twist cmd_velocity_;

};

class HeightChange{

public:
  HeightChange(const ros::NodeHandle& nh,
                 const ros::NodeHandle& nh_private);
  HeightChange() : HeightChange(ros::NodeHandle(), ros::NodeHandle("~") ) {}

  virtual ~HeightChange();

  void run();

private:

  // ROS 
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber is_up_sub_;
  ros::Subscriber move_sub_;
  ros::Subscriber alt_c_sub_;

  ros::Publisher desired_velocity_pub_;

  // Callback for networks outputs
  void is_up(const std_msgs::Empty& msg);
  void move(const std_msgs::Empty& msg);
  void change_altitude(const std_msgs::Empty& msg);

  std::string name_;
  bool is_up_;
  bool change_altitude_, should_move_;

};

}
