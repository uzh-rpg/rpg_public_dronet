#include "dronet_control/deep_navigation.h"


namespace deep_navigation
{

deepNavigation::deepNavigation(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& nh_private)
  :   nh_(nh),
      nh_private_(nh_private),
      name_(nh_private.getNamespace())
{
  ROS_INFO("[%s]: Initializing Deep Control Node", name_.c_str());
  loadParameters();
  deep_network_sub_ = nh_.subscribe("cnn_predictions", 1, &deepNavigation::deepNetworkCallback, this);
  state_change_sub_ = nh_.subscribe("state_change", 1, &deepNavigation::stateChangeCallback, this);

  desired_velocity_pub_ = nh_.advertise < geometry_msgs::Twist > ("velocity", 1);

  steering_angle_ = 0.0;
  probability_of_collision_ = 0.0;

  // Aggressive initialization
  desired_forward_velocity_ = max_forward_index_;
  desired_angular_velocity_ = 0.0;

  use_network_out_ = false;

}


void deepNavigation::run()
{

  ros::Duration(2.0).sleep();

  ros::Rate rate(30.0);

  while (ros::ok())
  {

    // Desired body frame velocity to world frame
    double desired_forward_velocity_m = (1.0 -  probability_of_collision_) * max_forward_index_;
    if (desired_forward_velocity_m <= 0.0)
    {
      ROS_INFO("Detected negative forward velocity! Drone will now stop!");
      desired_forward_velocity_m  = 0;
    }

    // Low pass filter the velocity and integrate it to get the position
    desired_forward_velocity_ = (1.0 - alpha_velocity_) * desired_forward_velocity_
        + alpha_velocity_ * desired_forward_velocity_m;

    ROS_INFO("Desired_Forward_Velocity [0-1]: %.3f ", desired_forward_velocity_);
    
    // Stop if velocity is prob of collision is too high
    if (desired_forward_velocity_ < ((1 - critical_prob_coll_) * max_forward_index_))
    {
      desired_forward_velocity_ = 0.0;
    }


    // Low pass filter the angular_velocity (Remeber to tune the bebop angular velocity parameters)
    desired_angular_velocity_ = (1.0 - alpha_yaw_) * desired_angular_velocity_ + alpha_yaw_ * steering_angle_;

    ROS_INFO("Desired_Angular_Velocity[0-1]: %.3f ", desired_angular_velocity_);

    // Prepare command velocity
    cmd_velocity_.linear.x = desired_forward_velocity_;
    cmd_velocity_.angular.z = desired_angular_velocity_;

    // Publish desired state
    if (use_network_out_)
    {
        desired_velocity_pub_.publish(cmd_velocity_);
    }
    else
        ROS_INFO("NOT PUBLISHING VELOCITY");

    ROS_INFO("Collision Prob.: %.3f - OutSteer: %.3f", probability_of_collision_, steering_angle_);
    ROS_INFO("--------------------------------------------------");

    rate.sleep();

    ros::spinOnce();

  }

}

void deepNavigation::deepNetworkCallback(const dronet_perception::CNN_out::ConstPtr& msg)
{

  probability_of_collision_ = msg->collision_prob;
  steering_angle_ = msg->steering_angle;

  // Output modulation
  if (steering_angle_ < -1.0) { steering_angle_ = -1.0;}
  if (steering_angle_ > 1.0) { steering_angle_ = 1.0;}

}

void deepNavigation::stateChangeCallback(const std_msgs::Bool& msg)
{
    //change current state
    use_network_out_ = msg.data;
}

void deepNavigation::loadParameters()
{

  ROS_INFO("[%s]: Reading parameters", name_.c_str()); 
  nh_private_.param<double>("alpha_velocity", alpha_velocity_, 0.3);
  nh_private_.param<double>("alpha_yaw", alpha_yaw_, 0.5);
  nh_private_.param<double>("max_forward_index", max_forward_index_, 0.2);
  nh_private_.param<double>("critical_prob", critical_prob_coll_, 0.7);

}

} // namespace deep_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "deep_navigation");
  deep_navigation::deepNavigation dn;

  dn.run();

  return 0;
}
