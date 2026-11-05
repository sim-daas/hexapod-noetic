cmake_minimum_required(VERSION 3.0.2)
project(hexapod_noetic)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  interbotix_xs_msgs
  interbotix_xshexapod_control
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    geometry_msgs
    std_msgs
    interbotix_xs_msgs
    interbotix_xshexapod_control
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/policy_runner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
