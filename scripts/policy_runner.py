#!/usr/bin/env python3
"""
Policy runner skeleton for Interbotix PhantomX Mark IV hexapod.
Executes trained RL policy (ONNX Runtime) in ROS 1 Noetic.
"""

import rospy
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand


def main():
    rospy.init_node("hexapod_policy_runner")
    rospy.loginfo("Hexapod RL policy runner initialized.")


if __name__ == "__main__":
    main()
