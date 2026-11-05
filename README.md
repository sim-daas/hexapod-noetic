# hexapod-noetic

ROS 1 Noetic deployment package for the **Interbotix PhantomX Mark IV hexapod** running reinforcement learning policies trained in Isaac Lab (`hexasim`).

## Features
- Subscribes to `/pxmark4/joint_states` and onboard IMU.
- Reconstructs deployable 111-D observation frame (`o_p`).
- Executes exported policy via `onnxruntime` on Raspberry Pi 4 CPU at 50 Hz.
- Publishes position commands to `/pxmark4/commands/joint_group`.

## Build
Inside your catkin workspace on the robot:
```bash
cd ~/interbotix_ws
catkin_make --pkg hexapod_noetic
source devel/setup.bash
```
