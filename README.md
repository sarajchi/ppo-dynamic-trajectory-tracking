# PPO-Based Dynamic End-Effector Trajectory Tracking in MuJoCo

This project implements a reinforcement learning framework for dynamic end-effector trajectory tracking using Proximal Policy Optimisation (PPO) in MuJoCo.

A simulated Fetch robotic manipulator is trained to follow a continuous moving Cartesian trajectory while maintaining smooth and stable motion under uncertainty.



<img width="1280" height="960" alt="Adobe Express - Kp_50_Rd_0 2 (4)" src="https://github.com/user-attachments/assets/bd29acec-b217-4454-b6c5-8f648e0ed620" />


## Training Summary

The PPO policy was trained for approximately 5 million timesteps.

Final training statistics:

| Metric | Value |
|---|---|
| Total Timesteps | 5,001,216 |
| Success Rate | 100% |
| Mean Episode Reward | -19 |
| Final Policy Std | 0.245 |
| Final Learning Rate | 1.66e-08 |

The training converged stably with low PPO KL divergence and consistent trajectory tracking performance.
