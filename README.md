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

## Domain Randomization

During training, the trajectory and uncertainty parameters were randomized as follows:

| Parameter | Range |
|---|---:|
| Centre X | 1.3 ± 0.05 m |
| Centre Y | 0.75 ± 0.10 m |
| Centre Z | 0.6 ± 0.03 m |
| Radius | 0.10–0.20 m |
| Time step / speed factor | 0.01–0.035 |
| Observation noise std | 0.00–0.01 m |
| Action noise std | 0.00–0.03 |


## Evaluation Results

The trained PPO policy was evaluated across multiple trajectories and robustness scenarios, including shifted trajectory centres, larger trajectory radii, faster target motion, and observation/action noise disturbances.

Tracking performance was evaluated using:

- Mean tracking error
- Maximum tracking error
- Final tracking error
- Mean reward
- Final reward

| Scenario | Mean Error (m) | Max Error (m) | Final Error (m) | Mean Reward | Final Reward |
|---|---:|---:|---:|---:|---:|
| Nominal | 0.00400 | 0.10388 | 0.00234 | -0.01566 | -0.01264 |
| Shifted Centre | 0.00428 | 0.14183 | 0.00175 | -0.01627 | -0.01201 |
| Larger Radius | 0.00496 | 0.14059 | 0.00278 | -0.01724 | -0.01319 |
| Faster Target | 0.00307 | 0.10389 | 0.00341 | -0.01464 | -0.01393 |
| Observation Noise | 0.00660 | 0.10398 | 0.00317 | -0.02982 | -0.02071 |
| Action Noise | 0.00505 | 0.10318 | 0.00367 | -0.02470 | -0.01911 |
| Observation + Action Noise | 0.00727 | 0.10327 | 0.00526 | -0.03299 | -0.02472 |

The results demonstrate stable trajectory tracking and robustness under multiple uncertainty conditions while maintaining low tracking error, smooth end-effector motion, and stable control behaviour under noisy observations and actuator disturbances.


## Tracking Error Across Scenarios

The figure below compares end-effector tracking error across multiple evaluation scenarios, including trajectory perturbations and observation/action noise conditions.

The PPO policy maintains low steady-state tracking error and stable convergence under all tested disturbances.

<img width="1920" height="1440" alt="all_scenarios_tracking_error" src="https://github.com/user-attachments/assets/369c4fc3-8a15-43b7-ba6a-764b5ae10251" />

