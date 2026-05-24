# PPO-Based Dynamic End-Effector Trajectory Tracking in MuJoCo

This project implements a reinforcement learning framework for dynamic end-effector trajectory tracking using Proximal Policy Optimisation (PPO) in MuJoCo.

A simulated Fetch robotic manipulator is trained to follow a continuous moving Cartesian trajectory while maintaining smooth and stable motion under uncertainty.


## Nominal Scenario Demonstration

The trained PPO policy tracking a continuous circular Cartesian trajectory in the nominal evaluation scenario.

https://github.com/user-attachments/assets/394f0ca8-79fa-4074-baba-08098bcf95aa


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




## Observation Space

The observation space was designed to provide sufficient spatial and dynamic information for stable trajectory tracking and robust policy learning.

The observation vector includes:

- End-effector position
- Target position
- Relative position error
- Velocity-related information

### Observation Formulation

```python
observation = [
    end_effector_position,
    target_position,
    target_position - end_effector_position,
    velocity_information,
]
```

### Design Motivation

The observation design was selected to improve:

- trajectory tracking accuracy
- motion smoothness
- policy stability
- robustness under uncertainty

Relative position information helps the PPO policy directly estimate tracking error and improves generalisation across different trajectory centres and radii.

Velocity information was included to support dynamic trajectory tracking and reduce motion lag and oscillatory behaviour.

This observation formulation enables the policy to learn both spatial alignment and temporal motion consistency.

---

## Action Space

The action space was designed for continuous robotic control of the end-effector motion.

### Action Formulation

```python
action = continuous_control_command
```

The PPO policy outputs continuous actions at every timestep to control the robotic motion smoothly along the target trajectory.

### Design Motivation

A continuous action space was selected because robotic trajectory tracking requires smooth and physically realistic motion generation.

Continuous control improves:

- motion smoothness
- tracking stability
- low-jitter behaviour
- robustness to disturbances

Discrete actions were avoided because they can produce abrupt motion transitions and unstable tracking behaviour in robotic manipulation tasks.

PPO is particularly suitable for continuous control problems due to its stable policy updates and strong performance in robotic reinforcement learning applications.

---

## Reward Function

The reward function was designed to encourage accurate and smooth trajectory tracking while maintaining stable and realistic robotic control behaviour.

### Reward Formulation

```python
reward = 0.0

reward -= position_error
reward -= 0.09 * velocity_error
reward -= 0.01 * action_magnitude
reward -= 0.02 * action_change
```

Where:

- `position_error` is the Euclidean distance between the target and end-effector positions
- `velocity_error` measures the mismatch between target and end-effector velocities
- `action_magnitude` penalises excessively large control commands
- `action_change` penalises abrupt changes between consecutive actions

### Design Motivation

The reward function was designed to balance:

- tracking accuracy
- smooth motion generation
- control stability
- robustness under noisy conditions

The position error term encourages accurate target tracking.

The velocity error term improves temporal consistency and reduces dynamic mismatch during motion.

The action magnitude penalty discourages aggressive control behaviour and improves stability.






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

## Nominal Trajectory Tracking

The trained PPO policy accurately tracks the circular target trajectory under nominal conditions with smooth and stable end-effector motion.

<img width="1920" height="1440" alt="trajectory_xy" src="https://github.com/user-attachments/assets/8880f976-ae15-4861-87e0-135274ff496c" />


## Robustness to Observation Noise

The policy maintains stable trajectory tracking under noisy observations, demonstrating robustness to sensor uncertainty and state perturbations.

<img width="1920" height="1440" alt="trajectory_xy" src="https://github.com/user-attachments/assets/78470d02-b5c5-4876-82d6-c602869aa4d3" />


## Robustness to Observation and Action Noise

The PPO controller preserves stable trajectory tracking performance under simultaneous observation and action noise disturbances, while maintaining low trajectory deviation and smooth control behaviour.

<img width="1920" height="1440" alt="trajectory_xy" src="https://github.com/user-attachments/assets/f71b13a6-0c66-4da4-a97a-3bda0cac346c" />

## Reward Across Evaluation Scenarios

The reward curves demonstrate stable policy behaviour and consistent control performance across all evaluation scenarios.

<img width="1920" height="1440" alt="all_scenarios_rewards" src="https://github.com/user-attachments/assets/be3a2f3d-f74d-4d5c-8583-1d1e33a4460b" />

Additional trajectory visualisations for all scenarios are available in the `plots/` directory.


The action change penalty reduces jitter and encourages smooth robotic motion suitable for real-world deployment.

This reward formulation enables the PPO policy to learn stable, accurate, and robust end-effector trajectory tracking under multiple uncertainty conditions.


## Robustness Demonstration

Evaluation of the trained PPO policy under combined observation and action noise disturbances.

The policy maintains stable trajectory tracking behaviour despite noisy observations and perturbed control actions, demonstrating robustness under uncertainty.


https://github.com/user-attachments/assets/1e9260f3-2663-421c-aade-fa7fbbc98f77


Additional videos for all scenarios are available in the `videos/` directory.


## Robustness Analysis

The trained PPO controller maintained stable trajectory tracking performance across multiple uncertainty and disturbance scenarios, including shifted trajectory centres, larger trajectory radii, faster target motion, observation noise, action noise, and combined observation-action noise perturbations.

The evaluation results demonstrate that the policy generalised effectively beyond the nominal training trajectory while preserving smooth end-effector motion and low tracking error.

Under observation and action noise disturbances, the controller continued to follow the desired Cartesian trajectory with only minor deviations, indicating robustness to noisy sensing and perturbed control commands.

The combined observation and action noise scenario produced the highest tracking error among all evaluated conditions; however, the policy remained stable and successfully completed the trajectory without divergence or oscillatory behaviour.

These results indicate that the domain randomisation strategy and reward formulation contributed to improved robustness and stable dynamic trajectory tracking under uncertainty.


## Environment

- Python 3.10
- Stable-Baselines3 2.8.0
- PyTorch 2.11
- Gymnasium 1.2.3
- MuJoCo 3.3


## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
ppo-dynamic-trajectory-tracking/
│
├── models/              # Trained PPO policies
├── plots/               # Evaluation plots
├── videos/              # Evaluation videos
├── train.py             # PPO training script
├── evaluation.py        # Evaluation and plotting script
├── record_video.py      # Video recording utility
├── requirements.txt     # Python dependencies
└── README.md
```


## Quick Start

Train the PPO policy:

```bash
python train.py
```

Evaluate the trained model:

```bash
python evaluation.py
```

Record evaluation videos:

```bash
python record_video.py
```
