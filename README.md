![Python](https://img.shields.io/badge/Python-3.10-blue)
![PPO](https://img.shields.io/badge/RL-PPO-green)
![MuJoCo](https://img.shields.io/badge/Simulator-MuJoCo-red)



# PPO-Based Dynamic End-Effector Trajectory Tracking in MuJoCo

This project implements a reinforcement learning framework for dynamic end-effector trajectory tracking using Proximal Policy Optimisation (PPO) in MuJoCo.

A simulated Fetch robotic manipulator is trained to follow a continuous moving Cartesian trajectory while maintaining smooth and stable motion under uncertainty.


---
## Key Features

- PPO-based continuous robotic control
- Dynamic Cartesian trajectory tracking
- Robustness under observation and action noise
- Domain randomisation during training
- Evaluation across multiple disturbance scenarios
- Automated evaluation plotting and trajectory visualisation
- MuJoCo-based robotic simulation


---
## Nominal Scenario Demonstration

The trained PPO policy tracking a continuous circular Cartesian trajectory in the nominal evaluation scenario.


https://github.com/user-attachments/assets/a17a533e-263b-4316-90f9-7a1faa7bb029

---
## Training Summary

The PPO policy was trained for approximately 5 million interaction timesteps in the MuJoCo simulation environment.

Final training statistics:

| Metric | Value |
|---|---|
| Total Timesteps | 5,001,216 |
| Mean Episode Reward | -19 |
| Success Rate | 100% |
| PPO KL Divergence | 0.000156 |
| Final Policy Std | 0.245 |
| Final Learning Rate | 1.66e-08 |


The PPO policy converged stably with low KL divergence, smooth policy updates, and consistent trajectory tracking performance.


---
## Method Overview

The proposed framework combines PPO-based continuous control with dynamic Cartesian trajectory generation in a MuJoCo robotic simulation environment.

The policy receives spatial and motion-related observations and outputs continuous control actions to minimise trajectory tracking error while maintaining smooth robotic motion.

Domain randomisation and noise perturbations were incorporated during training to improve robustness and policy generalisation.




---
## Observation Space

The observation space was designed to provide sufficient spatial and dynamic information for stable trajectory tracking, predictive motion behaviour, and robust policy learning under uncertainty.

The observation vector combines robot state information from the FetchReach environment with additional trajectory-related features.

The observation vector includes:

- End-effector position
- Target position
- Goal-relative positional information
- End-effector velocity-related information
- Target velocity
- Target acceleration
- Next target position
- Trajectory phase encoding

### Observation Formulation

```python
observation = [
    base_environment_observation,
    target_velocity,
    target_acceleration,
    next_target_position,
    trajectory_phase,
]
```

Where:

- `base_environment_observation` contains robot state information provided by the FetchReach environment
- `target_velocity` provides dynamic motion information of the moving target
- `target_acceleration` supports motion anticipation during trajectory tracking
- `next_target_position` provides short-horizon predictive information
- `trajectory_phase = [sin(t), cos(t)]` encodes periodic trajectory progression

Observation noise was additionally injected during training to improve robustness and policy generalisation under uncertain sensing conditions.

## Design Motivation

The observation design was selected to improve:

- trajectory tracking accuracy
- predictive motion behaviour
- motion smoothness
- policy stability
- robustness under uncertainty

Relative position information helps the PPO policy directly estimate tracking error and generalise across different trajectory centres and radii.

Velocity and acceleration information improve temporal consistency and reduce tracking lag during dynamic motion.

Including future target information enables the policy to anticipate target movement rather than reacting only to instantaneous position error.

Trajectory phase encoding provides periodic motion context for continuous circular trajectory tracking.

This augmented observation formulation enables the policy to learn both spatial alignment and temporal motion consistency in dynamic robotic control tasks.




---
## Action Space

The action space was designed for continuous robotic control of the end-effector motion in Cartesian trajectory tracking tasks.

### Action Formulation

```python
policy_action = PPO_policy(observation)

noisy_action = policy_action + action_noise

clipped_action = clip(
    noisy_action,
    action_lower_bound,
    action_upper_bound,
)

env.step(clipped_action)
```



The PPO policy outputs continuous control actions at every timestep to generate smooth robotic motion along the target trajectory.

Action noise was additionally injected during training to improve robustness against control disturbances and imperfect actuation conditions.

### Design Motivation

A continuous action space was selected because robotic trajectory tracking requires smooth and physically realistic motion generation.

Continuous control improves:

- motion smoothness
- tracking stability
- low-jitter behaviour
- robustness to disturbances

Discrete actions were avoided because they can produce abrupt motion transitions and unstable tracking behaviour in robotic manipulation tasks.

Action clipping was used to maintain valid actuator commands within the environment action bounds.

PPO is particularly suitable for continuous robotic control problems due to its stable policy updates and strong performance in robotic reinforcement learning applications.












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

The action magnitude penalty discourages excessively large control commands and improves control stability.

The action change penalty encourages smoother control transitions and reduces oscillatory behaviour between consecutive timesteps.




---
## Trajectory Representation

The target trajectory is represented as a continuous circular Cartesian trajectory in the XY plane with a fixed Z coordinate.

At each simulation timestep, the desired target position is generated from a parameterised circular trajectory:

```python
target = [
    center_x + radius * cos(t),
    center_y + radius * sin(t),
    center_z,
]
```




---
## Domain Randomisation and Noise Injection

During training, the trajectory and uncertainty parameters were randomized as follows:

| Parameter | Range |
|---|---:|
| Centre X | 1.3 ± 0.05 m |
| Centre Y | 0.75 ± 0.10 m |
| Centre Z | 0.6 ± 0.03 m |
| Radius | 0.10–0.20 m |
| Trajectory timestep (dt) | 0.01–0.035 s |
| Observation noise std | 0.00–0.01 observation units |
| Action noise std | 0.00–0.03 action units |

---
## Evaluation Results

The trained PPO policy was evaluated across multiple trajectories and robustness scenarios, including shifted trajectory centres, larger trajectory radii, faster target motion, and observation/action noise disturbances.

Tracking performance was evaluated using:

- Mean tracking error
- Maximum tracking error
- Final tracking error
- Mean reward
- Final reward

The trained PPO policy achieved low tracking error and stable control performance across all evaluation scenarios.

| Scenario | Mean Error (m) | Max Error (m) | Final Error (m) | Mean Reward | Final Reward |
|---|---:|---:|---:|---:|---:|
| Nominal | 0.00400 | 0.10388 | 0.00234 | -0.01566 | -0.01264 |
| Shifted Centre | 0.00428 | 0.14183 | 0.00175 | -0.01627 | -0.01201 |
| Larger Radius | 0.00496 | 0.14059 | 0.00278 | -0.01724 | -0.01319 |
| Faster Target | 0.00307 | 0.10389 | 0.00341 | -0.01464 | -0.01393 |
| Observation Noise | 0.00660 | 0.10398 | 0.00317 | -0.02982 | -0.02071 |
| Action Noise | 0.00505 | 0.10318 | 0.00367 | -0.02470 | -0.01911 |
| Observation + Action Noise | 0.00727 | 0.10327 | 0.00526 | -0.03299 | -0.02472 |

The results demonstrate stable trajectory tracking and robustness to multiple uncertainty conditions, while maintaining low tracking error, smooth end-effector trajectory-tracking behaviour, and stable control behaviour under noisy observations and actuator disturbances.

---
## Tracking Error Across Scenarios

The figure below compares end-effector tracking error across multiple evaluation scenarios, including trajectory perturbations and observation/action noise conditions.

The PPO policy maintains low steady-state tracking error and stable trajectory-tracking performance under all tested disturbances.
<img width="1920" height="1440" alt="all_scenarios_tracking_error" src="https://github.com/user-attachments/assets/369c4fc3-8a15-43b7-ba6a-764b5ae10251" />

---
## Nominal Trajectory Tracking

The trained PPO policy accurately tracks the circular target trajectory under nominal conditions with smooth and stable end-effector motion.

<img width="1920" height="1440" alt="trajectory_xy" src="https://github.com/user-attachments/assets/8880f976-ae15-4861-87e0-135274ff496c" />

---
## Robustness to Observation Noise

The policy maintains stable trajectory tracking under noisy observations, demonstrating robustness to sensor uncertainty and state perturbations.

<img width="1920" height="1440" alt="trajectory_xy" src="https://github.com/user-attachments/assets/78470d02-b5c5-4876-82d6-c602869aa4d3" />

---
## Robustness to Observation and Action Noise

The PPO policy preserves stable trajectory tracking performance under simultaneous observation and action noise disturbances, while maintaining low trajectory deviation and smooth control behaviour.

<img width="1920" height="1440" alt="trajectory_xy" src="https://github.com/user-attachments/assets/f71b13a6-0c66-4da4-a97a-3bda0cac346c" />

---
## Reward Across Evaluation Scenarios

The reward curves demonstrate stable policy behaviour and consistent control performance across all evaluation scenarios.

<img width="1920" height="1440" alt="all_scenarios_rewards" src="https://github.com/user-attachments/assets/be3a2f3d-f74d-4d5c-8583-1d1e33a4460b" />

Additional plots for all scenarios are available in the `plots/` directory.



---
## Robustness Demonstration

Evaluation of the trained PPO policy under combined observation and action noise disturbances.

The PPO policy maintains stable trajectory-tracking performance despite noisy observations and perturbed control actions, demonstrating robustness under uncertainty.

https://github.com/user-attachments/assets/e0d90623-a9cb-44a3-ab55-44f1c43affe7


Additional videos for all scenarios are available in the `videos/` directory.


---
## Robustness Analysis

The trained PPO policy maintained stable trajectory tracking performance across multiple uncertainty and disturbance scenarios, including shifted trajectory centres, larger trajectory radii, faster target motion, observation noise, action noise, and combined observation-action noise perturbations.

The evaluation results demonstrate that the policy generalised effectively beyond the nominal training trajectory while preserving smooth end-effector motion and low tracking error.

Under observation and action noise disturbances, the policy continued to follow the desired Cartesian trajectory with only minor deviations, indicating robustness to noisy sensing and perturbed control commands.

The combined observation and action noise scenario produced the highest tracking error among all evaluated conditions; however, the policy remained stable and successfully completed the trajectory without divergence or oscillatory behaviour.

These results indicate that the domain randomisation strategy and reward formulation contributed to improved robustness and stable dynamic trajectory tracking under uncertainty.

---
## Software Environment

- Python 3.10
- Stable-Baselines3 2.8.0
- PyTorch 2.11
- Gymnasium 1.2.3
- MuJoCo 3.3

---
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
├── .gitignore           # Git ignored files
├── LICENSE              # MIT license
└── README.md            # Project documentation
```


---
## Installation

Clone the repository:

```bash
git clone https://github.com/sarajchi/ppo-dynamic-trajectory-tracking.git
cd ppo-dynamic-trajectory-tracking
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:


### Windows

```bash
.venv\Scripts\activate
```


### Linux / macOS

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```




---
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
