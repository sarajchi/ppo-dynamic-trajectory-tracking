import os
import csv
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces
from stable_baselines3 import PPO

os.system('cls')
SEED = 42
MODEL_PATH = "models/ppo_fetchreach_tracking_policy"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
gym.register_envs(gymnasium_robotics)


class MovingTargetWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        center=np.array([1.3, 0.75, 0.6]),
        radius=0.15,
        dt=0.02,
        obs_noise_std=0.0,
        action_noise_std=0.0,
    ):
        super().__init__(env)

        self.step_count = 0
        self.center = center
        self.radius = radius
        self.dt = dt
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std

        self.prev_action = None
        self.prev_ee_pos = None

        old_dim = env.observation_space["observation"].shape[0]
        extra_dim = 11

        env.observation_space["observation"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(old_dim + extra_dim,),
            dtype=np.float64,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.step_count = 0
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.prev_ee_pos = obs["achieved_goal"].copy()

        target, target_vel, target_acc, next_target, phase = self._trajectory(
            self.step_count)

        self.env.unwrapped.goal = target.copy()
        obs["desired_goal"] = target.copy()

        obs = self._augment_observation(
            obs, target_vel, target_acc, next_target, phase)
        obs = self._add_observation_noise(obs)

        return obs, info

    def step(self, action):
        target, target_vel, target_acc, next_target, phase = self._trajectory(
            self.step_count)

        self.env.unwrapped.goal = target.copy()

        noisy_action = action + np.random.normal(
            0.0,
            self.action_noise_std,
            size=action.shape,
        )

        noisy_action = np.clip(
            noisy_action,
            self.env.action_space.low,
            self.env.action_space.high,
        )

        obs, reward, terminated, truncated, info = self.env.step(noisy_action)

        true_ee_pos = obs["achieved_goal"].copy()
        true_target = target.copy()

        ee_vel = (true_ee_pos - self.prev_ee_pos) / self.dt

        position_error = np.linalg.norm(true_target - true_ee_pos)
        velocity_error = np.linalg.norm(target_vel - ee_vel)

        action_magnitude = np.linalg.norm(noisy_action)
        action_change = np.linalg.norm(noisy_action - self.prev_action)

        reward = 0.0
        reward -= position_error
        reward -= 0.09 * velocity_error
        reward -= 0.01 * action_magnitude
        reward -= 0.02 * action_change

        obs["desired_goal"] = target.copy()
        obs = self._augment_observation(
            obs, target_vel, target_acc, next_target, phase)
        obs = self._add_observation_noise(obs)

        info["true_target"] = true_target
        info["true_ee_pos"] = true_ee_pos
        info["tracking_error"] = position_error

        self.prev_action = noisy_action.copy()
        self.prev_ee_pos = true_ee_pos.copy()
        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def _trajectory(self, step):
        t = step * self.dt

        target = np.array([
            self.center[0] + self.radius * np.cos(t),
            self.center[1] + self.radius * np.sin(t),
            self.center[2],
        ])

        target_vel = np.array([
            -self.radius * np.sin(t),
            self.radius * np.cos(t),
            0.0,
        ])

        target_acc = np.array([
            -self.radius * np.cos(t),
            -self.radius * np.sin(t),
            0.0,
        ])

        t_next = (step + 1) * self.dt

        next_target = np.array([
            self.center[0] + self.radius * np.cos(t_next),
            self.center[1] + self.radius * np.sin(t_next),
            self.center[2],
        ])

        phase = np.array([np.sin(t), np.cos(t)])

        return target, target_vel, target_acc, next_target, phase

    def _augment_observation(self, obs, target_vel, target_acc, next_target, phase):
        obs["observation"] = np.concatenate([
            obs["observation"],
            target_vel,
            target_acc,
            next_target,
            phase,
        ])

        return obs

    def _add_observation_noise(self, obs):
        obs["observation"] = obs["observation"].copy()
        obs["achieved_goal"] = obs["achieved_goal"].copy()
        obs["desired_goal"] = obs["desired_goal"].copy()

        obs["observation"] += np.random.normal(
            0.0,
            self.obs_noise_std,
            size=obs["observation"].shape,
        )

        obs["achieved_goal"] += np.random.normal(
            0.0,
            self.obs_noise_std,
            size=obs["achieved_goal"].shape,
        )

        return obs


def safe_filename(name):
    return name.lower().replace(" ", "_").replace("+", "plus")


def evaluate_scenario(
    name,
    center,
    radius,
    dt,
    obs_noise_std,
    action_noise_std,
    model_path,
    seed=42,
    n_steps=500,
):
    env = gym.make(
        "FetchReach-v4",
        render_mode=None,
        reward_type="dense",
        max_episode_steps=500,
        disable_env_checker=True,
    )

    env = MovingTargetWrapper(
        env,
        center=center,
        radius=radius,
        dt=dt,
        obs_noise_std=obs_noise_std,
        action_noise_std=action_noise_std,
    )

    obs, info = env.reset(seed=seed)
    model = PPO.load(model_path, env=env)

    errors = []
    rewards = []
    targets = []
    ee_positions = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        errors.append(info["tracking_error"])
        rewards.append(reward)
        targets.append(info["true_target"])
        ee_positions.append(info["true_ee_pos"])

        if terminated or truncated:
            break

    env.close()

    errors = np.array(errors)
    rewards = np.array(rewards)
    targets = np.array(targets)
    ee_positions = np.array(ee_positions)

    file_name = safe_filename(name)

    scenario_plot_dir = os.path.join(PLOTS_DIR, file_name)
    os.makedirs(scenario_plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(errors)
    plt.xlabel("Step")
    plt.ylabel("Tracking Error (m)")
    plt.title(f"Tracking Error - {name}")
    plt.grid(True)
    plt.savefig(f"{scenario_plot_dir}/tracking_error.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Reward - {name}")
    plt.grid(True)
    plt.savefig(f"{scenario_plot_dir}/reward.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(targets[:, 0], targets[:, 1], label="Target", linewidth=2)
    plt.plot(
        ee_positions[:, 0],
        ee_positions[:, 1],
        label="End-effector",
        linestyle=":",
        linewidth=3,
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"Target vs End-Effector - {name}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(f"{scenario_plot_dir}/trajectory_xy.png", dpi=300)
    plt.close()

    return {
        "scenario": name,
        "mean_error_m": np.mean(errors),
        "max_error_m": np.max(errors),
        "min_error_m": np.min(errors),
        "final_error_m": errors[-1],
        "mean_reward": np.mean(rewards),
        "final_reward": rewards[-1],
        "errors": errors,
        "rewards": rewards,
        "targets": targets,
        "ee_positions": ee_positions,
    }


scenarios = [
    {
        "name": "Nominal",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "name": "Shifted Centre",
        "center": np.array([1.35, 0.70, 0.58]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "name": "Larger Radius",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.20,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "name": "Faster Target",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.025,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "name": "Observation Noise",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.01,
        "action_noise_std": 0.0,
    },
    {
        "name": "Action Noise",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.03,
    },
    {
        "name": "Observation + Action Noise",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.01,
        "action_noise_std": 0.03,
    },
]

results = []

for scenario in scenarios:
    result = evaluate_scenario(
        name=scenario["name"],
        center=scenario["center"],
        radius=scenario["radius"],
        dt=scenario["dt"],
        obs_noise_std=scenario["obs_noise_std"],
        action_noise_std=scenario["action_noise_std"],
        model_path=MODEL_PATH,
        seed=SEED,
        n_steps=500,
    )
    results.append(result)


print("\n==============================")
print("FINAL EVALUATION TABLE")
print("==============================")
print(
    f"{'Scenario':30s} | "
    f"{'Mean Error (m)':>14s} | "
    f"{'Max Error (m)':>12s} | "
    f"{'Final Error (m)':>15s} | "
    f"{'Mean Reward':>12s} | "
    f"{'Final Reward':>13s}"
)

print("-" * 112)

for r in results:
    print(
        f"{r['scenario']:30s} | "
        f"{r['mean_error_m']:14.5f} | "
        f"{r['max_error_m']:12.5f} | "
        f"{r['final_error_m']:15.5f} | "
        f"{r['mean_reward']:12.5f} | "
        f"{r['final_reward']:13.5f}"
    )


with open(f"{RESULTS_DIR}/evaluation_summary.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Scenario",
        "Mean Error (m)",
        "Max Error (m)",
        "Min Error (m)",
        "Final Error (m)",
        "Mean Reward",
        "Final Reward",
    ])

    for r in results:
        writer.writerow([
            r["scenario"],
            r["mean_error_m"],
            r["max_error_m"],
            r["min_error_m"],
            r["final_error_m"],
            r["mean_reward"],
            r["final_reward"],
        ])


plt.figure()
for r in results:
    plt.plot(r["errors"], label=r["scenario"])

plt.xlabel("Step")
plt.ylabel("Tracking Error (m)")
plt.title("Tracking Error Across Evaluation Scenarios")
plt.legend(fontsize=7)
plt.grid(True)
plt.savefig(f"{PLOTS_DIR}/all_scenarios_tracking_error.png", dpi=300)
plt.close()


plt.figure()
for r in results:
    plt.plot(r["rewards"], label=r["scenario"])

plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Across Evaluation Scenarios")
plt.legend(fontsize=7)
plt.grid(True)
plt.savefig(f"{PLOTS_DIR}/all_scenarios_rewards.png", dpi=300)
plt.close()


# plt.figure()
# for r in results:
#     plt.plot(
#         r["ee_positions"][:, 0],
#         r["ee_positions"][:, 1],
#         label=r["scenario"],
#     )

# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")
# plt.title("End-Effector XY Trajectories Across Scenarios")
# plt.legend(fontsize=7)
# plt.grid(True)
# plt.axis("equal")
# plt.savefig(f"{PLOTS_DIR}/all_scenarios_ee_trajectories_xy.png", dpi=300)
# plt.close()


plt.figure(figsize=(10, 6))

for r in results:

    # Target trajectory (solid line)
    line = plt.plot(
        r["targets"][:, 0],
        r["targets"][:, 1],
        linewidth=2,
        label=f"{r['scenario']} Target",
    )

    colour = line[0].get_color()

    # End-effector trajectory (dotted line)
    plt.plot(
        r["ee_positions"][:, 0],
        r["ee_positions"][:, 1],
        linestyle=":",
        linewidth=2.5,
        color=colour,
        label=f"{r['scenario']} End-effector",
    )

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Target and End-Effector XY Trajectories Across Scenarios")

# Professional external legend
plt.legend(
    fontsize=6,
    ncol=1,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
)

plt.grid(True)
plt.axis("equal")

# Prevent layout cropping
plt.tight_layout()

plt.savefig(
    f"{PLOTS_DIR}/all_scenarios_target_vs_ee_trajectories_xy.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

plt.close()
