import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

os.system('cls')
SEED = 42
MODEL_PATH = "models/ppo_fetchreach_tracking_policy"
VIDEO_DIR = "videos"

os.makedirs(VIDEO_DIR, exist_ok=True)

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
        ee_vel = (true_ee_pos - self.prev_ee_pos) / self.dt

        position_error = np.linalg.norm(target - true_ee_pos)
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


def record_scenario(
    scenario_id,
    name,
    center,
    radius,
    dt,
    obs_noise_std,
    action_noise_std,
    n_steps=500,
):
    scenario_name = f"{scenario_id}. {name}"
    scenario_video_dir = os.path.join(VIDEO_DIR, scenario_name)

    os.makedirs(scenario_video_dir, exist_ok=True)

    env = gym.make(
        "FetchReach-v4",
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=500,
        width=1280,
        height=720,
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

    env = RecordVideo(
        env,
        video_folder=scenario_video_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix=name,
        disable_logger=True,
    )

    obs, info = env.reset(seed=SEED)
    model = PPO.load(MODEL_PATH, env=env)

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    print(f"Saved video for: {scenario_name}")


scenarios = [
    {
        "id": 1,
        "name": "nominal",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "id": 2,
        "name": "shifted_centre",
        "center": np.array([1.35, 0.70, 0.58]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "id": 3,
        "name": "larger_radius",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.20,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "id": 4,
        "name": "faster_target",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.025,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "id": 5,
        "name": "observation_noise",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.01,
        "action_noise_std": 0.0,
    },
    {
        "id": 6,
        "name": "action_noise",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.0,
        "action_noise_std": 0.03,
    },
    {
        "id": 7,
        "name": "observation_plus_action_noise",
        "center": np.array([1.3, 0.75, 0.6]),
        "radius": 0.15,
        "dt": 0.02,
        "obs_noise_std": 0.01,
        "action_noise_std": 0.03,
    },
]


if __name__ == "__main__":
    for scenario in scenarios:
        record_scenario(
            scenario_id=scenario["id"],
            name=scenario["name"],
            center=scenario["center"],
            radius=scenario["radius"],
            dt=scenario["dt"],
            obs_noise_std=scenario["obs_noise_std"],
            action_noise_std=scenario["action_noise_std"],
            n_steps=500,
        )
