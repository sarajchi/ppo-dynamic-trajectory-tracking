import gymnasium as gym
import gymnasium_robotics
import numpy as np

from gymnasium import spaces
from stable_baselines3 import PPO


SEED = 42
MODEL_NAME = "ppo_fetchreach_tracking_policy"

np.random.seed(SEED)

gym.register_envs(gymnasium_robotics)


class MovingTargetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.step_count = 0

        self.radius = 0.15
        self.center = np.array([1.3, 0.75, 0.6])
        self.dt = 0.02

        self.obs_noise_std = 0.0
        self.action_noise_std = 0.0

        self.prev_action = None
        self.prev_ee_pos = None

        old_obs_space = env.observation_space["observation"]
        old_dim = old_obs_space.shape[0]

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

        # Domain randomisation for trajectory centre
        self.center = np.array([
            1.3 + np.random.uniform(-0.05, 0.05),
            0.75 + np.random.uniform(-0.10, 0.10),
            0.6 + np.random.uniform(-0.03, 0.03),
        ])

        # Domain randomisation for trajectory radius
        self.radius = np.random.uniform(0.10, 0.20)

        # Domain randomisation for trajectory speed
        self.dt = np.random.choice(
            [
                np.random.uniform(0.01, 0.02),
                np.random.uniform(0.02, 0.035),
            ],
            p=[0.3, 0.7],
        )

        # State and action noise for robustness
        self.obs_noise_std = np.random.uniform(0.0, 0.01)
        self.action_noise_std = np.random.uniform(0.0, 0.03)

        target, target_vel, target_acc, next_target, phase = self._trajectory(
            self.step_count
        )

        self.env.unwrapped.goal = target.copy()
        obs["desired_goal"] = target.copy()

        obs = self._augment_observation(
            obs,
            target_vel,
            target_acc,
            next_target,
            phase,
        )

        obs = self._add_observation_noise(obs)

        return obs, info

    def step(self, action):
        target, target_vel, target_acc, next_target, phase = self._trajectory(
            self.step_count
        )

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

        ee_pos = obs["achieved_goal"]
        ee_vel = (ee_pos - self.prev_ee_pos) / self.dt

        position_error = np.linalg.norm(target - ee_pos)
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
            obs,
            target_vel,
            target_acc,
            next_target,
            phase,
        )

        obs = self._add_observation_noise(obs)

        self.prev_action = noisy_action.copy()
        self.prev_ee_pos = ee_pos.copy()
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

        phase = np.array([
            np.sin(t),
            np.cos(t),
        ])

        return target, target_vel, target_acc, next_target, phase

    def _augment_observation(
        self,
        obs,
        target_vel,
        target_acc,
        next_target,
        phase,
    ):
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


def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


env = gym.make(
    "FetchReach-v4",
    reward_type="dense",
    max_episode_steps=500,
    disable_env_checker=True,
)

env = MovingTargetWrapper(env)
env.reset(seed=SEED)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=linear_schedule(1e-4),
    n_steps=2048,
    batch_size=64,
    gamma=0.98,          # Suitable horizon for dynamic tracking
    target_kl=0.02,      # PPO update stability constraint
    ent_coef=0.001,      # Mild exploration regularisation
    policy_kwargs=dict(
        net_arch=[256, 256],
    ),
    tensorboard_log="./tensorboard_logs/",
    seed=SEED,
)

model.learn(total_timesteps=5_000_000)

model.save(f"models/{MODEL_NAME}")

env.close()
