from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import gymnasium as gym

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=256)
        self.net = th.nn.Sequential(
            th.nn.Linear(observation_space.shape[0], 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 256),
            th.nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)

class PPOAgent:
    def __init__(self, env):
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs={},
        )

        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            batch_size=64,
            n_steps=512,
        )

    def act(self, state):
        action, _ = self.model.predict(state)
        return action

    def train(self, steps=2048):
        self.model.learn(total_timesteps=steps)

    def save(self, path="models/ppo_88"):
        self.model.save(path)
