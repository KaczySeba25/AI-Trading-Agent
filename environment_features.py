import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FeatureEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, feature_extractor, initial_balance=100):
        super().__init__()

        self.fe = feature_extractor
        self.initial_balance = initial_balance

        self.obs_size = 88 + 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0
        self.last_features = None

        return self._get_state(), {}

    def _get_state(self):
        features = self.last_features
        if features is None:
            features = np.zeros(88, dtype=np.float32)

        return np.concatenate([
            features,
            np.array([self.balance, self.position], dtype=np.float32)
        ])

    def step(self, action):
        price = float(self.last_features[0])

        if action == 1:  # BUY
            self.position += 1
            self.balance -= price

        elif action == 2:  # SELL
            if self.position > 0:
                self.position -= 1
                self.balance += price

        equity = self.balance + self.position * price
        reward = equity - self.initial_balance

        return self._get_state(), reward, False, False, {}
