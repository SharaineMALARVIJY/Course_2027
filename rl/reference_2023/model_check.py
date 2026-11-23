"""Offline smoke test for the provided 2023 pretrained network (no ROS/Webots)."""
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from reference_logic import make_observation

path = Path(__file__).resolve().parent / 'models' / 'PPO_results_20240116.zip'
model = PPO.load(str(path), device='cpu')
obs = make_observation(np.full(201, 0.1), np.full(201, 0.1),
                       0.1, 0.0, initial=False)
assert model.observation_space.contains(obs), 'Observation space mismatch'
action, _ = model.predict(obs, deterministic=True)
assert np.asarray(action).shape == (2,), 'Incorrect pretrained action shape'
assert np.all(np.isfinite(action)), 'Nonfinite action'
print('Model 2023 loaded successfully')
print('Policy:', type(model.policy).__name__)
print('Timesteps:', model.num_timesteps)
print('Action:', action)
