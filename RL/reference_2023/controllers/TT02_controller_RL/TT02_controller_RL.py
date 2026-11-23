"""Entrainement PPO de la TT02 dans Webots (sans ROS 2).

Compatible avec le Supervisor du correctif reference_2023 :
    RESET:<id> -> RESET_OK:<id>, puis telemetrie POSE:x:y.
Le LiDAR fourni au reseau est le secteur brut 80..280 (201 rayons).
"""

import argparse
import math
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from vehicle import Driver


# Capteurs et actions : conserver les valeurs de la version validee.
FRONT_START = 80
FRONT_STOP = 281              # [80, 280], soit 200 degres / 201 rayons
LIDAR_MAX_M = 12.0
MIN_VALID_RAYS = 20
MAX_SPEED_M_S = 3.0 # 12.0 / 3.6
MAX_STEERING_RAD = 0.28
SPEED_STEP_M_S = 0.05
STEERING_STEP_RAD = math.radians(9.0)

# Fin d'episode et repositionnement.
COLLISION_M = 0.17
MAX_EPISODE_STEPS = 16_384
STUCK_GRACE_S = 1.5
STUCK_DURATION_S = 1.5
STUCK_COMMAND_KMH = 1.5
STUCK_SPEED_KMH = 0.2
STUCK_MIN_MOVEMENT_M = 0.05
RESET_MAX_ATTEMPTS = 5
RESET_REPLY_TIMEOUT_STEPS = 300
SENSOR_STABILIZATION_S = 0.30


class WebotsGymEnvironment(Driver, gym.Env):
    def __init__(self):
        Driver.__init__(self)
        gym.Env.__init__(self)
        self.basic_step_s = int(self.getBasicTimeStep()) / 1000.0

        self.lidar = self.getDevice("RpLidarA2")
        self.lidar.enable(int(self.getBasicTimeStep()))
        if self.lidar.getHorizontalResolution() != 360:
            raise RuntimeError("Le traitement LiDAR attend 360 mesures")

        self.receiver = self.getDevice("receiver")
        self.receiver.enable(64)
        self.emitter = self.getDevice("emitter")

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "current_lidar": gym.spaces.Box(0.0, 1.0, (201,), dtype=np.float32),
            "previous_lidar": gym.spaces.Box(0.0, 1.0, (201,), dtype=np.float32),
            "previous_speed": gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "previous_angle": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        })

        self.speed_m_s = 0.0
        self.steering_rad = 0.0
        self.previous_lidar = np.zeros(201, dtype=np.float32)
        self.episode_steps = 0
        self.episode_number = 0
        self.stuck_steps = 0
        self.reset_id = 0
        self.pose_history = deque(maxlen=128)
        self._set_motion(0.0, 0.0)

    def _tick(self):
        if Driver.step(self) == -1:
            raise RuntimeError("Webots a arrete la simulation")

    def _set_motion(self, speed_m_s, steering_rad):
        self.speed_m_s = float(np.clip(speed_m_s, 0.0, MAX_SPEED_M_S))
        self.steering_rad = float(np.clip(steering_rad, -MAX_STEERING_RAD, MAX_STEERING_RAD))
        self.setCruisingSpeed(self.speed_m_s * 3.6)  # Driver attend des km/h
        self.setSteeringAngle(self.steering_rad)     # Driver attend des radians

    def _observation(self, initial=False):
        raw = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)
        if raw.shape != (360,):
            raise RuntimeError(f"Image LiDAR inattendue : {raw.shape}")

        front = raw[FRONT_START:FRONT_STOP].copy()
        valid = np.isfinite(front) & (front > 0.0) & (front < LIDAR_MAX_M)
        front[~valid] = 0.0
        current = (front / LIDAR_MAX_M).astype(np.float32)
        nonzero = front[front > 0.0]
        nearest_m = float(nonzero.min()) if nonzero.size else None

        obs = {
            "current_lidar": current.copy(),
            "previous_lidar": current.copy() if initial else self.previous_lidar.copy(),
            "previous_speed": np.array(
                [0.0 if initial else self.speed_m_s / MAX_SPEED_M_S], dtype=np.float32
            ),
            "previous_angle": np.array(
                [0.0 if initial else self.steering_rad / MAX_STEERING_RAD], dtype=np.float32
            ),
        }
        self.previous_lidar = current.copy()
        return obs, nearest_m, int(valid.sum())

    def _consume_pose(self):
        while self.receiver.getQueueLength() > 0:
            message = self.receiver.getString()
            self.receiver.nextPacket()
            if message.startswith("POSE:"):
                try:
                    _, x, y = message.split(":", 2)
                    self.pose_history.append((self.getTime(), float(x), float(y)))
                except ValueError:
                    pass

    def _position_stuck(self):
        now = self.getTime()
        while self.pose_history and now - self.pose_history[0][0] > STUCK_DURATION_S:
            self.pose_history.popleft()
        if len(self.pose_history) < 2:
            return False
        first, last = self.pose_history[0], self.pose_history[-1]
        if last[0] - first[0] < STUCK_DURATION_S * 0.8:
            return False
        return math.hypot(last[1] - first[1], last[2] - first[2]) < STUCK_MIN_MOVEMENT_M

    def _request_reset(self):
        while self.receiver.getQueueLength() > 0:
            self.receiver.nextPacket()

        self.reset_id += 1
        request = f"RESET:{self.reset_id}"
        self.emitter.send(request)
        for _ in range(RESET_REPLY_TIMEOUT_STEPS):
            self._tick()
            while self.receiver.getQueueLength() > 0:
                reply = self.receiver.getString()
                self.receiver.nextPacket()
                if reply == f"RESET_OK:{self.reset_id}":
                    return
        raise RuntimeError(f"Aucune confirmation du Supervisor pour {request}")

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._set_motion(0.0, 0.0)
        self.episode_steps = 0
        self.stuck_steps = 0
        self.pose_history.clear()

        for _ in range(RESET_MAX_ATTEMPTS):
            self._request_reset()
            self._set_motion(0.0, 0.0)
            for _ in range(max(2, math.ceil(SENSOR_STABILIZATION_S / self.basic_step_s))):
                self._tick()
            obs, nearest_m, valid_count = self._observation(initial=True)
            if valid_count >= MIN_VALID_RAYS and nearest_m is not None and nearest_m >= COLLISION_M:
                self.episode_number += 1
                return obs, {}

        raise RuntimeError("Impossible d'obtenir une position de depart LiDAR valide apres 5 resets")

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        speed = max(0.1, self.speed_m_s + float(np.clip(action[0], -1, 1)) * SPEED_STEP_M_S)
        steering = self.steering_rad + float(np.clip(action[1], -1, 1)) * STEERING_STEP_RAD
        self._set_motion(speed, steering)
        self._tick()
        self.episode_steps += 1

        obs, nearest_m, valid_count = self._observation()
        self._consume_pose()
        measured_kmh = abs(float(self.getCurrentSpeed()))
        commanded_kmh = self.speed_m_s * 3.6

        if (self.episode_steps * self.basic_step_s >= STUCK_GRACE_S
                and commanded_kmh >= STUCK_COMMAND_KMH
                and measured_kmh <= STUCK_SPEED_KMH):
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        reason = None
        if valid_count < MIN_VALID_RAYS:
            reason = "lidar_invalide"
        elif nearest_m is not None and nearest_m < COLLISION_M:
            reason = "mur_proche"
        elif (self.episode_steps * self.basic_step_s >= STUCK_GRACE_S + STUCK_DURATION_S
              and commanded_kmh >= STUCK_COMMAND_KMH and self._position_stuck()):
            reason = "aucun_deplacement"
        elif self.stuck_steps * self.basic_step_s >= STUCK_DURATION_S:
            reason = "voiture_bloquee"
        elif measured_kmh > 30.0:
            reason = "vitesse_anormale"

        terminated = reason is not None
        truncated = not terminated and self.episode_steps >= MAX_EPISODE_STEPS
        if terminated:
            reward = -500.0
            print(f"[EPISODE {self.episode_number}] {reason} | pas={self.episode_steps}")
            self._set_motion(0.0, 0.0)
        else:
            proximity = (nearest_m if nearest_m is not None else 0.0) / LIDAR_MAX_M
            reward = float(12.0 * (proximity - COLLISION_M / LIDAR_MAX_M) + 3.0 * commanded_kmh)
            if truncated:
                self._set_motion(0.0, 0.0)

        return obs, reward, terminated, truncated, {"reason": reason}


def main():
    parser = argparse.ArgumentParser(description="Entrainement PPO - CoVAPSy Webots")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Nombre de pas d'entrainement")
    parser.add_argument("--output", default="PPO_results_20261123", help="Nom du fichier modele")
    args = parser.parse_args()

    env = WebotsGymEnvironment()
    try:
        model = PPO(
            "MultiInputPolicy", env,
            learning_rate=5e-4,
			# batch_size = 128, # pas sûr de moi
			# n_epochs = 5, # pas sûr de moi
            verbose=1,
            device="cpu",
            tensorboard_log="./PPO_Tensorboard",
        )
        model.learn(total_timesteps=args.steps)
        model.save(args.output)
        print(f"Modele enregistre : {args.output}.zip")
    finally:
        env._set_motion(0.0, 0.0)
        env.close()


if __name__ == "__main__":
    main()
