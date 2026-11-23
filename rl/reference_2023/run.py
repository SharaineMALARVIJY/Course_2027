#!/usr/bin/env python3
"""PPO train/demo/check using original 2023 policy conventions, ROS2 2025 car.

Run from repo root: python rl/reference_2023/run.py --demo --pretrained
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ros2_reference_env import ReferenceBlueCarEnv

HERE = Path(__file__).resolve().parent
TRAINED = HERE / 'models' / 'ppo_reference_2025.zip'
PRETRAINED = HERE / 'models' / 'PPO_results_20240116.zip'


def main():
    parser = argparse.ArgumentParser(description='CoVAPSy reference 2023 PPO for ROS2 blue car')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Test Gymnasium + sensors/commands')
    group.add_argument('--train', action='store_true', help='Train PPO from scratch or fine-tune')
    group.add_argument('--demo', action='store_true', help='Run deterministic inference')
    parser.add_argument('--steps', type=int, default=10000, help='Extra PPO transitions to train')
    parser.add_argument('--resume', action='store_true', help='Continue local reference model')
    parser.add_argument('--init-pretrained', action='store_true', help='Fine-tune the provided 2023 PPO')
    parser.add_argument('--pretrained', action='store_true', help='Demo with original 2023 model')
    parser.add_argument('--random-reset', action='store_true', help='Use /rl/reset_random; test before enabling')
    parser.add_argument('--step-ms', type=int, default=32, help='Simulated time per PPO action (2023: 32ms)')
    parser.add_argument('--episode-limit', type=int, default=16384)
    parser.add_argument('--demo-steps', type=int, default=1500)
    parser.add_argument('--flip-lidar', action='store_true', help='Swap +/- lidar directions for calibration')
    parser.add_argument('--flip-steering', action='store_true', help='Invert ROS steering command')
    args = parser.parse_args()

    if args.steps <= 0 or args.demo_steps <= 0 or args.episode_limit <= 0:
        parser.error('step counts and episode limit must be > 0')
    if sum((args.resume, args.init_pretrained, args.pretrained)) > 1:
        parser.error('--resume, --init-pretrained and --pretrained are mutually exclusive')
    if args.train and args.pretrained:
        parser.error('For fine-tuning use --train --init-pretrained')
    if args.demo and args.init_pretrained:
        parser.error('For inference use --demo --pretrained')

    (HERE / 'models').mkdir(parents=True, exist_ok=True)
    (HERE / 'tb').mkdir(parents=True, exist_ok=True)
    (HERE / 'checkpoints').mkdir(parents=True, exist_ok=True)
    env = ReferenceBlueCarEnv(
        step_ms=args.step_ms, max_episode_steps=args.episode_limit,
        random_resets=args.random_reset,
        flip_lidar=args.flip_lidar, flip_steering=args.flip_steering,
    )
    try:
        if args.check:
            print('[CHECK] Gymnasium environment validation...')
            check_env(env, warn=True)
            obs, info = env.reset()
            print('[CHECK] scan:', info['lidar'], 'state:', obs['current_lidar'].shape)
            for i in range(10):
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                print(f'[CHECK] step {i} reward={reward:.3f} front={info["front_min_m"]:.3f}m '
                      f'speed={info["speed_cmd_m_s"]:.2f}m/s')
                if terminated or truncated:
                    print('[CHECK] episode ended; reset')
                    env.reset()
            print('[CHECK] OK (check physical motion in Webots too)')
            return

        monitored = Monitor(env)
        if args.pretrained or args.init_pretrained:
            model_path = PRETRAINED
        else:
            model_path = TRAINED
        if args.demo or args.resume or args.init_pretrained:
            if not model_path.is_file():
                raise FileNotFoundError(f'Missing model: {model_path}')
            print(f'[MODEL] Loading: {model_path}')
            model = PPO.load(str(model_path), env=monitored, device='cpu')
            model.tensorboard_log = str(HERE / 'tb')
        else:
            model = PPO('MultiInputPolicy', monitored, learning_rate=5e-4,
                        verbose=1, device='cpu', n_steps=2048, batch_size=64,
                        tensorboard_log=str(HERE / 'tb'), seed=42)

        if args.demo:
            print('[DEMO] Inference deterministic=True. CTRL+C stops the car.')
            obs, _ = monitored.reset()
            episode_reward = 0.0
            for i in range(args.demo_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = monitored.step(action)
                episode_reward += reward
                if i % 50 == 0:
                    print(f'[DEMO] step={i} r={reward:.2f}, total={episode_reward:.1f}, '
                          f'd_front={info["front_min_m"]:.3f}m, '
                          f'v={info["speed_cmd_m_s"]:.2f}m/s, '
                          f'steer={info["steer_cmd_deg"]:.1f}deg')
                if terminated or truncated:
                    print(f'[DEMO] episode ended: collision={terminated}, '
                          f'length={info["step"]}, return={episode_reward:.1f}')
                    obs, _ = monitored.reset()
                    episode_reward = 0.0
            return

        callback = CheckpointCallback(save_freq=4096, save_path=str(HERE/'checkpoints'),
                                      name_prefix='ppo_reference', verbose=0)
        print(f'[TRAIN] {args.steps} transitions, step_ms={args.step_ms}, '
              f'reset={env.reset_service_name}, PPO n_steps={model.n_steps}')
        interrupted = False
        try:
            model.learn(total_timesteps=args.steps, callback=callback,
                        reset_num_timesteps=not (args.resume or args.init_pretrained),
                        tb_log_name='PPO_reference')
        except KeyboardInterrupt:
            print('[TRAIN] Interrupted by user: saving partial checkpoint')
            interrupted = True
        finally:
            # Never overwrite the original pre-trained checkpoint.
            model.save(str(TRAINED))
            print(f'[MODEL] Saved to {TRAINED} (interrupted={interrupted})')
    except KeyboardInterrupt:
        print('\n[STOP] Keyboard interrupt, stopping vehicle')
    finally:
        env.close()


if __name__ == '__main__':
    main()
