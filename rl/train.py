"""Run inside ~/Course_2027/rl with ROS 2 Jazzy and .venv-rl activated."""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from webots_rl_env import BlueCarEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    (project_dir / 'models').mkdir(parents=True, exist_ok=True)
    (project_dir / 'tb').mkdir(parents=True, exist_ok=True)
    model_path = project_dir / 'models' / 'ppo_blue.zip'
    env = BlueCarEnv(max_episode_steps=400)
    try:
        print('Verification de l environnement Gymnasium...')
        check_env(env, warn=True)
        print('check_env OK')
        if args.check:
            obs, info = env.reset()
            print('min avant:', info['front_min_m'], 'm ; shape:', obs['current_lidar'].shape)
            for _ in range(20):
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                print('reward:', round(reward, 3), info)
                if terminated or truncated:
                    break
            return
        monitored_env = Monitor(env)
        if args.demo or args.resume:
            if not model_path.exists():
                raise FileNotFoundError(f'{model_path} absent. Entrainer un modele avant.')
            model = PPO.load(str(model_path), env=monitored_env, device='cpu')
        else:
            model = PPO(
                'MultiInputPolicy', monitored_env,
                learning_rate=5e-4, verbose=1, device='cpu',
                n_steps=512, batch_size=64,
                tensorboard_log=str(project_dir / 'tb'), seed=42,
            )
        if args.demo:
            obs, _ = monitored_env.reset()
            for i in range(3000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = monitored_env.step(action)
                if i % 20 == 0:
                    print(i, 'reward', round(reward, 2), 'front', round(info['front_min_m'], 3))
                if terminated or truncated:
                    print('Episode fini au pas', i, 'crash:', terminated)
                    obs, _ = monitored_env.reset()
        else:
            try:
                model.learn(
                    total_timesteps=args.steps,
                    reset_num_timesteps=not args.resume,
                    tb_log_name='PPO_blue',
                )
            except KeyboardInterrupt:
                # A short prototype can be interrupted before the first update.
                # Still save a checkpoint; not necessarily trained yet.
                model.save(str(model_path))
                print('Interrompu. Checkpoint sauvegarde:', model_path)
                return
            model.save(str(model_path))
            print('Modele sauvegarde:', model_path)
    finally:
        env.close()


if __name__ == '__main__':
    main()
