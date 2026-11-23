# Lancer l'entraînement RL avec PPO

L'entraînement utilise **Webots R2025a**, **Gymnasium** et **Stable-Baselines3 (PPO)**. La voiture apprend à conduire à partir des mesures du LiDAR sur 200° à l'avant. Les collisions déclenchent automatiquement un nouvel épisode grâce au Supervisor.

**1. Lancer Webots en mode rapide (terminal 1)**

```bash
cd ~/Course_2027/RL/reference_2023
webots --mode=fast --no-rendering worlds/train_track_2023.wbt
```

**2. Activer l'environnement Python (terminal 2)**

```bash
cd ~/Documents/ROS/ros2_ws
source .venv-rl/bin/activate

export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH="$WEBOTS_HOME/lib/controller/python:$PYTHONPATH"

cd RL/reference_2023/controllers/TT02_controller_RL
```

**3. Démarrer l'entraînement PPO**

```bash
python TT02_controller_RL.py
```

Le paramètre `--steps` définit le nombre de transitions d'entraînement (1 millons par defaut). Le modèle est sauvegardé automatiquement à la fin.

**4. Visualiser les performances avec TensorBoard (terminal 3)**

```bash
tensorboard --logdir RL/reference_2023/controllers/TT02_controller_RL/PPO_Tensorboard
```

Ouvrir ensuite [http://localhost:6006](http://localhost:6006) pour suivre les récompenses (`rollout/ep_rew_mean`), la durée des épisodes (`rollout/ep_len_mean`) et les statistiques d'apprentissage PPO.