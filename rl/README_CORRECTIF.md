# Correctif LiDAR + interruption PPO (23/11/2026)

Depuis ~/Course_2027 :

```zsh
unzip -o ~/Downloads/Course_2027_RL_correctif.zip -d .
source /opt/ros/jazzy/setup.zsh
source .venv-rl/bin/activate
python rl/lidar_utils.py
python rl/scan_debug.py
python rl/train.py --check
python rl/train.py --steps 512
python rl/train.py --demo
```

Relancer Webots depuis un autre terminal, mode Play. `colcon build` n'est pas necessaire : seuls les scripts `rl/` sont modifies.

`front_min_m == 12.0` peut etre legitime si tous les obstacles sont au-dela de la portee ; `scan_debug.py` expose les vrais retours du capteur et la couverture angulaire. Un `valid_front=0` persistant et des valeurs 12.0 imposent de verifier la geometrie du LiDAR et/ou sa publication avant entrainement.

A 12 Hz en realtime, 512 pas prennent au minimum environ 43 secondes, et 10000 pas au moins 14 minutes (sans compter le calcul et les resets). Si `Ctrl+C` intervient, le script sauvegarde un checkpoint (eventuellement non entraine) dans `rl/models/ppo_blue.zip`.
