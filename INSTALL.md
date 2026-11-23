# Prototype RL Webots + ROS 2 pour Course_2027 (Kubuntu 24.04 / Jazzy / zsh)

## 0. AVERTISSEMENT
Code prototype adapte statiquement a l'archive Course_2027-main.zip. Non execute ici dans Webots/ROS 2. Le reset est un superviseur distinct du Ros2Supervisor deja present. La voiture bleue doit etre seule a recevoir des commandes.

## 1. Placer le patch
Depuis ~/Course_2027, extraire l'archive Course_2027_RL_prototype.zip a la racine du depot :

    cd ~/Course_2027
    unzip -o ~/Downloads/Course_2027_RL_prototype.zip -d .

Si l'archive n'est pas dans Downloads adapter le chemin.

## 2. Installer

    source /opt/ros/jazzy/setup.zsh
    sudo apt install ros-jazzy-std-srvs python3-venv python3-colcon-common-extensions
    python3 -m venv --system-site-packages .venv-rl
    source .venv-rl/bin/activate
    python -m pip install -r rl/requirements.txt
    colcon build --symlink-install --packages-select webots_sim
    source install/setup.zsh

## 3. Démarrer
Terminal 1, Zsh:

    cd ~/Course_2027
    source /opt/ros/jazzy/setup.zsh
    source install/setup.zsh
    ros2 launch webots_sim webots_sim.launch.py

Terminal 2, Zsh:

    source /opt/ros/jazzy/setup.zsh
    source ~/Course_2027/install/setup.zsh
    ros2 service list | grep /rl/reset
    ros2 service call /rl/reset std_srvs/srv/Trigger '{}'
    ros2 topic hz /blue_car/scan

Terminal 3, Zsh:

    cd ~/Course_2027
    source /opt/ros/jazzy/setup.zsh
    source install/setup.zsh
    source .venv-rl/bin/activate
    cd rl
    python train.py --check
    python train.py --steps 10000
    python train.py --demo

Continuer l'apprentissage :

    python train.py --resume --steps 100000

Graphiques :

    tensorboard --logdir tb

## 4. Diagnostic

- Si `/rl/reset` absent, vérifier les logs `rl_supervisor`, que le `rl_supervisor_bridge` se lance et que `DEF RL_CAR` existe dans le world chargé.
- Si `/blue_car/scan` absent, vérifier que `blue_car` et `blue_webots_bridge` sont actifs.
- Si l'agent crash immédiatement, imprimer le `front_min_m` via `--check`, vérifier que les angles LaserScan sont centrés sur l'avant; ajuster `CRASH_DISTANCE` seulement après contrôle visuel.
- Si l'environnement se fige, vérifier Webots en état Run et vitesse de simulation (et le Lidar 12 Hz).
- Le modèle entraîné avec ce prototype n'est PAS prêt au transfert physique : revoir normalisation LiDAR, gains, politique, récompense, direction, vitesse, latences et sécurité.
