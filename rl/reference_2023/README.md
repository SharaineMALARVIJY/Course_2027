# CoVAPSy — adaptation du RL original 2023 à Course_2027 (ROS 2 Jazzy / Webots R2025a)

Cette expérimentation **ne remplace pas** le précédent prototype `rl/train.py`.
Elle adapte le véritable `TT02_controller_RL.py` et son réseau préentraîné livrés
avec les annexes de *Bennani, Hoarau, Juton (2024)* à **la voiture bleue existante**.
La voiture jaune et la voiture violette conservent leurs contrôleurs d'origine.

> **Statut** : tests unitaires exécutés hors ROS. La simulation ROS 2 / Webots
> complète ne peut pas être lancée dans l'environnement de préparation ;
> les tests `--check` et `--demo --pretrained` sont à réaliser sur le PC équipé de Webots.

## 1. Créer une branche séparée et installer le patch

Partir de la branche contenant déjà notre premier prototype (`/rl/reset`, superviseur
RL, voiture bleue et launch ROS 2). Sauvegarder/committer les éventuelles modifications avant :

```zsh
cd ~/Course_2027
git switch feat/rl-prototype
git switch -c feat/rl-reference-2023
unzip -o ~/Downloads/Course_2027_RL_reference_patch.zip -d .
```

> Le patch modifie **uniquement** `src/webots_sim/webots_sim/rl_supervisor.py` et
> ajoute le dossier `rl/reference_2023/`. Il n'écrase ni l'ancien modèle, ni les
> fichiers `worlds/`, ni les `launch` du prototype précédent.

```zsh
source /opt/ros/jazzy/setup.zsh
source .venv-rl/bin/activate
python -m pip install -U stable-baselines3 gymnasium tensorboard
colcon build --symlink-install
source install/setup.zsh
```

À ajouter à `.gitignore` :

```gitignore
rl/reference_2023/tb/
rl/reference_2023/checkpoints/
rl/reference_2023/models/ppo_reference_2025.zip
```

La référence `rl/reference_2023/models/PPO_results_20240116.zip` est **volontairement conservée** : c'est le réseau préentraîné fourni dans l'annexe originale (environ 1 million de transitions). Ne pas confondre ce modèle avec celui que l'on entraînera ensuite.

## 2. Terminal 1 — Webots (avec superviseur RL)

```zsh
cd ~/Course_2027
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
ros2 launch webots_sim webots_sim.launch.py
```

Ne pas lancer d'autre contrôleur sur `/blue_car/cmd_ackermann` (FTG, téléop, ancien PPO). Laisser Webots en simulation **en cours** (non pausée).

## 3. Terminal 2 — Diagnostic de l'adaptation

```zsh
cd ~/Course_2027
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
source .venv-rl/bin/activate
python -m unittest discover -s rl/reference_2023 -p 'test_*.py'
python rl/reference_2023/model_check.py   # charge le reseau sans Webots
python rl/reference_2023/run.py --check
```

Observer Webots pendant le check : les commandes de la voiture bleue doivent agir sur ses roues. Si le reset échoue :

```zsh
ros2 service list | grep /rl/reset
ros2 service call /rl/reset std_srvs/srv/Trigger '{}'
ros2 service call /rl/reset_random std_srvs/srv/Trigger '{}'
ros2 topic hz /blue_car/scan
ros2 topic hz /clock
```

Les deux services doivent exister après re-build. `/rl/reset` reste déterministe, `/rl/reset_random` varie légèrement la pose sur le segment de départ de la piste 2025.

## 4. Test du modèle réellement entraîné en 2023 (sans entraînement)

```zsh
python rl/reference_2023/run.py --demo --pretrained --demo-steps 1500
```

Surveiller la trajectoire dans Webots. Le modèle original a été entraîné sur un **autre circuit** et une **autre configuration de voiture** : il est possible qu'il collisionne rapidement. Cela ne signifie pas que son chargement a échoué. Pour tester un éventuel sens de LiDAR inversé :

```zsh
python rl/reference_2023/run.py --demo --pretrained --flip-lidar
```

Pour tester le signe du braquage uniquement :

```zsh
python rl/reference_2023/run.py --demo --pretrained --flip-steering
```

Faire ces tests **séparément** pour diagnostiquer les conventions de signe ; ne pas inverser systématiquement sans observation. Stopper avec `Ctrl+C`.

## 5. Entraîner le réseau sur **notre** voiture et notre piste

Recommandation : commencer par *fine-tuner* les poids de 2023 sur le circuit 2025 :

```zsh
python rl/reference_2023/run.py --train --init-pretrained --steps 10000 --random-reset
```

Ou entraîner un PPO neuf en conservant exactement les conventions de 2023 :

```zsh
python rl/reference_2023/run.py --train --steps 10000 --random-reset
```

Le réseau adapté est enregistré dans `rl/reference_2023/models/ppo_reference_2025.zip`.
Des checkpoints sont écrits tous les 4096 pas, TensorBoard dans `rl/reference_2023/tb/`.

Poursuivre :

```zsh
python rl/reference_2023/run.py --train --resume --steps 50000 --random-reset
```

Tester le modèle adapté :

```zsh
python rl/reference_2023/run.py --demo --demo-steps 1500
```

Suivi :

```zsh
tensorboard --logdir rl/reference_2023/tb
```

## 6. Différences importantes entre la référence et le premier prototype

| Composant | Implémentation référence (annexe) | Adaptation |
|---|---|---|
| Action vitesse | `v += action[0] * 0.05` m/s | Conservée à l'identique |
| Action direction | `angle += action[1] * 9` degrés | Conservée ; saturation physique voiture 2025 à 0.28 rad |
| Vitesse max | 12 km/h = 3.333 m/s | Identique ; vitesse *commandée*, pas mesurée |
| LiDAR | 201 rayons rangés `[0..+100, -100..-1]` | Conversion fiable de ROS `LaserScan` vers **le même ordre** |
| Lidar invalide | 0 (normalisé) | Identique ; pas de remplacement par 12 m |
| Observation | Dict float64, 404 valeurs | Même espace Gym, pour charger la politique préentraînée |
| Reward | `12*(mini_normalise-.014) + 3*v_km/h` | Identique, collision si `mini<.014` ou échec acquisition |
| Épisode | 16384 actions | Identique, peut être limité avec `--episode-limit` |
| PPO | `MultiInputPolicy`, `n_steps=2048`, lr `5e-4` | Identique |
| Temps d'une action | 32 ms simulées environ dans Webots 2023b | Synchronisation à `/clock` avec `--step-ms 32` ; ce n'est pas une garantie que chaque commande a été appliquée pendant exactement 32 ms |
| Reset | 7 zones aléatoires sur **piste 2023** | Petites variations sur la zone de départ vérifiée de **piste 2025**, avec `/rl/reset_random` |

Les pistes et les limites physiques ne sont pas identiques : ce n'est **pas** une reproduction expérimentale stricte de l'article. C'est un portage des conventions d'apprentissage, destiné à mesurer la performance du modèle original puis à le réentraîner sur la simulation actuelle.

## 7. Diagnostic si la voiture conduit mal

- Le réseau 2023 est entraîné sur une autre géométrie de piste : ajuster/finetuner sur 2025 est nécessaire.
- `--flip-lidar` et `--flip-steering` permettent de tester les conventions d'orientation. Une inversion erronée rend l'inférence très mauvaise.
- Les mesures LiDAR mises à jour à ~12 Hz sont réutilisées entre les pas PPO de 32 ms, comme dans un contrôle Webots où la fréquence du LiDAR est inférieure à la fréquence du contrôleur.
- Sur la voiture 2025, l'angle de direction est limité à ±0.28 rad (~16°), contre ±0.35 rad sur le modèle 2023. On conserve le **signal normalisé d'origine** (sur 18°) pour ne pas modifier les entrées de PPO.
- Les valeurs d'odométrie du driver bleu sont estimées à partir de la consigne de vitesse : ne pas les considérer comme une vitesse physique mesurée.
- Une hausse de la récompense ne prouve pas que la voiture effectue des tours. Mesurer la distance réelle et les collisions au cours des évaluations.

## Sources

Les fichiers `upstream/` sont les **codes originaux non modifiés** issus des deux annexes fournies par l'utilisateur (article 2024, ENS Paris-Saclay). Les nouveaux fichiers `reference_logic.py`, `ros2_reference_env.py`, `run.py` et le superviseur ROS2 sont une adaptation pour `Course_2027`.
