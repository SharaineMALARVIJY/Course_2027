# CoVAPSy — Lancer le RL (Webots + ROS 2 Jazzy)

Guide pour **Zsh**, depuis le dépôt `~/Course_2027`. La simulation et les dépendances sont supposées déjà installées.

## 1. Terminal 1 — Démarrer Webots

```zsh
cd ~/Course_2027
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
ros2 launch webots_sim webots_sim.launch.py
```

Laisser Webots **en cours d'exécution** pendant l'entraînement et l'inférence. Ne pas lancer en parallèle la téléopération ou Follow the Gap sur la voiture bleue.

## 2. Terminal 2 — Préparer Python

```zsh
cd ~/Course_2027
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
source .venv-rl/bin/activate
```

### Vérification rapide (facultative)

```zsh
python rl/train.py --check
```

### Nouvel entraînement PPO

```zsh
python rl/train.py --steps 10000
```

Le modèle est sauvegardé dans `rl/models/ppo_blue.zip` **à la fin** de l'entraînement (ou lors d'un `Ctrl+C` géré par le script). Un nouveau lancement sans `--resume` repart de zéro et peut écraser ce modèle.

### Reprendre un entraînement existant

```zsh
python rl/train.py --resume --steps 100000
```

N'utiliser `--resume` qu'après la création de `rl/models/ppo_blue.zip`.

## 3. Inférence (démonstration)

Une fois l'entraînement terminé, **sans fermer Webots**, lancer dans le terminal Python :

```zsh
python rl/train.py --demo
```

Cette commande charge `rl/models/ppo_blue.zip` et pilote la voiture bleue avec la politique enregistrée. Arrêter la démo avec `Ctrl+C`.

## 4. Terminal 3 — Suivi de l'apprentissage (facultatif)

```zsh
cd ~/Course_2027
source .venv-rl/bin/activate
tensorboard --logdir rl/tb
```

Ouvrir `http://localhost:6006` pour visualiser les courbes.

## Résumé

1. **Webots** : `ros2 launch webots_sim webots_sim.launch.py`
2. **Apprentissage** : `python rl/train.py --steps 10000`
3. **Reprise** : `python rl/train.py --resume --steps 100000`
4. **Inférence** : `python rl/train.py --demo`

> Le premier entraînement peut fonctionner sans produire une bonne conduite : augmenter progressivement le nombre de pas et suivre les récompenses. Le modèle actuel est un **prototype de simulation**, pas un contrôleur validé pour la voiture réelle.
