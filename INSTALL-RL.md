# Course_2027 — Installation RL / Webots

Installation manuelle pour **Kubuntu 24.04 (Python 3.12)**. Les commandes sont à copier-coller depuis la **racine du dépôt `Course_2027`**. Aucun script d'installation ni installation Python globale n'est nécessaire.

Référence : [CoVAPSy — apprentissage par renforcement et transfert simulation-réalité (2024)](https://sti.eduscol.education.fr/si-ens-paris-saclay/ressources_pedagogiques/apprentissage-renforcement-transfert-simulation-vers-realite-pourla-conduite-voitures-autonomes).

> **Compatibilité** : le tutoriel de 2024 utilise Webots R2023b. Ici nous installons **Webots R2025a**, compatible avec Ubuntu 24.04. Son code n'est pas garanti compatible sans adaptation : cette procédure installe les outils, mais ne valide pas encore le projet CoVAPSy complet. ROS 2 Jazzy reste indépendant de cet environnement virtuel.

## 1. Prérequis système

```bash
sudo apt update
sudo apt install -y git wget python3 python3-venv python3-pip
python3 --version
```

La version attendue est Python 3.12.x.

## 2. Installer Webots R2025a

```bash
wget -O /tmp/webots_2025a_amd64.deb https://github.com/cyberbotics/webots/releases/download/R2025a/webots_2025a_amd64.deb
sudo apt install -y /tmp/webots_2025a_amd64.deb
webots --version
```

On peut lancer l'interface graphique avec `webots`.

## 3. Créer l'environnement Python du projet

Dans `Course_2027/`, à côté de `requirements-rl.txt` :

```bash
python3 -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install --upgrade pip
```

À chaque nouvelle session d'entraînement, depuis la racine du dépôt :

```bash
source .venv-rl/bin/activate

# Configurer l'API Python de Webots
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH="$WEBOTS_HOME/lib/controller/python:$PYTHONPATH"
```

## 4. Installer PyTorch — choisir UNE option (pour l'instant il n'y a que le cpu d'utilisé)

**Option A — GPU NVIDIA (CUDA 12.6)** :

Vérifier d'abord la présence du pilote :

```bash
nvidia-smi
```

Puis, dans `.venv-rl` :

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

Le pilote NVIDIA doit être compatible avec le runtime CUDA 12.6. La version CUDA indiquée par `nvidia-smi` est la capacité du pilote, **pas nécessairement** la version du toolkit installé. Il n'est pas nécessaire d'installer le toolkit CUDA système pour cette roue PyTorch.

**Option B — CPU uniquement (sans NVIDIA)** :

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

## 5. Installer les dépendances RL

```bash
python -m pip install -r requirements-rl.txt
```

## 6. Vérifier l'installation

```bash
python - <<'PY'
import torch
import gymnasium
import stable_baselines3
import numpy
print('PyTorch :', torch.__version__)
print('Gymnasium :', gymnasium.__version__)
print('Stable-Baselines3 :', stable_baselines3.__version__)
print('NumPy :', numpy.__version__)
print('CUDA disponible :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU :', torch.cuda.get_device_name(0))
PY
```

Test rapide Gymnasium + PPO (sans Webots) :

```bash
python - <<'PY'
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, device='cpu', verbose=0)
model.learn(total_timesteps=1024)
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print('Test PPO OK, action :', action)
env.close()
PY
```

**Remarque** : les réseaux MLP PPO très petits sont souvent plus rapides sur CPU ; même avec CUDA disponible, ce test utilise volontairement le CPU.

## Références

- [Tutoriel CoVAPSy 2024](https://sti.eduscol.education.fr/si-ens-paris-saclay/ressources_pedagogiques/apprentissage-renforcement-transfert-simulation-vers-realite-pourla-conduite-voitures-autonomes)
- [Webots R2025a](https://github.com/cyberbotics/webots/releases/tag/R2025a)
- [PyTorch — versions précédentes / index CUDA](https://pytorch.org/get-started/previous-versions/)
