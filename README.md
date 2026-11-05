# Course 2027

### Changer les perfs

Dans le fichier worlds/Piste_CoVAPSy_2025a.wbt il faut changer 

```
WorldInfo {
  basicTimeStep 8 # Reduced from 32 ms to improve vehicle dynamics at higher speeds
  FPS 30 # Rendering at ~30 FPS may be needed on slower machines
}
```

### Activer et désactiver les voitures

Dans le fichier worlds/Piste_CoVAPSy_2025a.wbt (à la toute fin) il faut décommenté les voitures 

Et en plus pour la voiture jaune et bleu il faut décommenté `webots_bridge` et `state_publisher` dans webots_sim.launch.py

### Comment lancer les controller ros2 de la voiture bleu

Lancer la simulation (sans ou avec camera)
```bash
ros2 launch webots_sim webots_sim.launch.py
ros2 launch webots_sim webots_sim_cam.launch.py
```

Téléoperation
```bash
ros2 launch car_rosconfr teleop_ps4.launch.py

ros2 run car_rosconfr
```

Algorithme réactif:
```bash
ros2 run car_rosconfr follow_gap_blue
ros2 run car_rosconfr dial_nav_blue
ros2 run car_rosconfr simple_reactive_with_recovery_node
```


