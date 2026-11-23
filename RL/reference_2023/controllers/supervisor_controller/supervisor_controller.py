"""Supervisor CoVAPSy 2023 : repositionnement sur commande de l'agent RL.

Conserve les zones de depart du tutoriel Eduscol et les trois compagnons.
Protocole avec le RL : RESET:<id> -> RESET_OK:<id>.
"""

import math
import random

from controller import Supervisor

NB_SPARRINGPARTNER_CARS = 0
RECEIVER_PERIOD_MS = 64
PI = math.pi

# Positions du monde train_track_2023.wbt, tirees du supervisor original.
# [[xmin,xmax], [ymin,ymax], angle_rad]
STARTING_POSITIONS = [
    [[0.00, 4.00], [5.10, 5.30], 0.00],
    [[4.95, 5.15], [0.30, 4.00], -PI / 2],
    [[2.70, 4.50], [-1.25, -0.75], PI],
    [[3.20, 3.45], [2.00, 2.40], PI / 2],
    [[1.90, 2.70], [3.10, 3.30], PI],
    [[0.00, 0.25], [-0.50, 1.50], -PI / 2],
    [[-2.20, -1.90], [-0.50, 2.30], PI / 2],
]


def random_pose(zone, reversed_direction):
    x = random.uniform(*zone[0])
    y = random.uniform(*zone[1])
    yaw = random.uniform(zone[2] - PI / 12, zone[2] + PI / 12)
    if reversed_direction:
        yaw += PI
    yaw = (yaw + PI) % (2 * PI) - PI
    return [x, y, 0.039], [0.0, 0.0, 1.0, yaw]


def lookup(supervisor, name):
    node = supervisor.getFromDef(name)
    if node is None:
        raise RuntimeError(f"Noeud DEF {name} introuvable dans le monde Webots")
    return node


def place(node, position, rotation):
    node.getField("rotation").setSFRotation(rotation)
    node.getField("translation").setSFVec3f(position)
    node.setVelocity([0.0] * 6)


def main():
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("receiver")
    receiver.enable(RECEIVER_PERIOD_MS)
    emitter = supervisor.getDevice("emitter")

    car = lookup(supervisor, "TT02_2023b_RL")
    companions = [lookup(supervisor, f"sparringpartner_car_{i}")
                  for i in range(NB_SPARRINGPARTNER_CARS)]

    print(f"[SUPERVISOR] Pret ({time_step} ms) ; "
          f"{len(companions)} compagnons")

    while supervisor.step(time_step) != -1:
        # Telemetrie facultative de position reelle : facilite le diagnostic
        # quand une voiture patine contre un mur (vitesse des roues non nulle).
        emitter.send(f"POSE:{car.getPosition()[0]:.4f}:{car.getPosition()[1]:.4f}")

        while receiver.getQueueLength() > 0:
            request = receiver.getString()
            receiver.nextPacket()
            if not request.startswith("RESET:"):
                print(f"[SUPERVISOR] Message inconnu : {request}")
                continue

            request_id = request.split(":", 1)[1]
            reverse = random.choice([False, True])
            selected = random.sample(STARTING_POSITIONS, 1 + len(companions))
            pos, rot = random_pose(selected[0], reverse)
            place(car, pos, rot)
            for node, zone in zip(companions, selected[1:]):
                pos_companion, rot_companion = random_pose(zone, reverse)
                place(node, pos_companion, rot_companion)

            # Une etape assure la prise en compte des nouvelles poses.
            if supervisor.step(time_step) == -1:
                return
            ack = f"RESET_OK:{request_id}"
            emitter.send(ack)
            print(f"[SUPERVISOR] {ack} | RL x={pos[0]:.2f} "
                  f"y={pos[1]:.2f} yaw={rot[3]:.2f}")


if __name__ == "__main__":
    main()
