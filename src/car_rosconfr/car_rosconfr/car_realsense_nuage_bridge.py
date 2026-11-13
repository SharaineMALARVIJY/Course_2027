from array import array
from copy import copy

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField


class NuageRealsenseBridge(Node):
    """
    NuageRealsenseBridge

    ====================

    Ce nœud ROS 2 sert de changeur de repère pour le nuage de points produit par la caméra RealSense

    FONCTIONNEMENT :
        1. Récupère le nuage de points publié sur /yellow_car/camera/colored_points.
        2. Vérifie le format des données reçues (coordonnées en float32, Little Endian...)
        3. Interprète les données binaires du nuage avec NumPy
        4. Transforme les coordonnées de chaque point pour passer du repère camera_link au repère camera_optical_frame
        5. Conserve les autres informations du nuage (couleurs RGB, organisation des points ...) sans les modifier
        6. Republie le nuage transformé sur /yellow_car/rgbd/points

    IMPORTANT :
        Le message PointCloud2 stocke les informations de chaque point sous forme
        d'une succession d'octets. Le programme utilise NumPy pour accéder
        directement aux coordonnées dans ces données binaires et les modifier,
        sans reconstruire entièrement le nuage de points.
    """

    def __init__(self):
        super().__init__('nuage_realsense_bridge')

        # On veut que les messages soient transmis quelque soit l'état du réseau, même si le destinataire est temporairement indisponible
        qos = QoSProfile(
            depth=5,  # Jusqu'à 5 messages peuvent être conservés en attente
            reliability=ReliabilityPolicy.RELIABLE  # Transmission fiable des messages
        )

        # PUBLISHER 
        # Publie le nuage de points transformé sur le topic de destination
        self.publisher = self.create_publisher(
            PointCloud2,
            '/yellow_car/rgbd/points',
            qos
        )

        # SUBSCRIBER 
        # Écoute le nuage de points produit par la caméra RealSense
        self.subscription = self.create_subscription(
            PointCloud2,
            '/yellow_car/camera/colored_points',
            self.callback,
            qos
        )

        self.get_logger().info('Bridge pour le nuage de points Realsense initialisé : /yellow_car/camera/colored_points -> /yellow_car/rgbd/points')

    def callback(self, msg):

        # Un message PointCloud2 contient une liste de champs :
        # x, y, z, rgb, ...
        
        # Chaque champ possède notamment :
        # - un nom 
        # - un type de donnée (FLOAT32, UINT32...) 
        # - un offset indiquant sa position en octets dans un point

        fields = {field.name: field for field in msg.fields}

        # Un float32 occupe 32 bits, soit 4 octets.
        
        # Exemple pour un point contenant x, y, z et rgb (16 octets) :
        #
        # Octets :  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
        #           -----------+------------+------------+-------------
        # Donnée :       x     |     y      |     z      |     RGB


        # x, y, z sont des FLOAT32 ? Leur offset est-il correct ? Le message est-il en Little Endian ?
        if msg.is_bigendian or any(
            name not in fields
            or fields[name].datatype != PointField.FLOAT32
            or fields[name].offset != offset
            for name, offset in [('x', 0), ('y', 4), ('z', 8)]
        ):
            self.get_logger().error('Type de message inattendu pour le nuage de points Realsense. Attendu : x, y, z en float32.')
            return

        # On utilise bytearray pour créer une COPIE MODIFIABLE de ces données car byte() est immuable, 
        # ça permet de modifier les coordonnées sans mdoifier le message original

        raw = bytearray(msg.data)

        # COmment interpréter les octets de chaque point :
        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': ['<f4', '<f4', '<f4'], # Little Endian float de 4 octets donc 32 bits
            'offsets': [0, 4, 8], # x commence à l'octet 0, y à l'octet 4, z à l'octet 8, on met pas rgb car on ne souhaite pas le modifier
            'itemsize': msg.point_step # taille totale d'un point en octets ici 16
        })

        # On crée une vue NumPy sur les données binaires du nuage de points, on ne reconstruit pas une nouvelle copie des données, c'est une sorte de pointeur quoi
        points = np.ndarray(
            shape=(msg.height, msg.width), # On conserve l'organisation de l'image de la caméra, chaque pixel correspond à un point 3D
            dtype=dtype, 
            buffer=raw, # Indique à NumPy d'utiliser directement les octets contenus dans raw sans les copier
            strides=(msg.row_step, msg.point_step) # Déplacement en octets pour passer d'une ligne à la suivante et d'un point à la suivante (ici 16 pour colonne et 10240 pour ligne si 640 points par ligne)
        )

        x = points['x'].copy()
        y = points['y'].copy()
        z = points['z'].copy()

        # Transformation du repère camera_link -> camera_optical_frame.+ : 
        #
        #       [ x' ]       [ 0  -1   0 ] [ x ]
        #       [ y' ]   =   [ 0   0  -1 ] [ y ]
        #       [ z' ]       [ 1   0   0 ] [ z ]
        #
        # Pas besoin de toucher aux couleurs RGB, la couleur de chaque point reste la même

        points['x'] = -y
        points['y'] = -z
        points['z'] = x

        # Copie du message pour conserver son formatage et ses métadonnées
        result = copy(msg)
        result.header = copy(msg.header)

        # On dit qu'on a changé de repère
        result.header.frame_id = 'camera_optical_frame'

        # On remplace les données par celles qu'on vient de transformer
        result.data = array('B', raw) # array('B', ...) convertit la liste d'octets en un tableau d'octets (type 'B' pour unsigned char)

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)

    node = NuageRealsenseBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()