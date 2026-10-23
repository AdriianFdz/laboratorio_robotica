#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState


class Suscriptor(Node):
    def __init__(self):
        super().__init__("suscriptor_articulaciones")
        self.create_subscription(JointState, "/joint_states", self.callback, 10)

    def callback(self, mensaje: JointState):
        if "finger_joint" in mensaje.name:
            print("Articulación: ", mensaje.name[0])
            print("No tiene posición, velocidad ni esfuerzo")
            print("--------------------------------------------------")
        else:
            for i in range(len(mensaje.name)):
                print("Articulación: ", mensaje.name[i])
                print("Posición: ", mensaje.position[i])
                print("Velocidad: ", mensaje.velocity[i])
                print("Esfuerzo: ", mensaje.effort[i])
                print("--------------------------------------------------")
            
if __name__ == '__main__':
    rclpy.init()
    nodo = Suscriptor()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    nodo.destroy_node()
    rclpy.try_shutdown()
