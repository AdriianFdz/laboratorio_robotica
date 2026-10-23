#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState


class SuscriptorArticulaciones(Node):
    """Realiza la suscripción al topic /joint_states y muestra por pantalla el nombre de la articulación, posición,
    velocidad y esfuerzo de cada articulación del robot."""
    def __init__(self):
        """Constructor de la clase SuscriptorArticulaciones."""
        super().__init__("suscriptor_articulaciones")
        self.create_subscription(JointState, "/joint_states", self._callback, 10)
        self.articulaciones: dict[str, dict[str, float]] = {}

    def _callback(self, mensaje: JointState) -> None:
        """Guarda los datos recibidos de cada articulación.
        
        Args:
            mensaje (JointState): Mensaje recibido del topic /joint_states.
        """
        if "finger_joint" in mensaje.name:
            return

        for i, nombre in enumerate(mensaje.name):
            datos = {}
            datos["posicion"] = mensaje.position[i]
            datos["velocidad"] = mensaje.velocity[i]
            datos["esfuerzo"] = mensaje.effort[i]
            self.articulaciones[nombre] = datos
            print(f"Articulación: {nombre}")
            print(f"Datos: {datos}")
            print("--------------------------------")
    def obtener_datos(self) -> dict[str, dict[str, float]]:
        """Obtiene los datos de las articulaciones del robot.
        
        Returns:
            dict[str, dict[str, float]]: Diccionario con los datos de las articulaciones del robot.
        """
        return self.articulaciones

def main() -> None:
    """Función principal del nodo SuscriptorArticulaciones."""
    rclpy.init()
    nodo = SuscriptorArticulaciones()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    nodo.destroy_node()
    rclpy.try_shutdown()   
if __name__ == '__main__':
    main()