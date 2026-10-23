#!/usr/bin/python3

import csv
import rclpy

from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from pathlib import Path


class SuscriptorArticulaciones(Node):
    """Realiza la suscripción al topic /joint_states y muestra por pantalla el nombre de la articulación, posición,
    velocidad y esfuerzo de cada articulación del robot."""
    def __init__(self):
        """Constructor de la clase SuscriptorArticulaciones."""
        super().__init__("suscriptor_articulaciones")
        self.create_subscription(JointState, "/joint_states", self._callback, 10)
        self.articulaciones: dict[str, dict[str, float]] = {}
        self.historial: list[dict] = []
        self.muestras_anteriores: dict[str, tuple[float, float]] = {}
        self.trayectoria_actual: str = "sin_trayectoria"

    def _callback(self, mensaje: JointState) -> None:
        """Guarda los datos recibidos de cada articulación.
        
        Args:
            mensaje (JointState): Mensaje recibido del topic /joint_states.
        """
        if "finger_joint" in mensaje.name:
            return

        tiempo = (
            mensaje.header.stamp.sec
            + mensaje.header.stamp.nanosec / 1_000_000_000
        )

        for i, nombre in enumerate(mensaje.name):
            datos = {}
            datos["posicion"] = mensaje.position[i]
            datos["velocidad"] = mensaje.velocity[i]
            datos["esfuerzo"] = mensaje.effort[i]
            self.articulaciones[nombre] = datos
            aceleracion = None
            velocidad_actual = datos["velocidad"]

            if nombre in self.muestras_anteriores:
                tiempo_anterior, velocidad_anterior = (
                    self.muestras_anteriores[nombre]
                )
                diferencia_tiempo = tiempo - tiempo_anterior

                if diferencia_tiempo > 0:
                    aceleracion = (
                        velocidad_actual - velocidad_anterior
                    ) / diferencia_tiempo

            self.muestras_anteriores[nombre] = (
                tiempo,
                velocidad_actual,
            )
            self.historial.append({
                "tiempo": tiempo,
                "articulacion": nombre,
                "posicion": datos["posicion"],
                "velocidad": datos["velocidad"],
                "aceleracion": aceleracion,
                "trayectoria": self.trayectoria_actual,
            })

    def guardar_csv(self, ruta: str) -> None:
        """Guarda el historial en un CSV con una fila por muestra."""
        archivo = Path(ruta)
        archivo.parent.mkdir(parents=True, exist_ok=True)

        campos = [
            "trayectoria",
            "tiempo",
            "articulacion",
            "posicion",
            "velocidad",
            "aceleracion",
        ]

        muestras = self.historial.copy()

        with archivo.open("w", newline="", encoding="utf-8") as fichero:
            escritor = csv.DictWriter(fichero, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(muestras)

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