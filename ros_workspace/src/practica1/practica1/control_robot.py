import rclpy
from math import pi, sin, cos
from typing import List
import time
import threading

from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    RobotState
)
from moveit_msgs.srv import GetCartesianPath

from sensor_msgs.msg import JointState

from geometry_msgs.msg import Pose

from tf2_ros import Buffer, TransformListener

from suscriptor_articulaciones import SuscriptorArticulaciones

FACTOR_ESCALA = 0.1
ARTICULACIONES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
class ControlRobot(Node):
    """Clase que permite controlar el robot."""
    def __init__(self, suscriptor_articulaciones: SuscriptorArticulaciones):
        """Constructor de la clase ControlRobot.
        
        Args:
            suscriptor_articulaciones (SuscriptorArticulaciones): Instancia del suscriptor de articulaciones.
        """
        super().__init__("control_robot")
        self.suscriptor_articulaciones = suscriptor_articulaciones

        self.articulaciones = ARTICULACIONES.copy()
        self.group_name = "robot"

        # Cliente de acción para mover el robot a una posición de articulaciones       
        self.cliente_move_group = ActionClient(
            self,
            MoveGroup,
            "move_action",
        )

        # Cliente de servicio para calcular la trayectoria cartesiana
        self.cliente_calculo_trayectoria = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        # Cliente de acción para ejecutar la trayectoria calculada
        self.cliente_ejecucion_trayectoria = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        
    @staticmethod
    def __esperar(future: rclpy.task.Future) -> object:
        """Espera a que termine una operación asíncrona de ROS."""
        while not future.done():
            time.sleep(0.01)
        return future.result()

    def mover_articulaciones(self, joint_goal: List[float], wait: bool = True) -> bool:
        """Mueve el robot a las articulaciones especificadas."""
        restricciones = Constraints()
        for nombre, valor in zip(self.articulaciones, joint_goal):
            restricciones.joint_constraints.append(JointConstraint(
                joint_name=nombre, position=float(valor),
                tolerance_above=0.001, tolerance_below=0.001, weight=1.0))
        return self.__ir_a(restricciones, wait)

    def __ir_a(self, restricciones: Constraints, wait: bool) -> bool:
        """Mueve el robot a las restricciones especificadas."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = FACTOR_ESCALA
        goal.request.max_acceleration_scaling_factor = FACTOR_ESCALA
        goal.request.goal_constraints.append(restricciones)
        goal.request.start_state.is_diff = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options.plan_only = False

        goal_handle = self.__esperar(self.cliente_move_group.send_goal_async(goal))
        if not goal_handle.accepted:
            return False
        if not wait:
            return True
        resultado = self.__esperar(goal_handle.get_result_async()).result
        return resultado.error_code.val == MoveItErrorCodes.SUCCESS

    def pose_actual(self) -> Pose:
        """Devuelve la pose actual del extremo del robot en el marco world."""
        transformacion = self.tf_buffer.lookup_transform(
            "world",
            "wrist_3_link",
            Time(),
            timeout=Duration(seconds=2.0),
        )

        pose = Pose()

        pose.position.x = transformacion.transform.translation.x
        pose.position.y = transformacion.transform.translation.y
        pose.position.z = transformacion.transform.translation.z

        pose.orientation.x = transformacion.transform.rotation.x
        pose.orientation.y = transformacion.transform.rotation.y
        pose.orientation.z = transformacion.transform.rotation.z
        pose.orientation.w = transformacion.transform.rotation.w

        return pose

    def generar_circulo(self, radio: float = 0.02) -> List[Pose]:
        """Genera un círculo en el plano XY desde la pose actual."""
        pose_inicial = self.pose_actual()
        numero_puntos = 40
        poses = []

        for i in range(1, numero_puntos + 1):
            angulo = 2 * pi * i / numero_puntos

            pose = Pose()
            pose.position.x = (
                pose_inicial.position.x + radio * (cos(angulo) - 1)
            )
            pose.position.y = (
                pose_inicial.position.y + radio * sin(angulo)
            )
            pose.position.z = pose_inicial.position.z
            pose.orientation = pose_inicial.orientation

            poses.append(pose)

        return poses

    def mover_circular(self, radio: float = 0.02) -> bool:
        """Calcula y ejecuta una trayectoria circular."""
        solicitud = GetCartesianPath.Request()
        solicitud.header.frame_id = "world"
        solicitud.start_state.is_diff = True
        solicitud.group_name = self.group_name
        solicitud.link_name = "wrist_3_link"
        solicitud.max_step = 0.005
        solicitud.avoid_collisions = True
        solicitud.waypoints = self.generar_circulo(radio)

        self.cliente_calculo_trayectoria.wait_for_service()
        respuesta = self.__esperar(
            self.cliente_calculo_trayectoria.call_async(solicitud)
        )

        if (
            respuesta.error_code.val != MoveItErrorCodes.SUCCESS
            or respuesta.fraction < 0.999
        ):
            self.get_logger().error(
                f"No se pudo completar el círculo: {respuesta.fraction:.1%}"
            )
            return False

        objetivo = ExecuteTrajectory.Goal()
        objetivo.trajectory = respuesta.solution

        self.cliente_ejecucion_trayectoria.wait_for_server()
        manejador = self.__esperar(
            self.cliente_ejecucion_trayectoria.send_goal_async(objetivo)
        )

        if not manejador.accepted:
            self.get_logger().error("MoveIt ha rechazado el círculo.")
            return False

        resultado = self.__esperar(manejador.get_result_async()).result
        return resultado.error_code.val == MoveItErrorCodes.SUCCESS
    
    def mover_lineal(self, desplazamiento_x: float) -> bool:
        """Mueve linealmente el extremo del robot en el eje X."""
        pose_inicial = self.pose_actual()

        pose_final = Pose()
        pose_final.position.x = pose_inicial.position.x + desplazamiento_x
        pose_final.position.y = pose_inicial.position.y
        pose_final.position.z = pose_inicial.position.z
        pose_final.orientation = pose_inicial.orientation

        solicitud = GetCartesianPath.Request()
        solicitud.header.frame_id = "world"
        solicitud.group_name = self.group_name
        solicitud.link_name = "wrist_3_link"
        solicitud.max_step = 0.01
        solicitud.jump_threshold = 0.0
        solicitud.avoid_collisions = True

        solicitud.waypoints = [pose_final]

        futuro = self.cliente_calculo_trayectoria.call_async(solicitud)
        respuesta = self.__esperar(futuro)

        if respuesta.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error("No se pudo calcular la trayectoria lineal.")
            return False

        if respuesta.fraction < 0.99:
            self.get_logger().error(
                f"Trayectoria incompleta: {respuesta.fraction:.1%}"
            )
            return False

        self.get_logger().info("Trayectoria lineal calculada correctamente.")

        objetivo = ExecuteTrajectory.Goal()
        objetivo.trajectory = respuesta.solution

        objetivo_ejecucion = self.__esperar(
            self.cliente_ejecucion_trayectoria.send_goal_async(objetivo)
        )

        if not objetivo_ejecucion.accepted:
            self.get_logger().error("MoveIt ha rechazado la trayectoria.")
            return False

        resultado_ejecucion = self.__esperar(
            objetivo_ejecucion.get_result_async()
        )

        if resultado_ejecucion.result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error("La ejecución de la trayectoria ha fallado.")
            return False

        self.get_logger().info("Trayectoria lineal ejecutada correctamente.")
        return True

def main() -> None:
    """Función principal del nodo ControlRobot."""
    rclpy.init()
    suscriptor = SuscriptorArticulaciones()
    control = ControlRobot(suscriptor)

    ejecutor = MultiThreadedExecutor()
    ejecutor.add_node(suscriptor)
    ejecutor.add_node(control)

    hilo = threading.Thread(target=ejecutor.spin, daemon=True)
    hilo.start()

    try:
        control.get_logger().info("Esperando a que se reciban datos de las articulaciones...")
        control.cliente_move_group.wait_for_server()
        control.get_logger().info("Conexión con el servidor de MoveIt establecida.")

        suscriptor.trayectoria_actual = "joint"
        
        # Movimiento Joint
        resultado_joint = control.mover_articulaciones([
            0.5,
            -pi / 2,
            -pi / 2,
            -pi / 2,
            pi / 2,
            0.0
        ])
        print(f"Resultado del movimiento a articulaciones: {resultado_joint}")

        suscriptor.trayectoria_actual = "lineal"

        # Movimiento lineal
        resultado_lineal = control.mover_lineal(0.1)

        print(f"Resultado del movimiento lineal: {resultado_lineal}")

        suscriptor.trayectoria_actual = "circular"

        # Movimiento circular de 2 cm de radio
        resultado_circular = control.mover_circular(0.02)
        print(f"Resultado del movimiento circular: {resultado_circular}")

        suscriptor.trayectoria_actual = "sin_trayectoria"

        time.sleep(1)
        print(suscriptor.obtener_datos())

        print(f"Muestras guardadas: {len(suscriptor.historial)}")
        print("Primeras dos muestras con aceleración calculada:")
        mostradas = 0

        for muestra in suscriptor.historial:
            if muestra["aceleracion"] is not None:
                print(muestra)
                mostradas += 1

                if mostradas == 2:
                    break

        velocidad_maxima = max(
            (abs(m["velocidad"]) for m in suscriptor.historial),
            default=0.0,
        )
        print(f"Máxima velocidad registrada: {velocidad_maxima} rad/s")

        for trayectoria in ("joint", "lineal", "circular"):
            cantidad = sum(
                1 for muestra in suscriptor.historial
                if muestra["trayectoria"] == trayectoria
            )
            print(f"Muestras de {trayectoria}: {cantidad}")

        ruta_csv = "/home/laboratorio/ros_workspace/resultados/practica1.csv"
        suscriptor.guardar_csv(ruta_csv)
        print(f"Datos guardados en: {ruta_csv}")

    except KeyboardInterrupt:
        control.get_logger().info("Nodo ControlRobot detenido por el usuario.")
    finally:
        ejecutor.shutdown()
        suscriptor.destroy_node()
        control.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()