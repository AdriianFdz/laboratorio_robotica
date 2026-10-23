import rclpy
from math import pi
from typing import List
import time
import threading

from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
)

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
        self.move_group_client = ActionClient(self, MoveGroup, "move_action")

        
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

        goal_handle = self.__esperar(self.move_group_client.send_goal_async(goal))
        if not goal_handle.accepted:
            return False
        if not wait:
            return True
        resultado = self.__esperar(goal_handle.get_result_async()).result
        return resultado.error_code.val == MoveItErrorCodes.SUCCESS

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
        control.move_group_client.wait_for_server()
        control.get_logger().info("Conexión con el servidor de MoveIt establecida.")
        
        resultado = control.mover_articulaciones([
            0.3,
            -pi / 2,
            -pi / 2,
            -pi / 2,
            pi / 2,
            0.0
        ])

        time.sleep(1)
        print(suscriptor.obtener_datos())

    except KeyboardInterrupt:
        control.get_logger().info("Nodo ControlRobot detenido por el usuario.")
    finally:
        ejecutor.shutdown()
        suscriptor.destroy_node()
        control.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()