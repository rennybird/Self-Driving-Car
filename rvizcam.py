#Treshold -20 (ขอบซ้าย), -15 (ขอบขวา) at topic "control action"

import ctypes
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import Float32, String  # Import String for steering instructions

from camera_controller.yolop_inference2 import YolopTRT

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = 0
        self.previous_error = 0

    def compute(self, setpoint, pv, delta_time):
        error = setpoint - pv
        self.integral += error * delta_time
        derivative = (error - self.previous_error) / delta_time if delta_time > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class YolopInferenceNode(Node):
    def __init__(self, categories):
        super().__init__('yolop_inference_node')
        self.categories = categories
        
        self.plugin_file = "/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b2/ros2_camera/src/camera_controller/camera_controller/libmyplugins.so"
        ctypes.CDLL(self.plugin_file)

        self.yolop = YolopTRT("/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b2/ros2_camera/src/camera_controller/camera_controller/yolop.trt", self.categories)
        self.bridge = CvBridge()
        
        self.controller = PIDController(kp=0.1, ki=0.02, kd=0.05)
        self.setpoint = -180  # Target for lane centering
        
        # ROS publishers and subscribers
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image', 10)
        self.deviation_publisher = self.create_publisher(Float32, 'lane_deviation', 10)
        self.steering_instruction_publisher = self.create_publisher(String, 'steering_instruction', 10)
        self.control_action_publisher = self.create_publisher(Float32, 'control_action', 10)
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.last_time = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            processed_images, _, deviations = self.yolop.infer([cv_image])
            processed_image = processed_images[0] if processed_images else None
            deviation = deviations[0] if deviations else None

            if processed_image is not None:
                processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding="bgr8")
                processed_image_msg.header = msg.header
                self.processed_image_publisher.publish(processed_image_msg)

            if deviation is not None:
                current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                delta_time = current_time - self.last_time if self.last_time is not None else 0.1
                control_action = self.controller.compute(self.setpoint, deviation, delta_time)
                self.last_time = current_time

                instruction = "Straight" if abs(control_action) < 0.1 else ("Steer Left" if control_action < 0 else "Steer Right")
                instruction_msg = String()
                instruction_msg.data = instruction
                self.steering_instruction_publisher.publish(instruction_msg)
                self.get_logger().info(f"Published Steering Instruction: {instruction}")

                control_action_msg = Float32()
                control_action_msg.data = control_action
                self.control_action_publisher.publish(control_action_msg)
                self.get_logger().info(f"Published Control Action: {control_action}")

                deviation_msg = Float32()
                deviation_msg.data = deviation
                self.deviation_publisher.publish(deviation_msg)
                self.get_logger().info(f"Published Deviation: {deviation}")

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"YOLOP Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    categories = ["car"]
    yolop_node = YolopInferenceNode(categories)
    rclpy.spin(yolop_node)
    yolop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

