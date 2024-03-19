import ctypes
import rclpy
from rclpy.timer import Timer
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import Float32, String  # Import String for steering instructions

from camera_controller.yolop_inference3 import YolopTRT

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_error, dt):
        """Compute the PID control signal."""
        self.integral += current_error * dt
        derivative = (current_error - self.previous_error) / dt
        output = self.kp * current_error + self.ki * self.integral + self.kd * derivative
        self.previous_error = current_error
        return output


class YolopInferenceNode(Node):
    def __init__(self, categories):
        super().__init__('yolop_inference_node')
        self.categories = categories
        # Path to your plugin and model files
        self.plugin_file = "/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b2/ros2_camera/src/camera_controller/camera_controller/libmyplugins.so"
        ctypes.CDLL(self.plugin_file)
        self.yolop = YolopTRT("/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b2/ros2_camera/src/camera_controller/camera_controller/yolop.trt", self.categories)
        self.bridge = CvBridge()
        self.delayed_steering_angle = None
        self.delayed_steering_instruction = None
        self.delayed_raw_deviation = None

        # ROS publishers
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image', 10)
        self.steering_angle_publisher = self.create_publisher(Float32, 'steering_angle', 10)
        self.steering_instruction_publisher = self.create_publisher(String, 'steering_instruction', 10)
        self.raw_deviation_publisher = self.create_publisher(Float32, 'raw_deviation', 10)
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.pid_controller = PIDController(kp=0.1, ki=0.01, kd=0.005)
        self.timer = self.create_timer(1.0, self.publish_delayed_data)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            processed_images, steering_angles = self.yolop.infer([cv_image])

            for processed_image in processed_images:
                if processed_image is not None:
                    processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
                    self.processed_image_publisher.publish(processed_image_msg)

            if steering_angles:
                current_deviation = steering_angles[0]
                dt = 1.0 / 30  # Example time step

                raw_deviation_msg = Float32()
                raw_deviation_msg.data = current_deviation
                self.raw_deviation_publisher.publish(raw_deviation_msg)

                steering_adjustment = self.pid_controller.compute(current_deviation, dt)
                adjustment_msg = Float32()
                adjustment_msg.data = steering_adjustment
                self.steering_angle_publisher.publish(adjustment_msg)

                # Adjust the logic based on the angle range for going straight
                # Logic to determine steering instruction
                if -70 <= current_deviation <= -50:
                    instruction = "Straight"
                elif current_deviation < -70:
                    instruction = "Right"
                else:
                    instruction = "Left"

                # Store the values for delayed publication
                self.delayed_steering_angle = steering_adjustment
                self.delayed_steering_instruction = instruction
                self.delayed_raw_deviation = current_deviation
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"YOLOP Inference Error: {e}")

    def publish_delayed_data(self):
        """Publishes the delayed data."""
        if self.delayed_steering_angle is not None:
            adjustment_msg = Float32()
            adjustment_msg.data = self.delayed_steering_angle
            self.steering_angle_publisher.publish(adjustment_msg)
            self.delayed_steering_angle = None  # Reset after publishing

        if self.delayed_steering_instruction is not None:
            instruction_msg = String()
            instruction_msg.data = self.delayed_steering_instruction
            self.steering_instruction_publisher.publish(instruction_msg)
            self.delayed_steering_instruction = None  # Reset after publishing

        if self.delayed_raw_deviation is not None:
            raw_deviation_msg = Float32()
            raw_deviation_msg.data = self.delayed_raw_deviation
            self.raw_deviation_publisher.publish(raw_deviation_msg)
            self.delayed_raw_deviation = None  # Reset after publishing

def main(args=None):
    rclpy.init(args=args)
    categories = ["car"]
    yolop_node = YolopInferenceNode(categories)
    rclpy.spin(yolop_node)
    yolop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

