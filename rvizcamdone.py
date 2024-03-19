import ctypes
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import Float32, String  # Import String for steering instructions

from camera_controller.yolop_inference2 import YolopTRT

class YolopInferenceNode(Node):
    def __init__(self, categories):
        super().__init__('yolop_inference_node')
        self.categories = categories

        self.plugin_file = "/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b2/ros2_camera/src/camera_controller/camera_controller/libmyplugins.so"
        ctypes.CDLL(self.plugin_file)

        self.yolop = YolopTRT("/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b2/ros2_camera/src/camera_controller/camera_controller/yolop.trt", self.categories)
        self.bridge = CvBridge()

        self.processed_image_publisher = self.create_publisher(Image, 'processed_image', 10)
        self.deviation_publisher = self.create_publisher(Float32, 'lane_deviation', 10)
        self.steering_instruction_publisher = self.create_publisher(String, 'steering_instruction', 10)  # Steering instruction publisher

        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
    
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
                deviation_msg = Float32()
                deviation_msg.data = deviation
                self.deviation_publisher.publish(deviation_msg)
                self.process_steering_instruction(deviation)  # Process steering based on deviation
            else:
                self.get_logger().info("Lane not detected confidently.")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"YOLOP Inference Error: {e}")

    def process_steering_instruction(self, deviation):
        """
        Decide on steering instruction based on deviation.
        """
        if deviation < -200:  # Adjust thresholds as needed
            instruction = "Steer Right"
        elif deviation > -160:  # Adjust thresholds as needed
            instruction = "Steer Left"
        else:
            instruction = "Straight"
        
        instruction_msg = String()
        instruction_msg.data = instruction
        self.steering_instruction_publisher.publish(instruction_msg)
        self.get_logger().info(f"Steering Instruction: {instruction}")

def main(args=None):
    rclpy.init(args=args)
    categories = ["car"]
    yolop_node = YolopInferenceNode(categories)
    rclpy.spin(yolop_node)
    yolop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

