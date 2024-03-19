import ctypes
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from camera_controller.yolop_inference import YolopTRT

class YolopInferenceNode(Node):
    def __init__(self):
        super().__init__('yolop_inference_node')

        # Load the custom plugin library for TensorRT
        self.plugin_file = "/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b/ros2_camera/src/camera_controller/camera_controller/libmyplugins.so"
        ctypes.CDLL(self.plugin_file)

        # Initialize YOLOP with the TensorRT engine
        self.yolop = YolopTRT("/media/inc/64fdc28e-1c24-427f-bec3-0bf20738fe6b/ros2_camera/src/camera_controller/camera_controller/yolop.trt")

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10)
        self.subscription  # Prevent unused variable warning

    def image_callback(self, msg):
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

                # Log the properties of the cv_image
                self.get_logger().info(f'Image shape: {cv_image.shape}')
                self.get_logger().info(f'Image dtype: {cv_image.dtype}')

                # Perform YOLOP inference
                processed_images = self.yolop.infer(cv_image)

                # Iterate through processed images and display
                for processed_image in processed_images:
                    if processed_image is not None:
                        cv2.namedWindow('YOLOP Output', cv2.WINDOW_NORMAL)
                        cv2.imshow('YOLOP Output', processed_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Processed image is None.")
                        
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")




def main(args=None):
    rclpy.init(args=args)
    yolop_node = YolopInferenceNode()
    rclpy.spin(yolop_node)
    yolop_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()