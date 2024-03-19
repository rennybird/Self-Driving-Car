import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ResultSubscriberNode(Node):
    def __init__(self):
        super().__init__('result_subscriber_node')
        self.subscription = self.create_subscription(
            Image,
            'result_image',
            self.result_callback,
            10)
        self.bridge = CvBridge()

    def result_callback(self, msg):
        print("Received a result image callback.")

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Display the result image
        cv2.imshow('Result Image', cv_image)
        cv2.waitKey(1)  # Adjust the waitKey value based on your display preferences

def main(args=None):
    rclpy.init(args=args)

    result_subscriber_node = ResultSubscriberNode()

    rclpy.spin(result_subscriber_node)

    result_subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
