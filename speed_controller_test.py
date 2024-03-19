import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ActualSpeedPublisher(Node):
    def __init__(self):
        super().__init__('actual_speed_publisher')
        self.publisher_ = self.create_publisher(Float64, 'actual_speed_controller', 10)
        self.timer = self.create_timer(0.5, self.publish_actual_speed)
        self.actual_speed_constant = 4.8
        

    def publish_actual_speed(self):
        msg = Float64()
        msg.data = self.actual_speed_constant
        self.publisher_.publish(msg)   
        self.get_logger().info(f'Actual Speed Published: {msg.data}')   

class DesiredSpeedSubscriber(Node):
    def __init__(self):
        super().__init__('desired_speed_subscriber')
        self.subscription = self.create_subscription(Float64, 'desired_speed_controller', self.desired_speed_callback, 10)

    def desired_speed_callback(self, msg):
        desired_speed = msg.data
        self.get_logger().info(f'Received Desired Speed: {desired_speed}')

def main(args=None):
    rclpy.init(args=args)
    actual_speed_publisher = ActualSpeedPublisher()
    desired_speed_subscriber = DesiredSpeedSubscriber()

    try:
        while rclpy.ok():
            rclpy.spin_once(actual_speed_publisher)
            rclpy.spin_once(desired_speed_subscriber)
    except KeyboardInterrupt:
        pass

    actual_speed_publisher.destroy_node()
    desired_speed_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()