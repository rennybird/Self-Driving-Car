import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import String, Float32, Float64

class DesiredSpeedPublisher(Node):
    def __init__(self):
        super().__init__('desired_speed_publisher')
        self.publisher_ = self.create_publisher(Float64, 'desired_speed_controller', 10)
        self.timer = self.create_timer(0.5, self.publish_desired_speed)
        self.desired_speed_constant = 5.0

    def publish_desired_speed(self):
        msg = Float64()
        msg.data = self.desired_speed_constant
        self.publisher_.publish(msg)      

class SteeringAngleSubscriber(Node):
    def __init__(self):
        super().__init__('steering_angle_subscriber')
        self.subscription = self.create_subscription(Float32, 'steering_angle_want', self.steering_angle_callback, 10)
        self.subscription 

    def steering_angle_callback(self, msg):
        steering_angle = msg.data
        self.get_logger().info(f'Received Steering Angle: {steering_angle} deg')
    
class ActualSpeedSubscriber(Node):
    def __init__(self):
        super().__init__('actual_speed_subscriber')
        self.subscription = self.create_subscription(Float64, 'actual_speed_controller', self.actual_speed_callback, 10)

    def actual_speed_callback(self, msg):
        actual_speed = msg.data
        self.get_logger().info(f'Received Actual Speed: {actual_speed} km/hr')

def main(args=None):
    rclpy.init(args=args)
    desired_speed_publisher = DesiredSpeedPublisher()
    steering_angle_subscriber = SteeringAngleSubscriber()
    actual_speed_subscriber = ActualSpeedSubscriber()

    try:
        while rclpy.ok():
            rclpy.spin_once(desired_speed_publisher)
            rclpy.spin_once(steering_angle_subscriber)
            rclpy.spin_once(actual_speed_subscriber)
    except KeyboardInterrupt:
        pass

    desired_speed_publisher.destroy_node()
    steering_angle_subscriber.destroy_node()
    actual_speed_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
