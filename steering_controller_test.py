import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class SteeringAngleSubscriber(Node):
    def __init__(self):
        super().__init__('steering_angle_subscriber')
        self.subscription = self.create_subscription(Float64, 'steering_angle_want', self.steering_angle_callback, 10)
        self.subscription  # prevent unused variable warning

    def steering_angle_callback(self, msg):
        steering_angle = msg.data
        print(f'Received steering angle: {steering_angle}')

def main(args=None):
    rclpy.init(args=args)
    steering_angle_subscriber = SteeringAngleSubscriber()
    rclpy.spin(steering_angle_subscriber)
    steering_angle_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()