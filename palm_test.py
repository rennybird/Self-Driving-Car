import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class ProcessSteeringInstructionSubscriber(Node):
    def __init__(self, processed_steering_publisher):
        super().__init__('process_steering_instruction_subscriber')
        self.subscription = self.create_subscription(Float32, 'control_action', self.control_action_callback, 10)
        self.processed_steering_publisher = processed_steering_publisher

    def control_action_callback(self, msg):
        control_action = msg.data
        # Constants for linear transformation; adjust these based on your specific requirements
        m = 1.0  # Adjust this factor based on control action scaling
        c = 0    # This can be adjusted if there's a need for an offset
        y = round(m * control_action + c, 0)  # Linear transformation

        # Clamp the steering angle to the maximum allowable values [-180, 180]
        y = max(min(y, 180), -180)

        # Publish the steering angle
        self.processed_steering_publisher.publish_steering_angle(y)  
        self.get_logger().info(f'Received Control Action: {control_action}, Calculated Steering Angle: {y}')

class ProcessSteeringInstructionPublisher(Node):
    def __init__(self):
        super().__init__('process_steering_instruction_publisher')
        self.publisher_ = self.create_publisher(Float32, 'steering_angle_want', 10)

    def publish_steering_angle(self, y):
        msg = Float32()
        msg.data = y
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Steering Angle: {y}')

def main(args=None):
    rclpy.init(args=args)
    process_steering_instruction_publisher = ProcessSteeringInstructionPublisher()
    process_steering_instruction_subscriber = ProcessSteeringInstructionSubscriber(process_steering_instruction_publisher)
    rclpy.spin(process_steering_instruction_subscriber)
    process_steering_instruction_publisher.destroy_node()
    process_steering_instruction_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
