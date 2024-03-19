import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float64

class ProcessSteeringInstructionSubscriber(Node):
    def __init__(self, processed_steering_publisher):
        super().__init__('process_steering_instruction_subscriber')
        self.subscription = self.create_subscription(String, 'steering_instruction', self.steering_instruction_callback, 10)
        self.processed_steering_publisher = processed_steering_publisher

    def steering_instruction_callback(self, msg):
        instruction = msg.data
        if instruction == 'Straight':
            self.processed_steering_publisher.publish_steering_angle(0.0)
        elif instruction == 'Left':
            self.processed_steering_publisher.publish_steering_angle(15.0)
        elif instruction == 'Right':
            self.processed_steering_publisher.publish_steering_angle(-15.0)
        else:
            self.get_logger().warn(f'Unknown Instruction: {instruction}')

        self.get_logger().info(f'Received Steering Instruction: {instruction}')

class ProcessSteeringInstructionPublisher(Node):
    def __init__(self):
        super().__init__('process_steering_instruction_publisher')
        self.publisher_ = self.create_publisher(Float32, 'steering_angle_want', 10)

    def publish_steering_angle(self, angle):
        msg = Float32()
        msg.data = angle
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Steering Angle: {angle}')

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
