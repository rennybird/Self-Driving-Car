import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import os
import can

class SteeringControl(Node):
    def __init__(self):
        super().__init__('steering_control')

        self.yaw_control = 0.0
        self.angle = 0.0
        self.elec_angle_dec = 0.0
        self.elec_angle_hex = 0.0
        self.DATA_Hh = 0.0
        self.DATA_Hl = 0.0
        self.DATA_Lh = 0.0
        self.DATA_Ll = 0.0
        self.msg_sent = None

        # Subscribe to the topic where the steering angle is published
        self.subscription = self.create_subscription(Float32, 'steering_angle_want', self.steering_angle_callback, 10)

        # Setup CAN interface
        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 250000')
        os.system("sudo ifconfig can0 txqueuelen 250000")
        os.system('sudo ifconfig can0 up')

        self.can0 = can.interface.Bus(channel='can0', bustype='socketcan')
        msg_init = can.Message(arbitration_id=0x06000001, data=[0x23, 0x0C, 0x20, 0x01, 0x00, 0x00, 0x00, 0x00])

        self.can0.send(msg_init)
        self.get_logger().info("Message sent on Enable state data: {}".format(msg_init))

    def steering_angle_callback(self, msg):
        if -540 <= msg.data <= 540:
            self.yaw_control = msg.data
            self.angle = self.yaw_control
            self.elec_angle_dec = self.angle * 27
            self.elec_angle_hex = ('{:0>8X}'.format(int(self.elec_angle_dec) & (2**32-1)))

            self.DATA_Hh = ((int(self.elec_angle_hex[0:2], 16)))
            self.DATA_Hl = ((int(self.elec_angle_hex[2:4], 16)))
            self.DATA_Lh = ((int(self.elec_angle_hex[4:6], 16)))
            self.DATA_Ll = ((int(self.elec_angle_hex[6:8], 16)))

            msg_sent = can.Message(arbitration_id=0x06000001, data=[0x23, 0x02, 0x20, 0x01, self.DATA_Lh, self.DATA_Ll, self.DATA_Hh, self.DATA_Hl])
            self.can0.send(msg_sent)
            self.get_logger().info("Steering Angle: {}".format(self.yaw_control))
        else:
            self.get_logger().info("Received angle is out of range: {}".format(msg.data))


def main(args=None):
    rclpy.init(args=args)
    node = SteeringControl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
