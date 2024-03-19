#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rclpy                        
from rclpy.node import Node         
from sensor_msgs.msg import Image   
from cv_bridge import CvBridge      
import cv2                          

class ImagePublisher(Node):

    def __init__(self, name):
        super().__init__(name)                                           
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)  
        self.timer = self.create_timer(0.1, self.timer_callback)         
        self.cap = cv2.VideoCapture(0)                                   
        self.cv_bridge = CvBridge()                                      

    def timer_callback(self):
        ret, frame = self.cap.read()  # Capture a frame from the webcam
        if ret == True:
            # Log the shape of the captured frame
            self.get_logger().info('Captured frame with shape: {}'.format(frame.shape))
            
            # Publish the captured frame as an image message
            self.publisher_.publish(
                self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8'))
        else:
            # Log a warning message if frame capture fails
            self.get_logger().warning('Failed to capture a frame from the webcam')


def main(args=None):                                 
    rclpy.init(args=args)                            
    node = ImagePublisher("topic_webcam_pub")       
    rclpy.spin(node)                                 
    node.destroy_node()                              
    rclpy.shutdown()                                 
