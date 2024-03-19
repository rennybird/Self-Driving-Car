import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch

# Import your utilities and model functions (adjust the imports according to your file structure)
from utils import select_device, LoadImages, non_max_suppression, plot_one_box

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.model = self.load_model()
        self.device = select_device()  # Select the computation device
        self.model.to(self.device)
        self.model.eval()

    def load_model(self):
        # Adjust the path to your model
        model_path = 'yolopv2.pt'
        model = torch.jit.load(model_path)
        return model

    def listener_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Process the image
        processed_image = self.process_image(cv_image)

        # Display the processed image
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(1)

    def process_image(self, image):
        # Convert OpenCV image to tensor
        img_tensor = self.image_to_tensor(image)
        img_tensor = img_tensor.to(self.device)

        self.get_logger().info(f"Tensor shape before model input: {img_tensor.shape}")
        
        # Ensure the tensor is in the format [batch_size, channels, height, width]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.shape[1] != 3:  # if channels are not in the second dimension
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # Change HWC to CHW

        with torch.no_grad():
            
            # Model inference
            pred = self.model(img_tensor)[0]
            
            # Apply non-maximum suppression
            pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)[0]

            # Draw bounding boxes and labels on image
            for det in pred:
                if len(det):
                    # Rescale boxes from img_size to original size
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], image.shape).round()

                    for *xyxy, conf, cls in det:
                        label = f'{cls} {conf:.2f}'
                        plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)

        return image


    def image_to_tensor(self, image):
        # Convert an OpenCV image to a PyTorch tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).to(torch.float32)
        image /= 255.0  # normalize to [0, 1]
        image = image.permute(2, 0, 1)  # Change HWC to CHW
        image = image.unsqueeze(0)  # Add batch dimension
        return image

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
