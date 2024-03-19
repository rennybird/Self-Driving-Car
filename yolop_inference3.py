# 2022/10/26 by ausk
"""
An example that uses TensorRT's Python api to make yolop inferences.
"""

from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import ctypes
import os
import shutil
import random
import sys
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import numpy as np
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

# Define the function to capture frames from the webcam
# Define the function to capture frames from the webcam
def get_camera_frames(batch_size):
    # Use 0 or -1 to access the default camera, or replace with other numbers for different cameras
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return
    # Optionally set the resolution; depends on device compatibility
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    while True:
        batch = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break  # Break if no frames are returned, i.e., camera error
            batch.append(resized_frame)
        if len(batch) > 0:
            yield batch
        else:
            break

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    """
    tl = ( line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText( img, label,  (c1[0], c1[1] - 2), 0,  tl / 3, [225, 255, 255], thickness=tf,  lineType=cv2.LINE_AA)

host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []

class YolopTRT(object):
    """
    description: Warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, categories):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        self.categories = categories
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()


        for binding in engine:
            print('bingding: ', binding, engine.get_tensor_shape(binding))
            size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.input_h = 384
        self.input_w = 640
        self.img_h = 360
        self.img_w = 640

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size


    def infer(self, raw_image_generator):
        self.ctx.push()  # Make self the active context

        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        batch_image_raw = []
        steering_angles = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])

        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)

        np.copyto(host_inputs[0], batch_input_image.ravel())  # Copy input image to host buffer

        start = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)  # Transfer input data to GPU

        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)  # Run inference

        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)  # Transfer predictions back from GPU

        stream.synchronize()  # Synchronize the stream
        end = time.time()

        detout = host_outputs[0]  # Detection output
        segout = host_outputs[1].reshape((self.batch_size, self.img_h, self.img_w))  # Segmentation output
        laneout = host_outputs[2].reshape((self.batch_size, self.img_h, self.img_w))  # Lane segmentation output

        for i in range(self.batch_size):
            img = batch_image_raw[i]
            nh, nw = img.shape[:2]

            # Define your custom region of interest (ROI)
            # roi_x_start, roi_y_start = 100, 250  # Adjust these values as necessary
            # roi_x_end, roi_y_end = 300, 400
            roi_x_start, roi_y_start = 150, 250  # Adjust these values as necessary
            roi_x_end, roi_y_end = 300, 400
            roi_center_x = (roi_x_start + roi_x_end) // 2
            roi_color, thickness = (0, 255, 0), 2

            # Draw the ROI rectangle on the image for visual reference
            cv2.rectangle(img, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), roi_color, thickness)

            # Resize the lane segmentation output for further processing
            lane = cv2.resize(laneout[i], (nw, nh), interpolation=cv2.INTER_NEAREST)
            lane_line_x, lane_line_y = [], []
            x_range = None  # Initialize x_range to None or a default value
            y_ransac_pred = None  # Initialize y_ransac_pred to None or a default value
            

            # Process each row within the ROI to detect the lane line
            for y in range(roi_y_start, roi_y_end, 10):  # Adjust step as necessary
                x_indices = np.where(lane[y, roi_x_start:roi_x_end] == 1)[0]  # Lane pixels in the ROI
                if len(x_indices) > 0:
                    x_mean = np.mean(x_indices).astype(int) + roi_x_start  # Adjust mean to global coordinates
                    lane_line_x.append(x_mean)
                    lane_line_y.append(y)


            # Fit a quadratic curve (2nd degree polynomial) if sufficient points are detected
            if len(lane_line_x) > 2 and len(lane_line_y) > 2:
                X = np.array(lane_line_x).reshape(-1, 1)
                y = np.array(lane_line_y)

                min_samples_required = max(2, int(len(X) * 0.2))
                if len(X) >= min_samples_required:
                    ransac = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(min_samples=min_samples_required))
                    ransac.fit(X, y)

                    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
                    y_ransac_pred = ransac.predict(x_range)

                    # Calculate extended points for the RANSAC predicted lane line for visualization
                    # Define the extend_line function
                    def extend_line(x1, y1, x2, y2, length):
                        line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        x2_new = x2 + (x2 - x1) / line_len * length
                        y2_new = y2 + (y2 - y1) / line_len * length
                        return int(x2_new), int(y2_new)

                    # After RANSAC fitting and y_ransac_pred calculation
                    if x_range is not None and y_ransac_pred is not None:
                        x1, y1 = int(x_range[0][0]), int(y_ransac_pred[0])
                        x2, y2 = int(x_range[-1][0]), int(y_ransac_pred[-1])
                        x2_extended, y2_extended = extend_line(x1, y1, x2, y2, 500)  # Extend by 500 pixels for visualization

                        # Draw the extended RANSAC predicted lane line
                        cv2.line(img, (x1, y1), (x2_extended, y2_extended), (0, 255, 0), 2)

                        # Draw the imaginary line through the entire height of the image
                        cv2.line(img, (roi_center_x, 0), (roi_center_x, img.shape[0]), (255, 0, 0), 2)

                        # Calculate steering angle based on the extended line
                        delta_y = y2_extended - y1
                        delta_x = x2_extended - x1
                        angle_radians = np.arctan2(delta_y, delta_x - (roi_center_x - x1))
                        steering_angle_degrees = (np.degrees(angle_radians))
                        steering_angles.append(steering_angle_degrees)
                        y_intersect_approx = np.interp(roi_center_x, [x1, x2_extended], [y1, y2_extended])

                        cv2.circle(img, (roi_center_x, int(y_intersect_approx)), 5, (0, 0, 255), -1)

                        # Calculate the steering angle direction
                        steering_direction = "Right" if steering_angle_degrees > -70 else "Left" if steering_angle_degrees < -50 else "Straight"

                        # Display the steering angle and direction on the image
                        cv2.putText(img, f"Steering Angle: {steering_angle_degrees:.2f}°, Turn {steering_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                        # Optionally, add a line to visually represent the direction of the turn based on the steering angle
                        if steering_angle_degrees > 0:
                            # Turn Right: Draw a line to the right of the intersection point
                            cv2.line(img, (roi_center_x, int(y_intersect_approx)), (roi_center_x + 100, int(y_intersect_approx)), (0, 255, 255), 2)
                        elif steering_angle_degrees < 0:
                            # Turn Left: Draw a line to the left of the intersection point
                            cv2.line(img, (roi_center_x, int(y_intersect_approx)), (roi_center_x - 100, int(y_intersect_approx)), (0, 255, 255), 2)
                        # Display the steering angle on the image
                        cv2.putText(img, f"Steering Angle: {steering_angle_degrees:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Handle the case when RANSAC fitting is not possible
                        cv2.putText(img, "Insufficient data for RANSAC fitting", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        steering_angles.append(None)

            else:
                cv2.putText(img, "Not enough lane points", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.ctx.pop()  # Deactivate the context
        return batch_image_raw, steering_angles

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (114, 114, 114)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        image = (image - (0.485, 0.456, 0.406)) /(0.229, 0.224, 0.225)
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


if __name__ == "__main__":
    # Load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolop.trt"

    ctypes.CDLL(PLUGIN_LIBRARY)
    categories = ["car"]

    # Initialize YolopTRT instance
    yolop_wrapper = YolopTRT(engine_file_path)

    # Adjust output video settings if necessary
    output_video_path = "camera_output.avi"
    codec = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 30.0
    frame_size = (640, 360)
    out = cv2.VideoWriter(output_video_path, codec, frame_rate, frame_size)

    try:
        # Change here to use the camera feed
        for batch in get_camera_frames(yolop_wrapper.batch_size):
            batch_image_raw, use_time = yolop_wrapper.infer(batch)

            # Process and display the results here
            for img in batch_image_raw:
                out.write(img)
                cv2.imshow('YOLOP Output', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        yolop_wrapper.destroy()
        out.release()
        cv2.destroyAllWindows()

    print("done!")
