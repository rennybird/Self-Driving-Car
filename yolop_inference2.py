# 2022/10/26 by ausk
"""
An example that uses TensorRT's Python api to make yolop inferences.
"""
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


CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
categories = ["car"]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    """
    tl = (line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label,  (c1[0], c1[1] - 2), 0,  tl / 3,
                    [225, 255, 255], thickness=tf,  lineType=cv2.LINE_AA)


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
        batch_input_image = np.empty(
            shape=[self.batch_size, 3, self.input_h, self.input_w])

        deviations = []

        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(
                image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())

        start = time.time()
        # Transfer input data to GPU
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

        context.execute_async(batch_size=self.batch_size, bindings=bindings,
                              stream_handle=stream.handle)  # Run inference

        for i in range(len(host_outputs)):
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)

        stream.synchronize()  # Synchronize the stream
        end = time.time()

        detout = host_outputs[0]  # Detection output
        segout = host_outputs[1].reshape(
            (self.batch_size, self.img_h, self.img_w))  # Segmentation output
        laneout = host_outputs[2].reshape(
            (self.batch_size, self.img_h, self.img_w))  # Lane segmentation output

        for i in range(self.batch_size):
            img = batch_image_raw[i]
            nh, nw = img.shape[:2]
            steering_instruction = "Error"

            # Define your custom region of interest (ROI)
            roi_x_start, roi_y_start = 100, 250  # Adjust these values as necessary
            roi_x_end, roi_y_end = 300, 400
            roi_color, thickness = (0, 255, 0), 2

            # Draw the ROI rectangle on the image for visual reference
            cv2.rectangle(img, (roi_x_start, roi_y_start),
                          (roi_x_end, roi_y_end), roi_color, thickness)

            # Resize the lane segmentation output for further processing
            lane = cv2.resize(laneout[i], (nw, nh),
                              interpolation=cv2.INTER_NEAREST)
            lane_line_x, lane_line_y = [], []

            # Process each row within the ROI to detect the lane line
            for y in range(roi_y_start, roi_y_end, 10):  # Adjust step as necessary
                x_indices = np.where(lane[y, roi_x_start:roi_x_end] == 1)[
                    0]  # Lane pixels in the ROI
                if len(x_indices) > 0:
                    # Adjust mean to global coordinates
                    x_mean = np.mean(x_indices).astype(int) + roi_x_start
                    lane_line_x.append(x_mean)
                    lane_line_y.append(y)

            # Fit a quadratic curve (2nd degree polynomial) if sufficient points are detected
            if len(lane_line_x) > 2:
                curve_fit = np.polyfit(lane_line_y, lane_line_x, 2)
                curve = np.poly1d(curve_fit)

                # Generate smooth lane line points
                smooth_y = np.linspace(min(lane_line_y), max(
                    lane_line_y), num=100)  # Adjust num for smoothness
                smooth_x = curve(smooth_y)

                # Draw the smooth lane line on the original image
                for x, y in zip(smooth_x, smooth_y):
                    cv2.circle(img, (int(x), int(y)), 2,
                               (0, 255, 255), -1)  # Yellow dots

                # Calculate the lane center at the bottom of the image
                # x-coordinate of the lane center at the bottom
                lane_center_x = smooth_x[-1]
                # Expected position of the left lane (e.g., a quarter of the frame's width)
                # Top center of the image
                imaginary_line_start = (self.img_w // 2, 0)
                imaginary_line_end = (self.img_w // 2, self.img_h)
                expected_lane_center_x = imaginary_line_end[0]

                # Calculate the deviation of the lane's bottom position from the expected position
                deviation = lane_center_x - expected_lane_center_x
                print(f"Deviation: {deviation}")
                
                lane_detected = True  # Assume lane is detected by default
                # Threshold for minimum detection points to consider a lane detected
                minimal_detection_points = 5

                # After detecting the lane and calculating the deviation
                if len(lane_line_x) < minimal_detection_points:
                    lane_detected = False

                if lane_detected:
                    # Determine steering recommendation based on the deviation
                    if deviation > -160:
                        # Deviation is less negative than -160, meaning it's to the right of the desired band
                        steering_recommendation = "Steer Left"
                        deviations.append(deviation)
                    elif deviation < -200:
                        # Deviation is more negative than -200, meaning it's to the left of the desired band
                        steering_recommendation = "Steer Right"
                        deviations.append(deviation)
                    else:
                        # Deviation is within the acceptable range (-160 to -200)
                        steering_recommendation = "Stay Straight"
                        deviations.append(deviation)
                else:
                    # Fallback logic for when the lane line is not confidently detected
                    steering_recommendation = "Stay Straight"
                    print("Lane not detected confidently, maintaining straight path.")

                # Overlay steering recommendation on the image
                # After drawing the smooth lane line on the original image
                for index, (x, y) in enumerate(zip(smooth_x, smooth_y)):
                    # Labeling start, middle, and end points of the lane line
                    if index == 0:  # Start point
                        label = "Start"
                    elif index == len(smooth_x) // 2:  # Middle point
                        label = "Middle"
                    elif index == len(smooth_x) - 1:  # End point
                        label = "End"
                    else:
                        label = None  # Only label specific points to avoid clutter

                    if label is not None:
                        coordinate_text = f"{label}: ({int(x)}, {int(y)})"
                        cv2.putText(img, coordinate_text, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 0), 1, cv2.LINE_AA)  # Adjusted text position for clarity

                # Labeling the lane center at the bottom of the image
                lane_center_label = "Lane Center"
                cv2.putText(img, f"{lane_center_label}: ({int(lane_center_x)}, {nh})", (
                    50, nh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Labeling the deviation and steering recommendation
                cv2.putText(img, f"Deviation: {deviation:.2f} pixels", (
                    50, nh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Steering: {steering_recommendation}", (
                    50, nh - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # Optionally, draw a line for the expected lane position and label it
                cv2.line(img, (int(expected_lane_center_x), 0),
                         (int(expected_lane_center_x), nh), (255, 0, 0), 2)
                # Red dot at the detected lane center
                cv2.circle(img, (int(lane_center_x), nh-1), 5, (0, 0, 255), -1)

                # Draw the expected lane center at the bottom of the image (for visualization)
                # Green dot at the expected lane center
                cv2.circle(img, (int(expected_lane_center_x), nh-1),
                           5, (0, 255, 0), -1)

                # Draw the deviation line
                cv2.line(img, (int(lane_center_x), nh-1),
                         (int(expected_lane_center_x), nh-1), (255, 255, 0), 2)
                deviation_left_boundary = -200  # More negative, meaning further to the left
                deviation_right_boundary = -160  # Less negative, meaning closer to the center

                # Calculate the x-coordinates of the deviation range boundaries
                left_boundary_x = expected_lane_center_x + deviation_left_boundary
                right_boundary_x = expected_lane_center_x + deviation_right_boundary

                # Draw lines representing the boundaries of the acceptable deviation range
                cv2.line(img, (left_boundary_x, 0), (left_boundary_x, nh),
                         (255, 0, 255), 2)  # Magenta line for left boundary
                # Magenta line for right boundary
                cv2.line(img, (right_boundary_x, 0),
                         (right_boundary_x, nh), (255, 0, 255), 2)
                if not lane_detected:
                    cv2.putText(img, "Lane detection confidence low, maintaining course",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.ctx.pop()  # Deactivate the context
        return batch_image_raw, end - start, deviations

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
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (
                114, 114, 114)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        image = (image - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
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
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - \
                (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - \
                (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - \
                (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - \
                (self.input_w - r_h * origin_w) / 2
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
        boxes = self.non_max_suppression(
            pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
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
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / \
                2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / \
                2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / \
                2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / \
                2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                              0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                              0], box2[:, 1], box2[:, 2], box2[:, 3]

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
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(
                boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def process_camera_frame(frame, yolop_wrapper, categories):
        frame = cv2.resize(frame, (yolop_wrapper.input_w, yolop_wrapper.input_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        frame = np.transpose(frame, [2, 0, 1])
        frame = np.expand_dims(frame, axis=0)
        frame = np.ascontiguousarray(frame)

        # Perform inference with the YOLOP model
        batch_image_raw, use_time = yolop_wrapper.infer([frame])

        # Process and display the results here
        for i, img in enumerate(batch_image_raw):
            cv2.imshow('YOLOP Output', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    if __name__ == "__main__":
        PLUGIN_LIBRARY = "build/libmyplugins.so"
        engine_file_path = "build/yolop.trt"
        categories = ["car"]

        ctypes.CDLL(PLUGIN_LIBRARY)

        # Initialize YolopTRT instance
        yolop_wrapper = YolopTRT(engine_file_path, categories)

        try:
            print('batch size is', yolop_wrapper.batch_size)

            for i in range(1):
                batch_image_raw, use_time = yolop_wrapper.infer(
                    yolop_wrapper.get_raw_image_zeros())
                print(
                    'warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))

            # Assuming you have a camera or ROS subscription to receive frames
            while True:
                # Replace this with your code to receive frames from ROS
                # Implement this function to receive frames from ROS
                frame = receive_camera_frame_from_ros()
                yolop_wrapper.process_camera_frame(frame, categories)

        except Exception as e:
            print("Error:", str(e))

        finally:
            # Destroy the instance
            yolop_wrapper.destroy()
