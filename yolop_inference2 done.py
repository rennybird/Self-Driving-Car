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
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        first_image_raw = next(iter(raw_image_generator), None)
        print("First image raw:", first_image_raw)

        for i, image_raw in enumerate(raw_image_generator):
            if i >= self.batch_size:
                break

            print(f"Received image {i} for processing. Shape: {image_raw.shape}")
            if len(image_raw.shape) != 3 or image_raw.shape[2] != 3:
                print(f"Skipping image {i} due to invalid format.")
                continue

            try:
                input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
                # ...

                # Ensure input_image is in CHW format before copying
                if input_image.shape == (self.input_h, self.input_w, 3):  # HWC format
                    input_image = np.transpose(input_image, (2, 0, 1))  # Convert to CHW format

                batch_image_raw.append(image_raw)
                batch_origin_h.append(origin_h)
                batch_origin_w.append(origin_w)

                print(f'Length of batch_origin_h: {len(batch_origin_h)}')
                print(f'Length of batch_origin_w: {len(batch_origin_w)}')

                np.copyto(batch_input_image[i], input_image)
                print('is it broken?2')

            except Exception as e:
                print(f"Error in processing image {i}: {e}")
                break
                
                # Ensure input_image is in CHW format before copying
                if input_image.shape == (self.input_h, self.input_w, 3):  # HWC format
                    input_image = np.transpose(input_image, (2, 0, 1))  # Convert to CHW format

                batch_image_raw.append(image_raw)
                batch_origin_h.append(origin_h)
                batch_origin_w.append(origin_w)
                np.copyto(batch_input_image[i], input_image)
                print('is it broken?2')

            except Exception as e:
                print(f"Error in processing image {i}: {e}")
                break

        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1

        detout = host_outputs[0]
        segout = host_outputs[1].reshape( (self.batch_size, self.img_h,self.img_w))
        laneout = host_outputs[2].reshape( (self.batch_size, self.img_h,self.img_w))

        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                detout[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )

            # Draw rectangles and labels on the original image
            img = batch_image_raw[i]
            nh = img.shape[0]
            nw = img.shape[1]
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                label="{}:{:.2f}".format( categories[int(result_classid[j])], result_scores[j])
                plot_one_box( box, img, label=label)

            seg  = cv2.resize(segout[i], (nw, nh), interpolation=cv2.INTER_NEAREST)
            lane = cv2.resize(laneout[i], (nw, nh), interpolation=cv2.INTER_NEAREST)
            color_area = np.zeros_like(img)
            color_area[seg==1]  = (0,255,0)
            color_area[lane==1] = (0,0,255)
            color_mask = np.mean(color_area, 2)
            img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_area[color_mask != 0] * 0.5
            img = img.astype(np.uint8)

        return batch_image_raw, end - start

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
        print(f"Image received in preprocess_image: Shape - {raw_bgr_image.shape}")
        if len(raw_bgr_image.shape) != 3 or raw_bgr_image.shape[2] != 3:
            print("Error in preprocess_image: Invalid image format")
            # Handle invalid image format
            empty_image = np.zeros((self.input_h, self.input_w, 3), dtype=np.float32)
            return empty_image, raw_bgr_image, 0, 0
            
        try:
            # Ensure the input image is in the expected format
            if len(raw_bgr_image.shape) == 3 and raw_bgr_image.shape[2] == 3:
                image_raw = raw_bgr_image
                h, w, c = image_raw.shape

                # Resize and pad the image to keep the aspect ratio and fit the expected model input size
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

                # Resize and pad
                image = cv2.resize(image_raw, (tw, th))
                image = cv2.copyMakeBorder(
                    image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )

                # Normalize the image
                image = image.astype(np.float32) / 255.0
                image = (image - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)

                # Change data layout from HWC to CHW
                image = np.transpose(image, [2, 0, 1])

                # Convert the image to row-major order
                image = np.ascontiguousarray(image)

                print(f"Preprocessed Image Shape: {image.shape}, Original Image Shape: {image_raw.shape}, Height: {h}, Width: {w}")
                return image, image_raw, h, w
            else:
                raise ValueError("Invalid image format")

        except Exception as e:
            print(f"Error in preprocess_image: {e}")
            # Return default values in case of error
            empty_image = np.zeros((self.input_h, self.input_w, 3), dtype=np.float32)
            return empty_image, raw_bgr_image, 0, 0


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
                batch_image_raw, use_time = yolop_wrapper.infer(yolop_wrapper.get_raw_image_zeros())
                print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))

            # Assuming you have a camera or ROS subscription to receive frames
            while True:
                # Replace this with your code to receive frames from ROS
                frame = receive_camera_frame_from_ros()  # Implement this function to receive frames from ROS
                yolop_wrapper.process_camera_frame(frame, categories)

        except Exception as e:
            print("Error:", str(e))

        finally:
            # Destroy the instance
            yolop_wrapper.destroy()