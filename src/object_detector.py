import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
from PIL import Image

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

to_detect = ['person', 'cell phone']

class ObjectDetector:
    def __init__(self, engine_path: str, conf_thres=0.7, iou_thres=0.45):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        self.input_shape = (1, 3, 480, 640)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('Failed to deserialize engine')
        self.context = self.engine.create_execution_context()

        # allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((name, shape, host_mem, device_mem))
            else:
                self.outputs.append((name, shape, host_mem, device_mem))

    def _preprocess(self, img: np.ndarray, expected_shape):
        self.img_height, self.img_width = img.shape[:2]
        # expected_shape: (1,C,H,W) or (C,H,W)
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] in (1,3,4):
            # HWC
            h, w, c = arr.shape
            C = expected_shape[1] if len(expected_shape) == 4 else expected_shape[0]
            H = expected_shape[2] if len(expected_shape) == 4 else expected_shape[1]
            W = expected_shape[3] if len(expected_shape) == 4 else expected_shape[2]
            if (h, w) != (H, W):
                pil = Image.fromarray((arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8))
                pil = pil.resize((W, H), Image.BILINEAR)
                arr = np.array(pil)
            # to CHW
            arr = arr.transpose(2, 0, 1).astype(np.float32)
        elif arr.ndim == 3 and arr.shape[0] in (1,3,4):
            # already CHW
            pass
        else:
            raise ValueError('Unsupported input shape')
        # add batch dim if needed
        if len(expected_shape) == 4:
            arr = np.expand_dims(arr, 0)
        return arr.ravel().astype(np.float32)

    def infer(self, img: np.ndarray):
        # assume single input
        input_name, input_shape, host_mem, device_mem = self.inputs[0]
        data = self._preprocess(img, input_shape)
        if data.size != host_mem.size:
            raise ValueError(f'Input size mismatch: {data.size} vs {host_mem.size}')
        np.copyto(host_mem, data)
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)

        # set addresses
        idx = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[idx])
            idx += 1

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        outputs = []
        for name, shape, host_o, dev_o in self.outputs:
            cuda.memcpy_dtoh_async(host_o, dev_o, self.stream)
            outputs.append(np.array(host_o).reshape(shape).copy())

        self.stream.synchronize()

        return self._postprocess(outputs)
    
    def _postprocess(self, outputs):
        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]
    
    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes
    
    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):

        unique_class_ids = np.unique(class_ids)

        keep_boxes = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices,:]
            class_scores = scores[class_indices]

            class_keep_boxes = self.nms(class_boxes, class_scores, iou_threshold)
            keep_boxes.extend(class_indices[class_keep_boxes])

        return keep_boxes

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

def annotate_image(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
    annotated = img.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_names[class_id] not in to_detect:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(annotated, f"{str(class_id)} {class_names[class_id]}: {str(score)}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated

if __name__ == '__main__':
    # Simple webcam loop: read frames, run inference, draw bounding boxes and class id, show
    engine = os.path.join(os.path.dirname(__file__), '..', 'model', 'yolov8m.engine')
    det = ObjectDetector(engine)

    cap = cv.VideoCapture("/dev/video0")
    if not cap.isOpened():
        print('Cannot open camera (index 0)')
        raise SystemExit(1)
    
    import time

    try:
        while True:
            now = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            inp = frame.astype(np.float32) / 255.0

            boxes, scores, class_ids = det.infer(inp)
            if len(boxes) == 0:
                continue
            print(f'Output shapes: {[boxes.shape, scores.shape, class_ids.shape]}')
            time_elapsed = time.time() - now
            print(f'Inference time: {time_elapsed:.3f} s')
            now = time.time()

            annotated = annotate_image(frame, boxes, scores, class_ids)
            cv.imshow('Object Detection', annotated)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            print(f'Frame total time: {(time.time() - now):.3f} s')
    finally:
        cap.release()
        cv.destroyAllWindows()