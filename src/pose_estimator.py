import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
from PIL import Image
import sys
from pathlib import Path

# Import RTMPose preprocessing/postprocessing helpers from the local lib package.
try:
    from juxtapose.pose.rtmpose.preprocessing import bbox_xyxy2cs, top_down_affine
    from juxtapose.pose.rtmpose.postprocessing import get_simcc_maximum
    from juxtapose.utils.plotting import Annotator
except Exception:
    # Add local lib path if running from repo (lib/juxtapose/src)
    repo_root = Path(__file__).resolve().parents[1]
    lib_src = repo_root / 'lib' / 'juxtapose' / 'src'
    if str(lib_src) not in sys.path:
        sys.path.insert(0, str(lib_src))
    from juxtapose.pose.rtmpose.preprocessing import bbox_xyxy2cs, top_down_affine
    from juxtapose.pose.rtmpose.postprocessing import get_simcc_maximum
    from juxtapose.utils.plotting import Annotator

class PoseEstimator:
    def __init__(self, engine_path: str, conf_thres=0.7, iou_thres=0.45):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        self.input_shape = (1, 3, 480, 640)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # RTMPose model defaults (used by preprocessing/postprocessing helpers)
        # model_input_size is (width, height) as used in juxtapose RTMPose
        self.model_input_size = (192, 256)
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)

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

        print(f'Loaded RTMPose engine from {engine_path}')

    def _preprocess(self, img: np.ndarray, expected_shape):
        """Use RTMPose preprocessing helpers to prepare input for the engine.

        We follow the same logic as `juxtapose.pose.rtmpose.RTMPose.preprocess`:
        - compute center, scale from bbox (use full-image bbox when none provided)
        - apply top_down_affine to get resized image
        - normalize with mean/std
        - transpose to CHW and add batch dim
        """
        # expected_shape is (1,C,H,W) or (C,H,W)
        h, w = img.shape[:2]
        # use full-image bbox (x1,y1,x2,y2)
        bbox = np.array([0, 0, w, h], dtype=np.float32)

        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(self.model_input_size, scale, center, img)

        # normalize (same mean/std as RTMPose)
        if hasattr(self, 'mean') and self.mean is not None:
            mean = np.array(self.mean, dtype=np.float32)
            std = np.array(self.std, dtype=np.float32)
            resized_img = (resized_img - mean) / std

        # to CHW, float32, batch dim
        arr = resized_img.transpose(2, 0, 1).astype(np.float32)
        if len(expected_shape) == 4:
            arr = np.expand_dims(arr, 0)
        data = arr.ravel().astype(np.float32)
        # return data plus center/scale so postprocess can map keypoints back to image coords
        return data, center, scale
        

    def infer(self, img: np.ndarray):
        # assume single input
        input_name, input_shape, host_mem, device_mem = self.inputs[0]
        data_center_scale = self._preprocess(img, input_shape)
        # _preprocess returns (data, center, scale)
        if isinstance(data_center_scale, tuple):
            data, center, scale = data_center_scale
        else:
            data = data_center_scale
            center, scale = None, None
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

        # wait for all kernels/copies to finish
        self.stream.synchronize()

        return self._postprocess(outputs, center, scale)
    
    def _postprocess(self, outputs, center=None, scale=None):
        """Use RTMPose postprocessing helpers to decode outputs from engine.

        Expects `outputs` to correspond to RTMPose ONNX outputs: [simcc_x, simcc_y]
        If `center` and `scale` are provided (from preprocess), keypoints will be mapped back to image coords.
        Returns keypoints and scores in same format as `juxtapose.pose.rtmpose.RTMPose.postprocess`.
        """
        # outputs may be a list of arrays; try to find simcc_x, simcc_y
        if outputs is None or len(outputs) == 0:
            return np.array([]), np.array([])

        # convert any pagelocked buffers to numpy arrays
        simcc = [np.asarray(o).copy() for o in outputs]

        # depending on engine, simcc_x and simcc_y ordering may vary; assume same order as RTMPose: (simcc_x, simcc_y)
        if len(simcc) >= 2:
            simcc_x, simcc_y = simcc[0], simcc[1]
        else:
            # if a single output packs both, try to split along axis
            raise RuntimeError('Unexpected RTMPose engine outputs')

        simcc_x = np.array(simcc_x).astype(np.float32)
        simcc_y = np.array(simcc_y).astype(np.float32)

        # decode simcc (locs shape: (N, K, 2), scores: (N, K))
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / 2.0  # simcc_split_ratio default 2.0

        # if center/scale provided, rescale keypoints to original image coords
        if center is not None and scale is not None:
            # center: (2,) or (N,2), scale: (2,) or (N,2); model_input_size is (w, h)
            model_w, model_h = self.model_input_size
            # keypoints currently in [0..model_w/model_h * split_ratio] domain divided by simcc_split_ratio
            # Following RTMPose.postprocess:
            # keypoints = keypoints / model_input_size * scale
            # keypoints = keypoints + center - scale / 2

            # handle batch dimension
            kpt = keypoints.astype(np.float32)
            # keypoints shape (N, K, 2), model_input_size scalar divide
            kpt[..., 0] = kpt[..., 0] / model_w * scale[0]
            kpt[..., 1] = kpt[..., 1] / model_h * scale[1]

            kpt = kpt + center - (scale / 2.0)
            keypoints = kpt

        return keypoints, scores

if __name__ == '__main__':
    # Simple inference test using a cropped person image 'man.png'
    engine = os.path.join(os.path.dirname(__file__), '..', 'model', 'rtmpose-m.engine')
    det = PoseEstimator(engine)

    repo_root = Path(__file__).resolve().parents[1]
    man_path = "/home/olel/Projects/productivity_robot_backend/src/man.png"

    img = cv.imread(man_path)
    if img is None:
        print('Failed to read man.png')
        raise SystemExit(1)

    # run inference
    keypoints, scores = det.infer(img)
    print('Keypoints shape:', None if keypoints is None else np.shape(keypoints))
    print('Scores shape:', None if scores is None else np.shape(scores))

    # draw keypoints/skeletons using Juxtapose Annotator style
    out = img.copy()
    if keypoints is not None and keypoints.size != 0:
        annotator = Annotator()
        # Annotator expects kpts shape: (num_humans, 17, 2)
        annotator.draw_skeletons(out, keypoints)
        annotator.draw_kpts(out, keypoints)

    out_path = repo_root / 'man_annotated.png'
    cv.imwrite(str(out_path), out)
    print(f'Wrote annotated image to {out_path}')