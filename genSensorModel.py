import cv2
import pandas as pd
import csv
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torchvision import ops
import numpy as np
from pathlib import Path
import ast
import os
import yaml

# Import necessary modules from yolov5 repository (if not already imported)
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class GenSensorModel(object):
    def __init__(self):
        # Load model parameters from configuration file
        with open("../dataset_params.yaml", "r") as stream:
            model_params = yaml.safe_load(stream)

        # Initialize model parameters
        self.conf_thres = model_params['confidence_threshold']
        self.iou_thres = model_params['iou_threshold']
        self.agnostic_nms = model_params['agnostic_nms']
        self.max_det = model_params['maximum_detections']
        self.line_thickness = model_params['line_thickness']
        self.weights = "best.pt"  # Update path as per your setup

        # Initialize YOLOv5 model
        self.device = select_device(str(model_params['device']))
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=model_params['dnn'], 
                                        data="data.yaml")  # Update path as per your setup
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )
        self.img_size = [model_params['inference_size_w'], model_params['inference_size_h']]
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.half = model_params['half']
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size), half=self.half)

    def run_model(self, im):
        im, im0 = self.preprocess(im)
        xyxy_return = []

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, None, self.agnostic_nms, max_det=self.max_det
        )

        # Process predictions 
        if pred[0] is not None:
            det = pred[0].cpu().numpy()
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if self.view_image:
                    c = int(cls)
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
                xyxy_return.append(xyxy)
            im0 = annotator.result()

        # Show image if view_image is True
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(0)

        # Return xyxy_return
        if len(xyxy_return) == 0:
            return None
        else:
            return xyxy_return
    
    def preprocess(self, img):
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        img = img[..., ::-1].transpose((0, 3, 1, 2)) 
        img = np.ascontiguousarray(img)
        return img, img0

