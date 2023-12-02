"""Usage - sources:
    $ python test.py --source video.mp4 --conf-thres 0.4 --iou-thres 0.5

"""
import ultralytics
import torch
import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from utils.general import (check_requirements, print_args)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

model = torch.hub.load(ROOT, 'custom', r"yolov5n.onnx", source='local')


def run(source, conf_thres=0.1, iou_thres=0.45):
    source = str(source)
    model.conf = conf_thres  # NMS confidence threshold
    model.iou = iou_thres  # NMS IoU threshold
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, size=640)
        results.render()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT /'video', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)