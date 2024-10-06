# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import cv2
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def segment_images2(
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    source=None,
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    device='',
    img_size=(180, 240),
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    conf_threshold=0.25,  # confidence threshold
    iou_threshold=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    classes=None,
    agnostic_nms=False,  # class-agnostic NMS
    half=False,  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
):
    if source is None:
        raise ValueError('The source of images file names must be specified.')

    img_paths = np.genfromtxt(source, dtype=str, delimiter=',')

    # load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    img_sz = check_img_size(img_size, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *img_sz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for img_path_pair in img_paths:
        depth_img_file_name = img_path_pair[0]
        rgb_img_file_name = img_path_pair[1]

        save_depth_path = os.path.dirname(depth_img_file_name)
        save_rgb_path = os.path.dirname(rgb_img_file_name)

        save_depth_path = os.path.join(save_depth_path, 'segmented')
        save_rgb_path = os.path.join(save_rgb_path, 'segmented')

        rgb_img_file_basename = os.path.basename(rgb_img_file_name)
        depth_img_file_basename = os.path.basename(depth_img_file_name)

        # load images
        depth_im0 = cv2.imread(depth_img_file_name)  # BGR
        rgb_im0 = cv2.imread(rgb_img_file_name)  # BGR

        assert depth_im0 is not None, f'Depth image Not Found {depth_im0}'
        assert rgb_im0 is not None, f'RGB image Not Found {rgb_im0}'

        rgb_im = letterbox(rgb_im0, img_sz, stride, auto=pt)[0]  # padded resize
        rgb_im = rgb_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        rgb_im = np.ascontiguousarray(rgb_im)  # contiguous

        depth_im = letterbox(depth_im0, img_sz, stride, auto=pt)[0]  # padded resize
        depth_im = depth_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        depth_im = np.ascontiguousarray(depth_im)  # contiguous

        with dt[0]:
            # torch version of the rbg image
            rgb_im = torch.from_numpy(rgb_im).to(device)
            rgb_im = rgb_im.half() if model.fp16 else rgb_im.float()  # uint8 to fp16/32
            # make a copy of the original image
            org_rgb_img = rgb_im.clone()
            rgb_im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(rgb_im.shape) == 3:
                rgb_im = rgb_im[None]  # expand for batch dim

            # torch version of the depth image
            depth_im = torch.from_numpy(depth_im).to(device)
            depth_im = depth_im.half() if model.fp16 else depth_im.float()  # uint8 to fp16/32

        with dt[1]:
            prediction, out = model(rgb_im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            prediction = non_max_suppression(prediction,
                                             conf_threshold,
                                             iou_threshold,
                                             classes,
                                             agnostic_nms,
                                             max_det=max_det,
                                             nm=32)

        # Process predictions
        for i, det in enumerate(prediction):
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], rgb_im.shape[2:], upsample=True)  # HWC
                from segmentor import Segmentor
                # print(rgb_img_file_basename)
                seg_img = Segmentor(org_rgb_img, masks[0:1]).remove_background()
                cv2.imwrite(os.path.join(save_rgb_path, rgb_img_file_basename), seg_img)
                seg_img = Segmentor(depth_im, masks[0:1]).remove_background()
                cv2.imwrite(os.path.join(save_depth_path, depth_img_file_basename), seg_img)

    vra = 0


@smart_inference_mode()
def segment_images(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        depth_source=ROOT / 'data/images/depth',
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        img_sz=(360, 480),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    img_sz = check_img_size(img_sz, s=stride)  # check image size

    # load images
    rgb_dataset = LoadImages(source, img_size=img_sz, stride=stride, auto=pt)
    depth_dataset = LoadImages(depth_source, img_size=img_sz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *img_sz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for (path, im, im0s, vid_cap, s), (d_path, d_im, d_im0s, d_vid_cap, d_s), in zip(rgb_dataset, depth_dataset):
        with dt[0]:
            # torch version of the rbg image
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # make a copy of the original image
            org_img = im.clone()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # torch version of the depth image
            d_im = torch.from_numpy(d_im).to(device)
            d_im = d_im.half() if model.fp16 else d_im.float()  # uint8 to fp16/32
            # make a copy of the original image
            d_org_img = d_im.clone()

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            prediction, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            prediction = non_max_suppression(prediction,
                                             conf_thres,
                                             iou_thres,
                                             classes,
                                             agnostic_nms,
                                             max_det=max_det,
                                             nm=32)

        # Process predictions
        for i, det in enumerate(prediction):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(rgb_dataset, 'frame', 0)
            p = Path(p)  # to Path
            d_p = Path(d_path)
            save_path = str(save_dir / p.name)  # im.jpg
            save_d_path = str(save_dir / d_p.name)
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                from segmentor import Segmentor
                seg_img = Segmentor(org_img, masks).remove_background()
                cv2.imwrite(save_path, seg_img)
                seg_img = Segmentor(d_org_img, masks).remove_background()
                cv2.imwrite(save_d_path, seg_img)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        depth_source=ROOT / 'data/images/depth',
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        img_sz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    img_sz = check_img_size(img_sz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_sz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=img_sz, stride=stride, auto=pt)
        depth_dataset = LoadImages(depth_source, img_size=img_sz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *img_sz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # for d_path, d_im, d_im0s, d_vid_cap, d_s in depth_dataset:
            #     d_im = torch.from_numpy(d_im).to(device)
            #     d_im = d_im.half() if model.fp16 else d_im.float()  # uint8 to fp16/32
            #     # make a copy of the original image
            #     org_d_img = d_im.clone()

            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # make a copy of the original image
            org_img = im.clone()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                # import segmentor as seg
                # seg.Segmentor(org_d_img, masks).remove_background()

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ----------------------------------------------------------------------------------------

                # Write results
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *img_sz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--depth_source', type=str, default=ROOT / 'data/images/depth', help='depth images')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--img_sz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    options = parser.parse_args()
    options.img_sz *= 2 if len(options.img_sz) == 1 else 1  # expand
    print_args(vars(options))
    return options


def main(options):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(options))


if __name__ == "__main__":
    seg_one = True
    if seg_one:
        # segment_images(ROOT / 'best_tiny_garsup_092024.pt', data=ROOT / 'data/custom.yaml')
        # opt = parse_opt()
        # main(opt)
        run(ROOT / 'best.pt')
    else:
        prefix = '/home/adriano/Desktop/avaltech/data_sets/segmentation/corsup/'
        img_path_file = 'nova_esperanca_img_to_segment.csv'
        segment_images2(ROOT / 'best.pt', prefix + img_path_file)

