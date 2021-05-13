import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import datetime

import json
import requests
from centroidtracker import CentroidTracker
from numpy import random

import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(opt, save_img=False):
    ct = CentroidTracker()
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M:%S") # current time

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    
    model.to(device).eval()
    
# =============================================================================
    filepath_mask = 'D:/Internship Crime Detection/YOLOv5 person detection/AjnaTask/Mytracker/yolov5/weights/mask.pt'
        
    model_mask = torch.load(filepath_mask, map_location = device)['model'].float()
    model_mask.to(device).eval()
    if half:
        model_mask.half()
        
    names_m = model_mask.module.names if hasattr(model_mask, 'module') else model_mask.names
# =============================================================================
    
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = False
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    memory = {}
    people_counter = 0
    in_people = 0
    out_people = 0
    people_mask = 0
    people_none = 0
    time_sum = 0
    # now_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
# =============================================================================
        pred_mask = model_mask(img)[0]
# =============================================================================
        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
# =============================================================================
        pred_mask = non_max_suppression(pred_mask, 0.4, 0.5, classes = [0, 1, 2], agnostic = None)
        
        if pred_mask is None:
            continue
        classification = torch.cat(pred_mask)[:, -1]
        if len(classification) == 0:
            print("----",None)
            continue
        index = int(classification[0])
        
        mask_class = names_m[index]
        print("MASK CLASS>>>>>>> \n", mask_class)
# =============================================================================

        # Create the haar cascade
        # cascPath = "D:/Internship Crime Detection/YOLOv5 person detection/AjnaTask/Mytracker/haarcascade_frontalface_alt2.xml"
        # faceCascade = cv2.CascadeClassifier(cascPath)
        
        
        t2 = time_synchronized()
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            img_center_y = int(im0.shape[0]//2)
            # line = [(int(im0.shape[1]*0.258),int(img_center_y*1.3)),(int(im0.shape[1]*0.55),int(img_center_y*1.3))]
            # print("LINE>>>>>>>>>", line,"------------")
            # line = [(990, 672), (1072, 24)]
            line = [(1272, 892), (1800, 203)]
            #  [(330, 468), (704, 468)]
            print("LINE>>>>>>>>>", line,"------------")
            cv2.line(im0,line[0],line[1],(0,0,255),5)
            
# =============================================================================
#             gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
#             # Detect faces in the image
#             faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30)
#             )
#             # Draw a rectangle around the faces
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(im0, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 text_x = x
#                 text_y = y+h
#                 cv2.putText(im0, mask_class, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                                     1, (0, 0, 255), thickness=1, lineType=2)
# =============================================================================
        
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                bbox_xyxy = []
                rects = [] # Is it correct?

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    # label = f'{names[int(cls)]}'
                    xyxy_list = torch.tensor(xyxy).view(1,4).view(-1).tolist()
                    plot_one_box(xyxy, im0, label='person', color=colors[int(cls)], line_thickness=3)
                    rects.append(xyxy_list)
                    
                    obj = [x_c, y_c, bbox_w, bbox_h,int(cls)]
                    #cv2.circle(im0,(int(x_c),int(y_c)),color=(0,255,255),radius=12,thickness = 10)
                    bbox_xywh.append(obj)
                    # bbox_xyxy.append(rec)
                    confs.append([conf.item()])
                    


                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = ct.update(rects) # xyxy
                # outputs = deepsort.update(xywhs, confss, im0) # deepsort
                index_id = []
                previous = memory.copy()
                memory = {}
                boxes = []
                names_ls = []
                


                # draw boxes for visualization
                if len(outputs) > 0:
                    
                    # print('output len',len(outputs))
                    for id_,centroid in outputs.items():
                        # boxes.append([output[0],output[1],output[2],output[3]])
                        # index_id.append('{}-{}'.format(names_ls[-1],output[-2]))
                        index_id.append(id_)
                        boxes.append(centroid)
                        memory[index_id[-1]] = boxes[-1]

                    
                    i = int(0)
                    print(">>>>>>>",boxes)
                    for box in boxes:
                        # extract the bounding box coordinates
                        # (x, y) = (int(box[0]), int(box[1]))
                        # (w, h) = (int(box[2]), int(box[3]))
                        x = int(box[0])
                        y = int(box[1])
                        # GGG
                        if index_id[i] in previous:
                            previous_box = previous[index_id[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            # (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (x,y)
                            p1 = (x2,y2)
                            
                            cv2.line(im0, p0, p1, (0,255,0), 3) # current frame obj center point - before frame obj center point
                        
                            if intersect(p0, p1, line[0], line[1]):
                                people_counter += 1
                                print('==============================')
                                print(p0,"---------------------------",p0[1])
                                print('==============================')
                                print(line[1][1],'------------------',line[0][0],'-----------------', line[1][0],'-------------',line[0][1])
                                # if p0[1] <= line[1][1]:
                                #     in_people +=1
                    
                                
                                # else:
                                #     # if mask_class == 'mask':
                                #     #     print("COUNTING MASK..", mask_class)
                                #     #     people_mask += 1
                                #     # if mask_class == 'none':
                                #     #     people_none += 1
                                #     out_people +=1 
                                if p0[1] >= line[1][1]:
                                    in_people += 1
                                    if mask_class == 'mask':
                                        people_mask += 1
                                    else:
                                        people_none += 1
                                else:
                                    out_people += 1
                            

                        i += 1

                    
                        
                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                
            else:
                deepsort.increment_ages()
            cv2.putText(im0, 'Person [down][up] : [{}][{}]'.format(out_people,in_people),(130,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)
            cv2.putText(im0, 'Person [mask][no_mask] : [{}][{}]'.format(people_mask, people_none), (130,100),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)
            # Print time (inference + NMS)
            
            
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            time_sum += t2-t1
            
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    # im0= cv2.resize(im0,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(save_path, im0)
                else:
                    
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    with torch.no_grad():
        detect(args)