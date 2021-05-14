# Object tracker
 
## Environment

Windows 10: 

    Anaconda 3
 
    python 3.6.17
 
GPU:

    Nvidia GTX 940MX

## Description

Algorithm for calculating number of people wearing mask and not wearing mask. 

## Prerequisites

Download yolov5 weights from [here](https://github.com/ultralytics/yolov5) 

Download DeepSort weights from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) and place ckpt.t7 file under pytorch_deep_sort/deep_sort/deep/checkpoint/

## Execution

``` git clone https://github.com/HarshRangwala/DeepSort-Surveillance ```

``` cd DeepSort-Surveillance ```

``` pip3 install -r requirements.txt ```

To run the main algorithm. Execute the command:

``` python PersonCounter.py --source task_video.mp4 --device 0 --weights yolov5/weights/yolov5s.pt ```

