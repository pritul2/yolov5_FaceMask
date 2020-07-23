# yolov5_FaceMask
* The dataset used for training the yolov5 is from roboflow.ai
## Output result from testing dataset
![output_img](https://user-images.githubusercontent.com/41751718/88246529-6cd26d00-ccb8-11ea-8dc5-d6393c0b54cc.png)
## Installation
1) Download and install yolov5
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
git clone https://github.com/pritul2/yolov5_FaceMask
```
2) Run inference 
For running inference you required trained weights which is obtained from my repo cloned as yolov5_FaceMask<br/>

```
$ python detect.py --weights last_mask_yolov5s_results.pt --conf 0.4 --source 0  # webcam
                                                                              file.jpg  # image 
                                                                              file.mp4  # video
                                                                              path/  # directory
                                                                              path/*.jpg  # glob
                                                                              rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                                                                              http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```
## Increasing accuracy and Future Scope
The dataset contains 149 Images which is very less for yolo architecture. So during training I performed augmentation and increased to 298 Images.<br/>
To get more accuracy the training dataset needs to increase.<br/>

## Output Results from open source images
