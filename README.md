[![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/github/pritul2/yolov5_FaceMask/blob/master/yolov5_train.ipynb)
<br/>
<i> Click on Train in Colab if .ipynb not opening </i>

# Description of Project

In this lock down situation, we decided to represent our idea on Personal Hygiene which is a contribution from our side to this pandemic of corona virus.

So, we have proposed AI based prototype in which we are detecting whether a Person has worn a Mask or not. Generally in shopping malls or near grocery shops or areas where the large group of people coming then it is difficult to monitor whether the person is wearing a mask or not. Also nowadays our government had made compulsory wear a mask else penalty is considered, then in that situation, our application will classify and tell whether the person had worn a mask or not and will make alert for an administrator.


# yolov5_FaceMask
* The dataset used for training the yolov5 is from roboflow.ai<br/>

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
![test5_out](https://user-images.githubusercontent.com/41751718/88254674-a8c6fb80-ccd3-11ea-8c09-54a7e39274f3.jpg)
![test4_out](https://user-images.githubusercontent.com/41751718/88254682-acf31900-ccd3-11ea-83b6-73659db53aa0.png)
![test3_out](https://user-images.githubusercontent.com/41751718/88254685-af557300-ccd3-11ea-9d2e-413c06820e5e.jpg)
![test2_out](https://user-images.githubusercontent.com/41751718/88254688-afee0980-ccd3-11ea-91fe-1f7591bb32c2.png)
![test1_out](https://user-images.githubusercontent.com/41751718/88254692-b11f3680-ccd3-11ea-9332-3a3428d40445.png)



-----------

## FaceMasque - Face Mask Classifier Package
This project is mainly made for detecting a person and classify weather a person is weared the mask or not.

To use this project simply follow following steps.

1) Use following command in your working directory through type it in command Prompt.
   pip install FaceMasque
 
2) Now, import it in your python file.
   import FaceMasque
 
3) Simply call folloeing function and you are ready with mask detection project. 
   detected_image = FaceMasque.mask_detection(original_image)
   
### The output is in below youtube link.    
https://youtu.be/CAJC8iWRqrk.   
 [![Person_with_mask](https://i.ytimg.com/vi/CAJC8iWRqrk/hqdefault.jpg?sqp=-oaymwEjCNACELwBSFryq4qpAxUIARUAAAAAGAElAADIQj0AgKJDeAE=&rs=AOn4CLBBlaQkGO8q7kj2NMuX4MRr1UB3-A)](https://youtu.be/CAJC8iWRqrk)

