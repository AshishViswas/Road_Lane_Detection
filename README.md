# Road Lane Detection using U-NET    
Lane detection is a computer vision task that involves identifying the boundaries of driving lanes in an image or video of a road scene. It plays a vital role in various applications, particularly in the realm of autonomous vehicles and advanced driver-assistance systems (ADAS). Convolutional neural networks (CNNs) are now the dominant approach for lane detection. Trained on large datasets of road images, these models can achieve high accuracy even in challenging situations. This project is implemented using UNET architecture which is a deep learning algorithm widely used for image segmentation tasks.

# U-NET Model
U-Net is a powerful, versatile neural network architecture designed specifically for semantic segmentation tasks, which involve dividing an image into different meaningful regions. Lane detection in self-driving cars is a perfect example of a semantic segmentation task, where the goal is to accurately identify and segment the lanes in a road image. UNET has the ability to extract line features and the ability to extract context which improves the accuracy of lane lines. The experimental results show that the improved neural network can obtain good detection performance in complex lane lines, and effectively improve the accuracy and time-sensitives of lane lines.     

The Link to the paper on U-Net for road-lane detection: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/2102.04738)  

# U-Net Architecture

![unet_architecture](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/fcd07bc7-71dd-4618-ac5b-258218d868f7)

# TuSimple Dataset
The TuSimple dataset is a large-scale dataset for autonomous driving research, focusing on lane detection and perception tasks. It's widely used in computer vision and autonomous driving communities for benchmarking and developing algorithms.         

The TuSimple dataset consists of 6,408 road images on US highways. The resolution of image is 1280×720. The dataset is composed of 3,626 for training, 358 for validation, and 2,782 for testing called the TuSimple test set of which the images are under different weather conditions.           

# Downloads :
Download the Full Dataset Here:  https://www.kaggle.com/datasets/manideep1108/tusimple   

Checkout the Kaggle Link for this project :  https://www.kaggle.com/code/ashishviswas/road-lane-detection   

Kaggle Link for code to make predictions :  https://www.kaggle.com/code/ashishviswas/lane-predictor

# Getting Started:
To run this project you can download the road-lane-detection.ipynb file provided in the repository and the dataset from the download section and can implement the whole process by executing each cell in notebook in order. The requirements are specified in the requirements.txt file      

I choose Kaggle to implement this because it provides inbuilt GPU accelerator which accelerate the training process, I used GPU T4 x2 to implement this. You can also choose google colab to run this, google colab also provides inbuilt GPU accelerator which fast up the training process much faster that using CPU.       

# Model Training
For model tarining, I used GPU T4 x2 accelerator which accelerated my trained process much more faster than using CPU. In the training process, the training Epochs are set to 15, batch size is 32 and the process went well with higher accuracy and low loss.

# Model Testing
The trained model has been tested on test generator which gave out a good accuracy of 95.5% and loss of 0.0035.      

You can download the weights file UNET_LANE_DETECTOR_WEIGHTS.h5 file from the repository and directly use it for predictions on new images.

# GRAPHICAL VISUALIZATION OF METRICS

![training_validation_plot (2)](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/194b3fa1-e1c9-4444-921f-808bde4e36b5)

The Above graph visualizes the metrics during the training process, it shows Training & Validation Loss and Training & Validation Accuracy with the starting value and ending value. The graphs shows the loss function and accuracy remained fairly on training and validation sets throughout the training process as shown in the visualization.

# Model Predictions
The lane-predictor.ipynb file can be downloaded which contains code for lane prediction on images and video file's. Some sample images and video files were taken from internet and lane prediction's were obtained using the trained model. Few results are:     

![road1](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/e7d97943-49ba-465f-acc2-3ca5639a78df)     ![masked_image_1](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/033603cb-61cd-4974-9a22-60455eb53717)

![night_road_2](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/6adcce50-240f-4ff0-bd8b-9f9c566341a0)     ![masked_image_night_2](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/549d5fd2-cfb5-48d1-805b-4fae3ed895e4)      
    
![bridge_light](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/7408f7db-a2af-4a9c-bd98-484e4234dffa)     ![masked_image_bridge_light](https://github.com/AshishViswas/Road_Lane_Detection/assets/130546401/0b563e3d-4a71-4599-8bd7-85829848146f)

Even Though there are slight disturbances, The predictions came out pretty good when tested on images found on internet, which are not from the test set.

# Challenges:
The given code is succesfull in detecting lanes in both images and video files under varying lightning conditions, but the implementation for Real-Time detection requires a video camera attached to the front of vehicle to make predictions in real-time which could not be performed due to absence of necessary technical equipment but the code can be extended for real-time lane detection which is left to be explored in further time. 
