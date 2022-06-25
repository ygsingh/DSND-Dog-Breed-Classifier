<h1 align="center">DSND Capstone Project</h1>

# Project Title: Dog Breed Classifier 
 

This repo contains capstone project files for [Data Scientist Nano Degree](https://www.udacity.com/course/data-scientist-nanodegree--nd025?) from Udacity. I choose this capstone project to learn to deploy machine leanring models on web. 

## Problem Introduction  


The goal of this project is to classifiy images of dogs according to their breed. The web UI accepts any user-specified image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

This project also resembles a real-world applicaiton. There are three major algorithm in this project:
- The algorithm for detecting human faces in an image is based on OpenCV's implementation of Haar feature-based cascade classifiers. 
- The algorithm for detecting dog in an image is based on CNN ResNet-50 architecture.
- The algorithm for detecting dog breed in an image is based on CNN Xception architecture.

Finally, these detection algorithms are packaged into a web application. All necessary files to run this application are provided in this repo. 

For a detailed steps on analysis and conclusion please refer to the Notebook attached in the main repo. 

## Strategy to solve the problem 

**Step 1: Gather data** 
 
We imported datasets of dog and human images. This data was made available in the Udacity's project workspace. 
  - There are 133 total dog categories.
  - There are 8351 total dog images.
  - There are 6680 training dog images.
  - There are 835 validation dog images.
  - There are 836 test dog images.
  - There are 13233 total human images.

**Step 2: Write a human detector** 
 
We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained dace detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). We have downloaded one of these detectors and stored it in the `haarcascades` directory.


**Step 3: Write a dog detector** 

We use a pre-trained ResNet-50 model to detect dogs in images. 

**Step 4a: Create a CNN to Classify Dog Breeds (from Scratch)** 

The following architecture was implemented to test the training and testing pipeline:

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 112, 112, 16)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 133)               8645      
=================================================================
Total params: 19,189
Trainable params: 19,189
Non-trainable params: 0
_________________________________________________________________

```

**Step 4b: Use a CNN to Classify Dog Breeds**

To reduce training time without sacrificing accuracy, we used transfer leaning. For understanding trasfer leaning we created a model that uses the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only added a global pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax. Here'e the architecture added to VGG-16.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_2 ( (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 133)               68229     
=================================================================
Total params: 68,229
Trainable params: 68,229
Non-trainable params: 0
_________________________________________________________________
```

**Step 5: Create a CNN to classify Dog Breeds (using Transfer Learning)**

In step 4b, we used trasnfer learning to create CNN using VGG-16 bottleneck features. In this step, we used bottleneck features from Xception pre-trained model. Once again these pre-trained weitghts were provided in Udacity's project workspace. Here's the architecture added to Xception model.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_3 ( (None, 2048)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 133)               272517    
=================================================================
Total params: 272,517
Trainable params: 272,517
Non-trainable params: 0
_________________________________________________________________
```

**Step 6: Create algorithm for human and dog images**

We created an algorithm that accpets a file path to an image and first determine whether the image contains a human, dog, or neither. Then,
* if a **dog** is detected in the image, return the predicted breed.
* if a **human** is detected in the image, return the resembling dog breed.
* if **neither** is detected in the image, provide output that indicates the same.

## Metrics

We are using accuracy to compare or select different models. More on this resutls sections. 

## Libraries Used

- Tensforflow
- Keras
- Numpy
- OpenCV
- Tqdm
- Sklearn
- Glob
- PIL (Pillow)
- Matplotlib

## Files 

```
.
├── README.md
├── app
│   ├── app.py                                   # Template file flash upload images
│   ├── dog_breed.py                             # Supporting files for ML models to identify dog breed
│   ├── haarcascades
│   │   └── haarcascade_frontalface_alt.xml      # OpenCV haarcascade data for detecting faces
│   ├── main.py                                  # Main file to run for flash application
│   ├── saved_models
│   │   └── weights.best.Xception.hdf5           # Xception model trained weights
│   ├── static
│   │   └── uploads
│   │       ├── Labrador_retriever_06457.jpg     # Uploaded test files
│   │       └── sample_human_2.png               # Uploaded test files
│   └── templates
│       └── upload.html                          # HTML tempalte for UI 
└── dog_app.ipynb                                # Jupyter notebook for project analysis 

```

## Instructions

```
python main.py
```

## Results 

**Test 1:** Testing Human Face Detector

* Percentage of the first 100 images in `human_files` have a detected human face:  100%
* Percentage of the first 100 images in `dog_files` have a detected human face:  11%

**Test 2:** Predictions with ResNet-50 dog detector

* Percentage of the first 100 images in `human_files` have a detected dog:  0%
* Percentage of the first 100 images in `dog_files`  have a detected dog:  100%

**Test 3:** Test CNN made from scratch on test dataset of dog images

Test accuracy: 3.8278%

**Test 4:** Test VGG-16 how well it identifies breed within test dataset of dog images

Test accuracy: 52.3923%

**Test 5:** Test Xception model on the test dataset of dog images

Test accuracy: 85.6459%

A comparison of models used in this project is provided here:

| Model Name | Test Accuracy  | Epocs  | Batch Size |
| :---:   | :-: | :-: |:-:|
| CNN (scratch) | 3.8278% | 20 | 40 |
| VGG-16 | 52.3923% | 100 | 20 |
| Xception | 85.6459% | 100 | 100 |


## Conclusion/Reflection/Improvements

Output is as expected. Xception is providing 85% accuracy which is better than all the other CNN models discussed in this project. However, we can improve the overall alogirthm in the following ways:

* First, use a CNN based algorithm for face detection instead of Haar Cascades. This will improve face detection in cases with occlusion, and side angle.

* Second, improve the accuracy of dog_breed detector by hyperparameter tuning. Epochs and batch sizes can be improved to get a better accuracy. 

* Data augmentation techniques can be implemeted to make a diverse/varied data set. 

* Thirdly, we can try a different CNN models such as FasterRCNN, etc to improve the accuracy further. These models are trained on coco dataset which is more extensive than imagenet.

