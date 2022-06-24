<h1 align="center">DSND Capstone Project</h1>

# Project Title: Dog Breed Classifier 
 
## Motivation 

This repo contains capstone project files for [Data Scientist Nano Degree](https://www.udacity.com/course/data-scientist-nanodegree--nd025?) from Udacity. I choose this capstone project to learn to deploy machine leanring models on web. 


The goal of this project is to classifiy images of dogs according to their breed. The web UI accepts any user-specified image as input. If a dofg is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

This project also resembles a real-world applicaiton. There are three major algorithm in this project:
- The algorithm for detecting human faces in an image is based on OpenCV's implementation of Haar feature-based cascade classifiers. 
- The algorithm for detecting dog in an image is based on CNN ResNet-50 architecture.
- The algorithm for detecting dog breed in an image is based on CNN Xception architecture.

Finally, these detection algorithms are packaged into a web application. All necessary files to run this application are provided in this repo. 

For a complete problem definition, analysis and conclusion please refer to the Notebook attached in the main repo. 

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

## Summary 

A comparison of models used in this project is provided here:

| Model Name | Test Accuracy  | Epocs  | Batch Size |
| :---:   | :-: | :-: |:-:|
| CNN (scratch) | 3.8278% | 20 | 40 |
| VGG-16 | 52.3923% | 100 | 20 |
| Xception | 85.6459% | 100 | 100 |



