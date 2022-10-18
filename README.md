# Minnesota Robotics Institute Preview
A sample of the scripts I have developed for my work with the Minnesota Robotics Institute and Interactive Robotics and Computer Vision Lab. My project focuses on enabling robots to identify Eurasian Watermilfoil, an invasive species in Minnesota lakes.

Image data not included in this repo. 

Images were annotated with Computer Vision Annotation Tool (CVAT), and exported with YOLO format.

Image_cropping.py
- Crops images according to yolo object detection coordinates. 

create_dataset.py
- Scales and crops images to the same size, then creates a dataset of image tensors and labels

ewm_conv_vae.py
- Applies convolutional Variational Autoencoder algorithm to dataset produced by create_dataset.py to generate synthetic images


