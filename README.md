# Supervised-Defect-Identification-from-the-Electronic-Device-Images

## Skills Employed

* Modelling Techniques: CNN Models - UNet, VGGNet, SegNet, convolutional autoencoder-decoder.

* Image Processing Techniques: Normalize RGB, Encode mask to RGB, Grayscale conversion.

* Tech Stack: Python, SQL.

* Libraries: Numpy, Pandas, Sklearn, Tensorflow, Keras, OpenCV.

## Introduction

In this project, we consider the problem of using supervised machine learning algorithms, in particular convolutional neural networks (CNN), for detect detection in the electronic device images (slider images). Our objective is to identify the type of defects from slider that are fed to our algorithms as either containing or not containing defects, and thereby, compare the efficiency of these models for the task of semantic segmentation. We consider images restricted to a size of 101x101 pixels with 3 color channels, containing either one or more types of defects of varying sizes and shapes. Such an approach to identifying defects panels from device images can have important applications in a variety of electronic industry, preeminent among which is its potential to reduce manual inspection cost in their factories.  In this report, we estimated the performance of various CNN models and achieved over 90% accuracy and 0.75 dice coefficient. In particular, we make use of the state of the art UNet architecture to achieve near perfect out-of-sample generalization performance on our slider images.

## Methodology

In this project, we employed supervised CNN autoencoder architecture. An example of the input image and the predicted mask is shown.
 ![pic](https://github.com/saha0073/Supervised-Defect-Identification-from-the-Electronic-Device-Images/blob/main/slider.PNG)
 
 ## Data
 
Our data for this project is comprised of two samples, which we designate as the training and test sets. The slider images act as the input and mask images are the target labels. The training dataset has 2100 images and test dataset has 450 images. The different color spots in the target masks represent different types of defects.
 
 ## Result
 
 We estimated the performance of various CNN models: UNet, VGGNet, and compared the performances. We find that UNet demonstrates better performance and  achieved over 90% accuracy, 0.75 dice coefficient. An image of the training process is shown below.
 ![pic](https://github.com/saha0073/Supervised-Defect-Identification-from-the-Electronic-Device-Images/blob/main/training.PNG)
 


## References

The following papers and literature were referred to, for execution and solution approach design.

* U-Net: Convolutional Networks for Biomedical Image Segmentation Olaf Ronneberger, Philipp Fischer, Thomas Brox 
* VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition Karen Simonyan, Andrew Zisserman

