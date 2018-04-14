 get movie name
==========

Prerequisites
------
* python==**2.7**
* pip==**9.0.1**
* pytesseract==**0.2.0**
* Pillow==**4.2.1**

<br><br/>

## Installation Instructions
```
bash build.sh
```

<br><br/>
## Code Preview

The following code performs the following:

* **convert image into gray-scale**: It converts the image(RGB) to grayscale
* **binary transformation**:  If pixel value is greater than a threshold value (150), it is assigned one value (white), else it is assigned another value (black).
* **OCR**: Using pytesseract, it performs optical character recognition in the pre-processed image











Process for predicting the location (input / output) for input images is as follows:

* **process_start**: This function will take the input image and predicts its location (indoor/outdoor).

* **load_labels**: This function returns the labels (indoor/outdoor) of 365 generic location in which the resnet model is trained.
    
* **hook_feature**: This function returns the list containing array (feature vector obtained from 
resnet model).

* **returnTF**: This function transforms (resize and normalize) the input images.

* **load_model**: This function loads the pre-trained resnet model.

* **get_predictions**: This function predicts the location (indoor/outdoor) of input image

<br><br/>
