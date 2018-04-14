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

* **binary transformation**:  If pixel value is greater than a threshold value (150), it is assigned one value (white), else it is assigned another value (black)

* **OCR**: Using pytesseract, it performs optical character recognition in the pre-processed image

