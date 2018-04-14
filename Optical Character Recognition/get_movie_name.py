import cv2
import os
from PIL import Image
import pytesseract
import json

img_loc = "sample_input/LOGAN.jpg"
img_name = img_loc.split(os.sep)[-1]

image = cv2.imread(img_loc)                              # read the input image 
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)            # convert into Gray Scale
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) # Performing the binary threshold

cv2.imwrite('temp_folder/thres.jpg',thresh)              # saving the intermediate image for future reference

text = pytesseract.image_to_string(Image.open(r'temp_folder/thres.jpg'))
text = ''.join(e for e in text if e.isalnum())           # removing undesired special characer
movie_name = {}
movie_name[img_name] = text

with open('sample_output/data.json', 'w') as outfile:     # saving the output to json format
    json.dump(movie_name, outfile)


