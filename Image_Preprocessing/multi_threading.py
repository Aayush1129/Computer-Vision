import os
import glob
import cv2
from scipy.misc import imsave
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import threading
import time
import pandas as pd

dir_path = os.getcwd()
if not os.path.exists(dir_path+os.sep+'data_log.csv'):
    df = pd.DataFrame(columns=['State1_files','num_s1','State2_files','num_s2','State3_files','num_s3'])
    df.to_csv('data_log.csv', index=False)

state_1 = dir_path+os.sep+'State_1'
state_2 = dir_path+os.sep+'State_2'
state_3 = dir_path+os.sep+'State_3'

def mkfolder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def data_frame():
    s1 = []
    for files in glob.glob(state_1+os.sep+'*'):
        s1.append(files.split(os.sep)[-1])
    num_s1 = len(s1)

    s2 = []
    for files in glob.glob(state_2+os.sep+'*'):
        s2.append(files.split(os.sep)[-1])
    num_s2 = len(s2)

    s3 = []
    for files in glob.glob(state_3+os.sep+'*'):
        s3.append(files.split(os.sep)[-1].split('_')[1])
    num_s3 = len(s3)

    df = pd.read_csv('data_log.csv')
    df = df.append({'State1_files':s1, 'num_s1':num_s1, 'State2_files':s2, 'num_s2':num_s2, 
                                'State3_files':s3, 'num_s3':num_s3}, ignore_index=True)
    df.to_csv('data_log.csv', index=False)


def preprocess(img_name, state2):
    image_name = img_name.split(os.sep)[-1]
    img = Image.open(img_name)
    #### Contrast
    enhancer = ImageEnhance.Contrast(img)
    sub_folder = state2+os.sep+image_name.split('.')[0]
    mkfolder(sub_folder)
    enhancer.enhance(1.25).save(sub_folder+os.sep+'contrast.jpg')
    #### Binarization
    #Pixels higher than this will be 1. Otherwise 0.
    THRESHOLD_VALUE = 127
    img = img.convert("L")
    imgData = np.asarray(img)
    thresholdedData = (imgData >= THRESHOLD_VALUE) * 1.0
    imsave(sub_folder+os.sep+'binarization.jpg', thresholdedData)
    #### Hough Transform
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150, apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    try:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
    except:
        pass
    cv2.imwrite(sub_folder+os.sep+'hough.jpg',img)

def merge_images(folders, state3):
    final_image = folders.split(os.sep)[-1]
    for image in glob.glob(folders+os.sep+'*.jpg'):
        if image.split(os.sep)[-1].split('.')[0]=='contrast':
            file1 = image
        elif image.split(os.sep)[-1].split('.')[0]=='binarization':
            file2 = image
        else:
            file3 = image
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image3 = Image.open(file3)

    (width1, height1) = image1.size
    new_width1 = width1/3

    image1 = image1.crop((0,0,new_width1,height1))
    image2 = image2.crop((new_width1,0,2*new_width1,height1))
    image3 = image3.crop((2*new_width1,0,3*new_width1,height1))

    result = Image.new('RGB', (width1, height1))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1/3, 0))
    result.paste(im=image3, box=(2*width1/3, 0))
    result.save(state3+os.sep+'processed_'+final_image+'.jpg')

start_time = time.time()
all_files = glob.glob(state_1+os.sep+'*.jpg')
all_files_names = [file.split(os.sep)[-1].split('.')[0] for file in all_files]
df = pd.read_csv('data_log.csv')
try:
    state_2_files = df.iloc[-1]['State2_files']
    state_2_files = [j for i,j in enumerate(state_2_files.split("'")) if i%2!=0]
except:
    state_2_files = []
final_list = [x for x in all_files_names if x not in state_2_files]
print(final_list)
try:
    thread_list1 = []
    for file in final_list:
        img_name = state_1+os.sep+file+'.jpg'
        single_thread = threading.Thread(target=preprocess,args = (img_name, state_2))
        thread_list1.append(single_thread)

    # starting the threads
    for single_thread in thread_list1:
        single_thread.start()

    # blocks the calling thread
    for single_thread in thread_list1:
        single_thread.join()

    end_time1 = time.time()
except KeyboardInterrupt:
    data_frame()

all_files = glob.glob(state_2+os.sep+'*')
all_files_names = [file.split(os.sep)[-1] for file in all_files]
df = pd.read_csv('data_log.csv')
try:
    state_3_files = df.iloc[-1]['State3_files']
    state_3_files = [j for i,j in enumerate(state_3_files.split("'")) if i%2!=0]
    state_3_files = [i.split('.')[0] for i in state_3_files]
except:
    state_3_files = []
final_list = [x for x in all_files_names if x not in state_3_files]

try:
    thread_list2 = []
    for file in final_list:
        mkfolder(state_3)
        folders = state_2+os.sep+file
        single_thread = threading.Thread(target=merge_images, args=(folders, state_3))
        thread_list2.append(single_thread)

    # starting the threads
    for single_thread in thread_list2:
        single_thread.start()

    # blocks the calling thread
    for single_thread in thread_list2:
        single_thread.join()
            
    end_time2 = time.time()
except KeyboardInterrupt:
    data_frame()

data_frame()
print('Time Taken for Task1: ', end_time1 - start_time)
print('Time Taken for Task2: ', end_time2 - end_time1)
print('Total Time: ', end_time2 - start_time)


