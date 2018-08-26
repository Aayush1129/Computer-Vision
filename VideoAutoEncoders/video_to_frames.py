

import os
import cv2
import glob
dir_path = os.getcwd()


### Temporary Video ###
video_path = '/home/ayush/Video'
img_path = '/home/ayush/Frames'

def mkfolder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

mkfolder(img_path)
for file in glob.glob(video_path+'/*'):
    print(file)
    vid_name = file.split('/')[-1].split('.')[0]
    print(vid_name)

    vidcap= cv2.VideoCapture(file)
    print(vidcap)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)
    frame_rate = float(vidcap.get(cv2.CAP_PROP_FPS))
    time = int(num_frames/frame_rate)
    print(time)

    ### For getting video of n fps
    n = 5
    count = 0
    # quality = video_path.split('/')[-1].split('.')[0]
    success,image=vidcap.read()
    success = True
    for count in range(0,time*n):
        vidcap.set(0,int(count*1000/n))
        success,image =vidcap.read()
        print('read a new frame: ',success)
        print('image name: ', img_path+'/'+ vid_name+'_frame%06d.jpg'%count)
        cv2.imwrite(img_path+'/'+ vid_name+'_frame%06d.jpg'%count,image)
        count+=1
        


### For getting video of 1 fps
'''
success = True
while success:
    success,image =vidcap.read()
    print('read a new frame: ',success)
    if success==True:
        cv2.imwrite(img_path+'/Game_5fps_epFinals_frame%d.jpg'%count, image)
        count=count+1
'''









