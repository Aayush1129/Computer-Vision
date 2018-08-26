import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt

dir_path = os.getcwd()
folder = '/home/ayush/Frames'
gen_image = '/home/ayush/gen_image'

def mkfolder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

mkfolder(gen_image)

w = 224
h = 224
c = 3
max_batch = 20
n_hidden = 128
seq_max_len = 20
learning_rate = 0.0001
num_epoch = 100
logs_path = '/home/ayush/Logs'
model_path = '/home/ayush/Model/model.ckpt'


## Final function for getting images in sorted order
def read_img(path):
    print(path)
    imgs = []
    for im in sorted(os.listdir(path)):
        print(im)
        img = io.imread(path+os.sep+im)
#       print(img)
        img = transform.resize(img, (w, h, c))
        imgs.append(img)
    return np.asarray(imgs, np.float32)

data = read_img(folder)
np.save('/home/aodev/auto-encoder-video/image_data.npy', data)
## saving the images numpy


# In[8]:
# Loading the image data
#data = np.load('/home/aodev/auto-encoder-video/image_data.npy')

print(data.shape)
# seqlen = tf.placeholder(tf.int32, [None])
seqlen = tf.constant(20,tf.int32)

sess = tf.Session()
## Restore weights
saver = tf.train.import_meta_graph('/home/aodev/auto-encoder-video/Model/model.ckpt.meta')
saver.restore(sess, '/home/aodev/auto-encoder-video/Model/model.ckpt')

########## Do not use       sess.run(tf.global_variables_initializer())       ###########
graph = tf.get_default_graph()

y = graph.get_tensor_by_name("output:0")                      ## output image
x =  graph.get_tensor_by_name("input:0")                      ## input image
pred = graph.get_tensor_by_name("conv_4/BiasAdd:0")           ## prediction (image) from the model
seqlen = graph.get_tensor_by_name("Placeholder:0")            ## Placeholder to provide seq len
# seqlen = tf.placeholder(dtype = tf.int32, shape = None)
lstm_output = graph.get_tensor_by_name("lstm_output:0")
print(x)
print(y)
print(pred)
print(seqlen)
print(lstm_output)

# for op in graph.get_operations()[700:1000]:
#     print str(op.name) 

canvas_recon = np.empty((224, 224, 3))

for i in range(int(len(data)/max_batch)):
    img_out, features = sess.run([pred, lstm_output], feed_dict={x:data[i*max_batch:(i+1)*max_batch], y:data[i*max_batch:(i+1)*max_batch], seqlen:[20]})
    print(img_out.shape)
    print('input max: ', np.max(data[i*max_batch:(i+1)*max_batch]))
    print('output max: ',np.max(img_out))
    print(features.shape)
    for j in range(img_out.shape[0]):
        canvas_recon = img_out[j:j+1,:,:,:]
        canvas_recon = np.reshape(canvas_recon, (224,224,3))
        canvas_recon = np.multiply(canvas_recon, 255)          ## To save the images
        canvas_recon = canvas_recon.astype(int)                ## converting into integers
        print('saved image: ', gen_image+os.sep+'Constructed_'+str(i*max_batch+j)+'.jpg')
        cv2.imwrite(gen_image+os.sep+'Constructed_'+str(i*max_batch+j)+'.jpg',canvas_recon)

