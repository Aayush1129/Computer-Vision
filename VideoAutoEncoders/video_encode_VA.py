
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from skimage import io, transform
import cv2

# In[2]:


# import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('GTK') 
# import matplotlib.pyplot as plt
# In[3]:


dir_path = os.getcwd()
folder = '/home/ayush/Frames'
constructed_img_path = '/home/ayush/constructed_frames_VA'
# print(os.listdir(path))


def mkfolder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

mkfolder(constructed_img_path)
# In[30]:


w = 224
h = 224
c = 3
max_batch = 20
n_hidden = 128
seq_max_len = 20
learning_rate = 0.0001
num_epoch = 100
logs_path = '/home/aodev/auto-encoder-video/Logs_VA'
model_path = '/home/aodev/auto-encoder-video/Model_VA/model.ckpt'

## Function for getting images in batch
def read_img_batch(file_list,batch_len,start):
    imgs = []
    for i in range(batch_len):
        img = io.imread(file_list[start+i])
    #   print(img)
        img = transform.resize(img, (w, h, c))
        imgs.append(img)
    return np.asarray(imgs, np.float32)


data = sorted(glob.glob(folder+'/*'))
data_len = len(data)
# print(data)
print(data_len)


tf.reset_default_graph()
##placeholders
x = tf.placeholder(tf.float32, shape = [None, w,h,c], name = 'input')
y = tf.placeholder(tf.float32, shape = [None, w,h,c], name = 'output')
seqlen = tf.placeholder(tf.int32, [None])
print(x)
print(y)
print(seqlen)


## Encoding Part ##
## Conv Layer
conv1 = tf.layers.conv2d(x, filters = 64, kernel_size = (5,5), strides = (3,3), activation=tf.nn.relu, name = 'conv_1', reuse = tf.AUTO_REUSE)
print(conv1)
pool1 = tf.layers.max_pooling2d(conv1, 2,2, name = 'pool_1')
print(conv1)
conv2 = tf.layers.conv2d(pool1, filters = 128, kernel_size = (5,5), strides = (3,3), activation=tf.nn.relu, name = 'conv_2', reuse = tf.AUTO_REUSE)
print(conv2)
pool2 = tf.layers.max_pooling2d(conv2, 2,2, name = 'pool_2')
print(pool2)
## Flatten
fl1 = tf.layers.flatten(pool2, name = 'fl_1')
print(fl1)
## Fully Connected
fc1 = tf.layers.dense(fl1, 1024, activation=tf.nn.relu, name = 'fc_1', reuse = tf.AUTO_REUSE)
print(fc1)
fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu, name = 'fc_2', reuse = tf.AUTO_REUSE)
print(fc2)


# In[13]:


fc_out = tf.reshape(fc2, shape = [1, -1, 256], name = 'fc_out')
print(fc_out)

# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
lstm_input = tf.unstack(fc_out, seq_max_len, 1)
# print(lstm_input)

## dynamic LSTM
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse = tf.AUTO_REUSE)
# print(lstm_cell)
outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, lstm_input, dtype=tf.float32, sequence_length=seqlen)
# print(outputs)
print(states)


# In[14]:


## Stacking outputs
outputs = tf.stack(outputs)
print(outputs)
outputs = tf.reshape(outputs, shape = [20, 128], name = 'lstm_output')
print(outputs)


# In[15]:


## Decoding Part ##
## Fully Connected
fc3 = tf.layers.dense(outputs, 256, activation=tf.nn.relu, name = 'fc_3', reuse = tf.AUTO_REUSE)
print(fc3)
fc4 = tf.layers.dense(fc3, 1024, activation=tf.nn.relu, name = 'fc_4', reuse = tf.AUTO_REUSE)
print(fc4)
fc5 = tf.layers.dense(fc4, 3200, activation=tf.nn.relu, name = 'fc_5', reuse = tf.AUTO_REUSE)
print(fc5)


# In[16]:


## Reshaping for conv Nets
reshape = tf.reshape(fc5, shape = [20, 5, 5, 128], name = 'reshape_1')
print(reshape)
'''
## Upsampling
upsample1 = tf.image.resize_images(reshape, size=(11,11), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
print(upsample1)
'''
## Upsampling
upsample1 = tf.layers.conv2d_transpose(reshape, filters = 128, kernel_size = 3, strides = 2, name = 'upsample1', reuse = tf.AUTO_REUSE)
print(upsample1)
## Conv Transpose layer
conv3 = tf.layers.conv2d_transpose(upsample1, filters = 64, kernel_size = (5,5), strides = (3,3), padding = 'valid', activation=tf.nn.relu, name = 'conv_3', reuse = tf.AUTO_REUSE)
print(conv3)

'''
## Upsampling
upsample2 = tf.image.resize_images(conv3, size=(74,74), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
print(upsample2)
'''
## Upsampling
upsample2 = tf.layers.conv2d_transpose(conv3, filters = 64, kernel_size = 3, strides = 2, name = 'upsample2', reuse = tf.AUTO_REUSE)
print(upsample2)

paddings = [[0,0],[1,2],[1,2],[0,0]]
pad1 = tf.pad(upsample2, paddings)
print(pad1)

## Conv Transpose layer
conv4 = tf.layers.conv2d_transpose(pad1, filters = 3, kernel_size = (5,5), strides = (3,3), name = 'conv_4', reuse = tf.AUTO_REUSE)
print(conv4)
y_pred = conv4



## Loss
loss = tf.reduce_mean(tf.pow(y-y_pred,2))*100000
print(loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
print(optimizer)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saving weights
saver = tf.train.Saver()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)



with tf.Session() as sess:
    sess.run(init)

    ##############Restore model weights from previously saved model############
    saver.restore(sess, model_path)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for ep in range(num_epoch):
        for i in range(int(len(data)/seq_max_len)):
            start = max_batch*i
            print('start: ', start)
            data_batch = read_img_batch(data,max_batch,start)
            # print(data_batch.shape)
            _, loss_ = sess.run([optimizer, loss], feed_dict = {x:data_batch, y:data_batch, seqlen : [20]})
            print('Epoch: ',ep, 'iteration: ',max_batch*i, 'Loss: ',loss_)
        print('Epoch: ',ep, 'Loss: ',loss_)
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
